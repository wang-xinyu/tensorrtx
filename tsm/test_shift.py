import numpy as np
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
import tensorrt as trt
import torch
from numpy.testing import assert_array_almost_equal

INPUT_BLOB_NAME = 'input'
OUTPUT_BLOB_NAME = 'output'


def shift_mit(x, num_segments, shift_div=8):
    """Official temporal shift module.
    
    Code Reference: https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/temporal_shift.py # noqa
    Cannot convert to ONNX Model.
    """
    nt, c, h, w = x.size()
    n_batch = nt // num_segments
    x = x.view(n_batch, num_segments, c, h, w)

    fold = c // shift_div

    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
    out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]  # shift right
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

    return out.view(nt, c, h, w)


def shift_mmaction2(x, num_segments, shift_div=8):
    """MMAction2 temporal shift module.
    
    Code Reference: https://github.com/open-mmlab/mmaction2/blob/master/mmaction/models/backbones/resnet_tsm.py # noqa
    Could convert to ONNX Model.
    """
    # [N, C, H, W]
    n, c, h, w = x.size()

    # [N // num_segments, num_segments, C, H*W]
    # can't use 5 dimensional array on PPL2D backend for caffe
    x = x.view(-1, num_segments, c, h * w)

    # get shift fold
    fold = c // shift_div

    # split c channel into three parts:
    # left_split, mid_split, right_split
    left_split = x[:, :, :fold, :]
    mid_split = x[:, :, fold:2 * fold, :]
    right_split = x[:, :, 2 * fold:, :]

    # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
    # because array on caffe inference must be got by computing

    # shift left on num_segments channel in `left_split`
    zeros = left_split - left_split
    blank = zeros[:, :1, :, :]
    left_split = left_split[:, 1:, :, :]
    left_split = torch.cat((left_split, blank), 1)

    # shift right on num_segments channel in `mid_split`
    zeros = mid_split - mid_split
    blank = zeros[:, :1, :, :]
    mid_split = mid_split[:, :-1, :, :]
    mid_split = torch.cat((blank, mid_split), 1)

    # right_split: no shift

    # concatenate
    out = torch.cat((left_split, mid_split, right_split), 2)

    # [N, C, H, W]
    # restore the original dimension
    return out.view(n, c, h, w)


def _tensorrt_shift_module(network,
                           input,
                           num_segments=8,
                           shift_div=8,
                           input_shape=(16, 64, 32, 32)):
    """Temporal shift module implemented by TensorRT Network Definition API."""
    fold = input_shape[1] // shift_div
    batch_size = input_shape[0] // num_segments

    # reshape
    reshape = network.add_shuffle(input)
    assert reshape
    reshape.reshape_dims = (batch_size, num_segments) + tuple(input_shape[-3:])

    # left
    left_split = network.add_slice(reshape.get_output(0),
                                   start=(0, 1, 0, 0, 0),
                                   shape=(batch_size, num_segments - 1, fold,
                                          input_shape[2], input_shape[3]),
                                   stride=(1, 1, 1, 1, 1))
    assert left_split
    left_split_shape = (batch_size, 1, fold, input_shape[2], input_shape[3])
    left_blank = network.add_constant(shape=left_split_shape,
                                      weights=np.zeros(left_split_shape,
                                                       np.float32))
    assert left_blank
    left = network.add_concatenation(
        [left_split.get_output(0),
         left_blank.get_output(0)])
    assert left
    left.axis = 1

    # mid
    mid_split_shape = (batch_size, 1, fold, input_shape[2], input_shape[3])
    mid_blank = network.add_constant(shape=mid_split_shape,
                                     weights=np.zeros(mid_split_shape,
                                                      np.float32))
    assert mid_blank
    mid_split = network.add_slice(reshape.get_output(0),
                                  start=(0, 0, fold, 0, 0),
                                  shape=(batch_size, num_segments - 1, fold,
                                         input_shape[2], input_shape[3]),
                                  stride=(1, 1, 1, 1, 1))
    assert mid_split
    mid = network.add_concatenation(
        [mid_blank.get_output(0),
         mid_split.get_output(0)])
    assert mid
    mid.axis = 1

    # right
    right = network.add_slice(reshape.get_output(0),
                              start=(0, 0, 2 * fold, 0, 0),
                              shape=(batch_size, num_segments,
                                     input_shape[1] - 2 * fold, input_shape[2],
                                     input_shape[3]),
                              stride=(1, 1, 1, 1, 1))

    # concat
    concat = network.add_concatenation(
        [left.get_output(0),
         mid.get_output(0),
         right.get_output(0)])
    assert concat
    concat.axis = 2

    # reshape
    reshape2 = network.add_shuffle(concat.get_output(0))
    assert reshape2
    reshape2.reshape_dims = input_shape
    return reshape2


def shift_tensorrt(x, num_segments, shift_div, input_shape):
    """Test TensorRT temporal shift module."""
    assert isinstance(x, np.ndarray)

    gLogger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(gLogger)
    config = builder.create_builder_config()

    # create engine
    explicit_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_flag)
    input = network.add_input(INPUT_BLOB_NAME, trt.float32, input_shape)
    assert input
    output = _tensorrt_shift_module(network,
                                    input,
                                    num_segments=num_segments,
                                    shift_div=shift_div,
                                    input_shape=input_shape)
    assert output

    # generate engine by builder/network/config
    output.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(output.get_output(0))
    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 20
    engine = builder.build_engine(network, config)
    del network
    assert engine.num_bindings == 2, f'{engine.num_bindings}'
    context = engine.create_execution_context()

    # buffer
    host_in = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    np.copyto(host_in, x.ravel())
    host_out = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    devide_in = cuda.mem_alloc(host_in.nbytes)
    devide_out = cuda.mem_alloc(host_out.nbytes)
    bindings = [int(devide_in), int(devide_out)]
    stream = cuda.Stream()

    # do inference
    cuda.memcpy_htod_async(devide_in, host_in, stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_out, devide_out, stream)
    stream.synchronize()

    return np.array(host_out.reshape(*input_shape))


if __name__ == '__main__':
    INPUT_SHAPE = (16, 64, 32, 32)
    assert len(INPUT_SHAPE) == 4
    NUM_SEGMENTS = 8
    SHIFT_DIV = 8

    # inference
    inputs = np.random.rand(*INPUT_SHAPE).astype(np.float32)
    inputs_pytorch = torch.tensor(inputs)
    with torch.no_grad():
        rmit = shift_mit(inputs_pytorch, NUM_SEGMENTS, SHIFT_DIV).numpy()
        rmmaction2 = shift_mmaction2(inputs_pytorch, NUM_SEGMENTS,
                                     SHIFT_DIV).numpy()
    rtensorrt = shift_tensorrt(inputs, NUM_SEGMENTS, SHIFT_DIV, INPUT_SHAPE)

    # test results
    assert_array_almost_equal(rmit, rtensorrt)
    assert_array_almost_equal(rmmaction2, rtensorrt)
    print("Tests PASSED")
