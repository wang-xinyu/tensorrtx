import argparse
import os
import struct

import numpy as np
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
import tensorrt as trt

BATCH_SIZE = 1
NUM_SEGMENTS = 8
INPUT_H = 224
INPUT_W = 224
OUTPUT_SIZE = 400
SHIFT_DIV = 8

assert INPUT_H % 32 == 0 and INPUT_W % 32 == 0, \
    "Input height and width should be a multiple of 32."

EPS = 1e-5
INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_weights(file):
    print(f"Loading weights: {file}")

    assert os.path.exists(file), f'Unable to load weight file {file}'

    weight_map = {}
    with open(file, "r") as f:
        lines = [line.strip() for line in f]
    count = int(lines[0])
    assert count == len(lines) - 1
    for i in range(1, count + 1):
        splits = lines[i].split(" ")
        name = splits[0]
        cur_count = int(splits[1])
        assert cur_count + 2 == len(splits)
        values = []
        for j in range(2, len(splits)):
            # hex string to bytes to float
            values.append(struct.unpack(">f", bytes.fromhex(splits[j])))
        weight_map[name] = np.array(values, dtype=np.float32)

    return weight_map


def add_shift_module(network, input, input_shape, num_segments=8, shift_div=8):
    fold = input_shape[1] // shift_div

    # left
    left_split = network.add_slice(input,
                                   start=(1, 0, 0, 0),
                                   shape=(num_segments - 1, fold,
                                          input_shape[2], input_shape[3]),
                                   stride=(1, 1, 1, 1))
    assert left_split
    left_split_shape = (1, fold, input_shape[2], input_shape[3])
    left_blank = network.add_constant(shape=left_split_shape,
                                      weights=np.zeros(left_split_shape,
                                                       np.float32))
    assert left_blank
    left = network.add_concatenation(
        [left_split.get_output(0),
         left_blank.get_output(0)])
    assert left
    left.axis = 0

    # mid
    mid_split_shape = (1, fold, input_shape[2], input_shape[3])
    mid_blank = network.add_constant(shape=mid_split_shape,
                                     weights=np.zeros(mid_split_shape,
                                                      np.float32))
    assert mid_blank
    mid_split = network.add_slice(input,
                                  start=(0, fold, 0, 0),
                                  shape=(num_segments - 1, fold,
                                         input_shape[2], input_shape[3]),
                                  stride=(1, 1, 1, 1))
    assert mid_split
    mid = network.add_concatenation(
        [mid_blank.get_output(0),
         mid_split.get_output(0)])
    assert mid
    mid.axis = 0

    # right
    right = network.add_slice(input,
                              start=(0, 2 * fold, 0, 0),
                              shape=(num_segments, input_shape[1] - 2 * fold,
                                     input_shape[2], input_shape[3]),
                              stride=(1, 1, 1, 1))

    # concat left mid right
    output = network.add_concatenation(
        [left.get_output(0),
         mid.get_output(0),
         right.get_output(0)])
    assert output
    output.axis = 1
    return output


def add_batch_norm_2d(network, weight_map, input, layer_name, eps):
    gamma = weight_map[layer_name + ".weight"]
    beta = weight_map[layer_name + ".bias"]
    mean = weight_map[layer_name + ".running_mean"]
    var = weight_map[layer_name + ".running_var"]
    var = np.sqrt(var + eps)

    scale = gamma / var
    shift = -mean / var * gamma + beta
    return network.add_scale(input=input,
                             mode=trt.ScaleMode.CHANNEL,
                             shift=shift,
                             scale=scale)


def bottleneck(network, weight_map, input, in_channels, out_channels, stride,
               layer_name, input_shape):
    shift = add_shift_module(network, input, input_shape, NUM_SEGMENTS,
                             SHIFT_DIV)
    assert shift

    conv1 = network.add_convolution(input=shift.get_output(0),
                                    num_output_maps=out_channels,
                                    kernel_shape=(1, 1),
                                    kernel=weight_map[layer_name +
                                                      "conv1.weight"],
                                    bias=trt.Weights())
    assert conv1

    bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0),
                            layer_name + "bn1", EPS)
    assert bn1

    relu1 = network.add_activation(bn1.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu1

    conv2 = network.add_convolution(input=relu1.get_output(0),
                                    num_output_maps=out_channels,
                                    kernel_shape=(3, 3),
                                    kernel=weight_map[layer_name +
                                                      "conv2.weight"],
                                    bias=trt.Weights())
    assert conv2
    conv2.stride = (stride, stride)
    conv2.padding = (1, 1)

    bn2 = add_batch_norm_2d(network, weight_map, conv2.get_output(0),
                            layer_name + "bn2", EPS)
    assert bn2

    relu2 = network.add_activation(bn2.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu2

    conv3 = network.add_convolution(input=relu2.get_output(0),
                                    num_output_maps=out_channels * 4,
                                    kernel_shape=(1, 1),
                                    kernel=weight_map[layer_name +
                                                      "conv3.weight"],
                                    bias=trt.Weights())
    assert conv3

    bn3 = add_batch_norm_2d(network, weight_map, conv3.get_output(0),
                            layer_name + "bn3", EPS)
    assert bn3

    if stride != 1 or in_channels != 4 * out_channels:
        conv4 = network.add_convolution(
            input=input,
            num_output_maps=out_channels * 4,
            kernel_shape=(1, 1),
            kernel=weight_map[layer_name + "downsample.0.weight"],
            bias=trt.Weights())
        assert conv4
        conv4.stride = (stride, stride)

        bn4 = add_batch_norm_2d(network, weight_map, conv4.get_output(0),
                                layer_name + "downsample.1", EPS)
        assert bn4

        ew1 = network.add_elementwise(bn4.get_output(0), bn3.get_output(0),
                                      trt.ElementWiseOperation.SUM)
    else:
        ew1 = network.add_elementwise(input, bn3.get_output(0),
                                      trt.ElementWiseOperation.SUM)
    assert ew1

    relu3 = network.add_activation(ew1.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu3

    return relu3


def create_engine(maxBatchSize, builder, dt, weights):
    weight_map = load_weights(weights)
    network = builder.create_network()

    data = network.add_input(INPUT_BLOB_NAME, dt,
                             (NUM_SEGMENTS, 3, INPUT_H, INPUT_W))
    assert data

    conv1 = network.add_convolution(input=data,
                                    num_output_maps=64,
                                    kernel_shape=(7, 7),
                                    kernel=weight_map["conv1.weight"],
                                    bias=trt.Weights())
    assert conv1
    conv1.stride = (2, 2)
    conv1.padding = (3, 3)

    bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), "bn1",
                            EPS)
    assert bn1

    relu1 = network.add_activation(bn1.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu1

    pool1 = network.add_pooling(input=relu1.get_output(0),
                                window_size=trt.DimsHW(3, 3),
                                type=trt.PoolingType.MAX)
    assert pool1
    pool1.stride = (2, 2)
    pool1.padding = (1, 1)

    cur_height = INPUT_H // 4
    cur_width = INPUT_W // 4
    x = bottleneck(network, weight_map, pool1.get_output(0), 64, 64, 1,
                   "layer1.0.", (NUM_SEGMENTS, 64, cur_height, cur_width))
    x = bottleneck(network, weight_map, x.get_output(0), 256, 64, 1,
                   "layer1.1.", (NUM_SEGMENTS, 256, cur_height, cur_width))
    x = bottleneck(network, weight_map, x.get_output(0), 256, 64, 1,
                   "layer1.2.", (NUM_SEGMENTS, 256, cur_height, cur_width))

    x = bottleneck(network, weight_map, x.get_output(0), 256, 128, 2,
                   "layer2.0.", (NUM_SEGMENTS, 256, cur_height, cur_width))
    cur_height = INPUT_H // 8
    cur_width = INPUT_W // 8
    x = bottleneck(network, weight_map, x.get_output(0), 512, 128, 1,
                   "layer2.1.", (NUM_SEGMENTS, 512, cur_height, cur_width))
    x = bottleneck(network, weight_map, x.get_output(0), 512, 128, 1,
                   "layer2.2.", (NUM_SEGMENTS, 512, cur_height, cur_width))
    x = bottleneck(network, weight_map, x.get_output(0), 512, 128, 1,
                   "layer2.3.", (NUM_SEGMENTS, 512, cur_height, cur_width))
    x = bottleneck(network, weight_map, x.get_output(0), 512, 256, 2,
                   "layer3.0.", (NUM_SEGMENTS, 512, cur_height, cur_width))
    cur_height = INPUT_H // 16
    cur_width = INPUT_W // 16
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.1.", (NUM_SEGMENTS, 1024, cur_height, cur_width))
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.2.", (NUM_SEGMENTS, 1024, cur_height, cur_width))
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.3.", (NUM_SEGMENTS, 1024, cur_height, cur_width))
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.4.", (NUM_SEGMENTS, 1024, cur_height, cur_width))
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.5.", (NUM_SEGMENTS, 1024, cur_height, cur_width))

    x = bottleneck(network, weight_map, x.get_output(0), 1024, 512, 2,
                   "layer4.0.", (NUM_SEGMENTS, 1024, cur_height, cur_width))
    cur_height = INPUT_H // 32
    cur_width = INPUT_W // 32
    x = bottleneck(network, weight_map, x.get_output(0), 2048, 512, 1,
                   "layer4.1.", (NUM_SEGMENTS, 2048, cur_height, cur_width))
    x = bottleneck(network, weight_map, x.get_output(0), 2048, 512, 1,
                   "layer4.2.", (NUM_SEGMENTS, 2048, cur_height, cur_width))

    pool2 = network.add_pooling(x.get_output(0),
                                window_size=trt.DimsHW(cur_height, cur_width),
                                type=trt.PoolingType.AVERAGE)
    assert pool2
    pool2.stride = (1, 1)

    fc1 = network.add_fully_connected(input=pool2.get_output(0),
                                      num_outputs=OUTPUT_SIZE,
                                      kernel=weight_map['fc.weight'],
                                      bias=weight_map['fc.bias'])
    assert fc1

    reshape = network.add_shuffle(fc1.get_output(0))
    assert reshape
    reshape.reshape_dims = (NUM_SEGMENTS, OUTPUT_SIZE)

    reduce = network.add_reduce(reshape.get_output(0),
                                op=trt.ReduceOperation.AVG,
                                axes=1,
                                keep_dims=False)
    assert reduce

    softmax = network.add_softmax(reduce.get_output(0))
    assert softmax
    softmax.axes = 1

    softmax.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(softmax.get_output(0))

    # Build engine
    builder.max_batch_size = maxBatchSize
    builder.max_workspace_size = 1 << 20
    engine = builder.build_cuda_engine(network)

    del network
    del weight_map

    return engine


def do_inference(context, host_in, host_out, batchSize):
    devide_in = cuda.mem_alloc(host_in.nbytes)
    devide_out = cuda.mem_alloc(host_out.nbytes)
    bindings = [int(devide_in), int(devide_out)]
    stream = cuda.Stream()

    cuda.memcpy_htod_async(devide_in, host_in, stream)
    context.execute_async(batch_size=batchSize,
                          bindings=bindings,
                          stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_out, devide_out, stream)
    stream.synchronize()


def inference_mmaction2(inputs, config, checkpoint):
    import torch
    from mmaction.models import build_model
    from mmcv import Config
    from mmcv.runner import load_checkpoint

    cfg = Config.fromfile(config)
    cfg.model.backbone.pretrained = None
    model = build_model(cfg.model,
                        train_cfg=None,
                        test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.eval()
    inputs = torch.tensor(inputs)
    with torch.no_grad():
        return model(return_loss=False, imgs=inputs)


def main(args):
    assert not (args.save_engine_path and args.load_engine_path)

    if args.load_engine_path:
        # load from local file
        runtime = trt.Runtime(TRT_LOGGER)
        assert runtime
        with open(args.load_engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    else:
        # Create network and engine
        assert args.tensorrt_weights
        builder = trt.Builder(TRT_LOGGER)
        engine = create_engine(BATCH_SIZE, builder, trt.float32,
                               args.tensorrt_weights)
    assert engine
    assert engine.num_bindings == 2

    if args.save_engine_path is not None:
        # save engine to local file
        with open(args.save_engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"{args.save_engine_path} Generated successfully.")

    context = engine.create_execution_context()
    assert context

    host_in = cuda.pagelocked_empty(BATCH_SIZE * NUM_SEGMENTS * 3 * INPUT_H *
                                    INPUT_W,
                                    dtype=np.float32)
    host_out = cuda.pagelocked_empty(BATCH_SIZE * OUTPUT_SIZE,
                                     dtype=np.float32)

    if args.test_mmaction2:
        assert args.mmaction2_config and args.mmaction2_checkpoint, \
            "MMAction2 config and checkpoint couldn't be None"

        data = np.random.randn(BATCH_SIZE, NUM_SEGMENTS, 3, INPUT_H,
                               INPUT_W).astype(np.float32)

        # TensorRT inference
        np.copyto(host_in, data.ravel())
        do_inference(context, host_in, host_out, BATCH_SIZE)

        # pytorch inference
        pytorch_results = inference_mmaction2(data, args.mmaction2_config,
                                              args.mmaction2_checkpoint)

        # test
        from numpy.testing import assert_array_almost_equal
        assert_array_almost_equal(host_out.reshape(-1),
                                  pytorch_results.reshape(-1),
                                  decimal=4)
        print("MMAction2 TEST PASSED")

    if args.test_cpp:
        assert args.cpp_result_path, "Should set --cpp-result-path"
        assert os.path.exists(args.cpp_result_path),\
            f"{args.cpp_result} doesn't exist"

        # C++ API fixed inputs
        inputs = np.ones((BATCH_SIZE, NUM_SEGMENTS, 3, INPUT_H, INPUT_W),
                         dtype=np.float32)

        # TensorRT inference
        np.copyto(host_in, inputs.ravel())
        do_inference(context, host_in, host_out, BATCH_SIZE)

        # Read cpp inference results
        with open(args.cpp_result_path, "r") as f:
            data = f.read().strip()
        cpp_results = np.array([float(d)
                                for d in data.split(" ")]).astype(np.float32)

        # test
        from numpy.testing import assert_array_almost_equal
        assert_array_almost_equal(host_out.reshape(-1),
                                  cpp_results.reshape(-1),
                                  decimal=4)
        print("CPP TEST PASSED")

    if args.input_video:
        # Get ONE prediction result from ONE video
        # Use demo.mp4 from MMAction2
        import cv2

        # get selected frame id of uniform sampling
        cap = cv2.VideoCapture(args.input_video)
        sample_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        avg_interval = sample_length / float(NUM_SEGMENTS)
        base_offsets = np.arange(NUM_SEGMENTS) * avg_interval
        clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int32)

        # read frames
        frames = []
        for i in range(max(clip_offsets) + 1):
            flag, frame = cap.read()
            if i in clip_offsets:
                frames.append(cv2.resize(frame, (INPUT_W, INPUT_W)))
        frames = np.array(frames)

        # preprocessing frames
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        frames = (frames - mean) / std
        frames = frames.transpose([0, 3, 1, 2])

        # TensorRT inference
        np.copyto(host_in, frames.ravel())
        do_inference(context, host_in, host_out, BATCH_SIZE)
        # For demo.mp4, should be 6, aka arm wrestling
        class_id = np.argmax(host_out.reshape(-1))
        print(
            f'Result class id {class_id}, socre {round(host_out[class_id]):.2f}'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensorrt-weights",
        type=str,
        default=None,
        help="Path to TensorRT weights, which is generated by gen_weights.py")
    parser.add_argument("--input-video",
                        type=str,
                        default=None,
                        help="Path to local video file")
    parser.add_argument("--save-engine-path",
                        type=str,
                        default=None,
                        help="Save engine to local file")
    parser.add_argument("--load-engine-path",
                        type=str,
                        default=None,
                        help="Saved engine file path")
    parser.add_argument("--test-mmaction2",
                        action='store_true',
                        help="Compare TensorRT results with MMAction2 Results")
    parser.add_argument("--mmaction2-config",
                        type=str,
                        default=None,
                        help="Path to MMAction2 config file")
    parser.add_argument("--mmaction2-checkpoint",
                        type=str,
                        default=None,
                        help="Path to MMAction2 checkpoint url or file path")
    parser.add_argument("--test-cpp",
                        action='store_true',
                        help="Compare Python API results with C++ API results")
    parser.add_argument("--cpp-result-path",
                        type=str,
                        default='./build/result.txt',
                        help="Path to C++ API results")

    main(parser.parse_args())
