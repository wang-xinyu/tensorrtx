import os
import sys
import struct
import argparse

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

BATCH_SIZE = 1
INPUT_H = 224
INPUT_W = 224
OUTPUT_SIZE = 1000
INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"

WEIGHT_PATH = "./models/regnet_x_400mf.wts"
ENGINE_PATH = "./models/regnet_x_400mf.engine"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_weights(file):
    print(f"Loading weights: {file}")

    assert os.path.exists(file), 'Unable to load weight file.'

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


def add_batchnorm_2d(network, weight_map, lname, input):
    gamma = weight_map[lname + ".weight"]
    beta = weight_map[lname + ".bias"]
    mean = weight_map[lname + ".running_mean"]
    var = weight_map[lname + ".running_var"]

    eps = 1e-5
    scale = gamma / np.sqrt(var + eps)
    shift = beta - mean * scale
    power = np.ones_like(scale)

    bn = network.add_scale(
        input=input,
        mode=trt.ScaleMode.CHANNEL,
        shift=shift,
        scale=scale,
        power=power,
    )
    return bn


def conv_bn_relu(network, weight_map, lname, input, out_ch, ksize, stride, padding, groups=1):
    # Conv
    conv = network.add_convolution_nd(
        input=input,
        num_output_maps=out_ch,
        kernel_shape=(ksize, ksize),
        kernel=weight_map[lname + ".0.weight"],
        bias=None
    )
    conv.stride_nd = (stride, stride)
    conv.padding_nd = (padding, padding)
    conv.num_groups = groups

    # BN
    bn = add_batchnorm_2d(network, weight_map, lname + ".1", conv.get_output(0))

    # ReLU
    relu = network.add_activation(bn.get_output(0), trt.ActivationType.RELU)

    return relu


def add_resbottleneck_block(network, weight_map, lname, input, in_ch, out_ch, stride, groups):
    # f.a conv shape
    # out_ch * in_ch * kernel_h * kernel_w = size
    # → out_ch = size / (in_ch * k_h * k_w)
    w_a = weight_map[lname + ".f.a.0.weight"]
    oc_a = int(w_a.size / (input.get_output(0).shape[1] * 1 * 1))
    bottleneck = oc_a   # f.a output channel.

    # f.c conv shape
    w_c = weight_map[lname + ".f.c.0.weight"]
    out_ch = int(w_c.size / (bottleneck * 1 * 1))

    # 1. projection branch
    in_tensor = input.get_output(0) if hasattr(input, "get_output") else input
    if stride != 1 or in_ch != out_ch:

        proj_conv = network.add_convolution_nd(
            input=in_tensor,
            num_output_maps=out_ch,
            kernel_shape=(1, 1),
            kernel=weight_map[lname + ".proj.0.weight"],
            bias=None
        )
        proj_conv.stride_nd = (stride, stride)

        proj_bn = add_batchnorm_2d(
            network, weight_map, lname + ".proj.1", proj_conv.get_output(0)
        )
        proj_out = proj_bn.get_output(0)
    else:
        proj_out = in_tensor

    # 2. BottleNeck f.a (1×1)
    a = conv_bn_relu(
        network, weight_map, lname + ".f.a",
        input=input.get_output(0),
        out_ch=bottleneck,
        ksize=1,
        stride=1,
        padding=0
    )

    # 3. f.b (3×3 groups)
    b = conv_bn_relu(
        network, weight_map, lname + ".f.b",
        input=a.get_output(0),
        out_ch=bottleneck,
        ksize=3,
        stride=stride,
        padding=1,
        groups=groups
    )

    # 4. f.c (1×1)
    c_conv = network.add_convolution_nd(
        input=b.get_output(0),
        num_output_maps=out_ch,
        kernel_shape=(1, 1),
        kernel=weight_map[lname + ".f.c.0.weight"],
        bias=None
    )
    c_bn = add_batchnorm_2d(network, weight_map, lname + ".f.c.1", c_conv.get_output(0))

    # 5. ElementWise sum
    ew = network.add_elementwise(
        c_bn.get_output(0),
        proj_out,
        trt.ElementWiseOperation.SUM
    )

    # 6. ReLU
    out = network.add_activation(ew.get_output(0), trt.ActivationType.RELU)
    return out


def create_engine(builder, config, dt):
    weight_map = load_weights(WEIGHT_PATH)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    data = network.add_input(INPUT_BLOB_NAME, dt, (1, 3, INPUT_H, INPUT_W))

    stem_out = conv_bn_relu(
        network=network,
        weight_map=weight_map,
        lname="stem",
        input=data,
        out_ch=32,
        ksize=3,
        stride=2,
        padding=1,
    )

    # block 1
    x = add_resbottleneck_block(
        network,
        weight_map,
        lname="trunk_output.block1.block1-0",
        input=stem_out,
        in_ch=32,
        out_ch=32,
        stride=2,
        groups=2
    )

    # block 2
    for i in range(2):
        stride = 2 if i == 0 else 1
        x = add_resbottleneck_block(
            network, weight_map,
            lname=f"trunk_output.block2.block2-{i}",
            input=x,
            in_ch=32 if i == 0 else 64,
            out_ch=64,
            stride=stride,
            groups=4
        )

    # block 3
    for i in range(7):
        stride = 2 if i == 0 else 1
        x = add_resbottleneck_block(
            network, weight_map,
            lname=f"trunk_output.block3.block3-{i}",
            input=x,
            in_ch=64 if i == 0 else 160,
            out_ch=160,
            stride=stride,
            groups=10
        )

    # block 4
    for i in range(12):
        stride = 2 if i == 0 else 1
        x = add_resbottleneck_block(
            network, weight_map,
            lname=f"trunk_output.block4.block4-{i}",
            input=x,
            in_ch=160 if i == 0 else 400,
            out_ch=400,
            stride=stride,
            groups=25
        )

    pool = network.add_pooling_nd(
        input=x.get_output(0),
        window_size=trt.DimsHW(7, 7),
        type=trt.PoolingType.AVERAGE
    )
    pool.stride_nd = trt.DimsHW(7, 7)

    # fully connected.
    # ===== Flatten =====
    flatten = network.add_shuffle(pool.get_output(0))
    flatten.reshape_dims = (1, 400)        # -> (1, 400)

    # ===== FC weights =====
    weight_map["fc.weight"] = weight_map["fc.weight"].reshape(1000, 400)
    fc_weight_tensor = network.add_constant(
        weight_map["fc.weight"].shape,
        weight_map["fc.weight"].astype(np.float32)
    )

    # ===== MatrixMultiply (FC) =====
    fc_mm = network.add_matrix_multiply(
        input0=flatten.get_output(0),           # (1, 400)
        op0=trt.MatrixOperation.NONE,
        input1=fc_weight_tensor.get_output(0),  # (1000, 400)
        op1=trt.MatrixOperation.TRANSPOSE       # → (400, 1000)
    )

    # ===== Bias =====
    weight_map["fc.bias"] = weight_map["fc.bias"].reshape(1, 1000)
    fc_bias_tensor = network.add_constant(
        weight_map["fc.bias"].shape,
        weight_map["fc.bias"].astype(np.float32)
    )

    fc = network.add_elementwise(
        fc_mm.get_output(0),
        fc_bias_tensor.get_output(0),
        trt.ElementWiseOperation.SUM
    )

    fc.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(fc.get_output(0))

    # Build engine
    engine = builder.build_serialized_network(network, config)

    return engine


def API_to_model():
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    engine = create_engine(builder, config, trt.float32)
    assert engine
    with open(ENGINE_PATH, "wb") as f:
        f.write(engine)


def do_inference(context, host_in, host_out, batchSize):
    engine = context.engine

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    print("Input tensor name is :", input_name)
    print("output tensor name is :", output_name)

    device_in = cuda.mem_alloc(host_in.nbytes)
    device_out = cuda.mem_alloc(host_out.nbytes)

    stream = cuda.Stream()

    cuda.memcpy_htod_async(device_in, host_in, stream)

    context.set_tensor_address(input_name, int(device_in))
    context.set_tensor_address(output_name, int(device_out))

    context.execute_async_v3(stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(host_out, device_out, stream)

    stream.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", action='store_true')
    parser.add_argument("-d", action='store_true')
    args = parser.parse_args()

    if not (args.s ^ args.d):
        print(
            "arguments not right!\n"
            "python regnet.py -s   # serialize model to plan file\n"
            "python regnet.py -d   # deserialize plan file and run inference"
        )
        sys.exit()

    if args.s:
        API_to_model()
    else:
        runtime = trt.Runtime(TRT_LOGGER)
        assert runtime

        with open(ENGINE_PATH, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine

        context = engine.create_execution_context()
        assert context

        data = np.ones((BATCH_SIZE, 3, INPUT_H, INPUT_W), dtype=np.float32)
        host_in = cuda.pagelocked_empty(data.size, dtype=np.float32)
        np.copyto(host_in, data.ravel())
        host_out = cuda.pagelocked_empty(OUTPUT_SIZE, dtype=np.float32)

        do_inference(context, host_in, host_out, 1)

        print(f'Output: {host_out}')
