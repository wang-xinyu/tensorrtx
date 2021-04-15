import argparse
import os
import struct
import sys

import numpy as np
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
import tensorrt as trt

BATCH_SIZE = 1
INPUT_H = 224
INPUT_W = 224
OUTPUT_SIZE = 1000
INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"
EPS = 1e-5

WEIGHT_PATH = "./resnet50.wts"
ENGINE_PATH = "./resnet50.engine"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_weights(file):
    print(f"Loading weights: {file}")

    assert os.path.exists(file), 'Unable to load weight file.'

    weight_map = {}
    with open(file, "r") as f:
        lines = f.readlines()
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


def addBatchNorm2d(network, weight_map, input, layer_name, eps):
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
               layer_name):

    conv1 = network.add_convolution(input=input,
                                    num_output_maps=out_channels,
                                    kernel_shape=(1, 1),
                                    kernel=weight_map[layer_name +
                                                      "conv1.weight"],
                                    bias=trt.Weights())
    assert conv1

    bn1 = addBatchNorm2d(network, weight_map, conv1.get_output(0),
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

    bn2 = addBatchNorm2d(network, weight_map, conv2.get_output(0),
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

    bn3 = addBatchNorm2d(network, weight_map, conv3.get_output(0),
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

        bn4 = addBatchNorm2d(network, weight_map, conv4.get_output(0),
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


def createLenetEngine(maxBatchSize, builder, config, dt):
    weight_map = load_weights(WEIGHT_PATH)
    network = builder.create_network()

    data = network.add_input(INPUT_BLOB_NAME, dt, (3, INPUT_H, INPUT_W))
    assert data

    conv1 = network.add_convolution(input=data,
                                    num_output_maps=64,
                                    kernel_shape=(7, 7),
                                    kernel=weight_map["conv1.weight"],
                                    bias=trt.Weights())
    assert conv1
    conv1.stride = (2, 2)
    conv1.padding = (3, 3)

    bn1 = addBatchNorm2d(network, weight_map, conv1.get_output(0), "bn1", EPS)
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

    x = bottleneck(network, weight_map, pool1.get_output(0), 64, 64, 1,
                   "layer1.0.")
    x = bottleneck(network, weight_map, x.get_output(0), 256, 64, 1,
                   "layer1.1.")
    x = bottleneck(network, weight_map, x.get_output(0), 256, 64, 1,
                   "layer1.2.")

    x = bottleneck(network, weight_map, x.get_output(0), 256, 128, 2,
                   "layer2.0.")
    x = bottleneck(network, weight_map, x.get_output(0), 512, 128, 1,
                   "layer2.1.")
    x = bottleneck(network, weight_map, x.get_output(0), 512, 128, 1,
                   "layer2.2.")
    x = bottleneck(network, weight_map, x.get_output(0), 512, 128, 1,
                   "layer2.3.")

    x = bottleneck(network, weight_map, x.get_output(0), 512, 256, 2,
                   "layer3.0.")
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.1.")
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.2.")
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.3.")
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.4.")
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.5.")

    x = bottleneck(network, weight_map, x.get_output(0), 1024, 512, 2,
                   "layer4.0.")
    x = bottleneck(network, weight_map, x.get_output(0), 2048, 512, 1,
                   "layer4.1.")
    x = bottleneck(network, weight_map, x.get_output(0), 2048, 512, 1,
                   "layer4.2.")

    pool2 = network.add_pooling(x.get_output(0),
                                window_size=trt.DimsHW(7, 7),
                                type=trt.PoolingType.AVERAGE)
    assert pool2
    pool2.stride = (1, 1)

    fc1 = network.add_fully_connected(input=pool2.get_output(0),
                                      num_outputs=OUTPUT_SIZE,
                                      kernel=weight_map['fc.weight'],
                                      bias=weight_map['fc.bias'])
    assert fc1

    fc1.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(fc1.get_output(0))

    # Build engine
    builder.max_batch_size = maxBatchSize
    builder.max_workspace_size = 1 << 20
    engine = builder.build_engine(network, config)

    del network
    del weight_map

    return engine


def APIToModel(maxBatchSize):
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    engine = createLenetEngine(maxBatchSize, builder, config, trt.float32)
    assert engine
    with open(ENGINE_PATH, "wb") as f:
        f.write(engine.serialize())

    del engine
    del builder


def doInference(context, host_in, host_out, batchSize):
    engine = context.engine
    assert engine.num_bindings == 2

    devide_in = cuda.mem_alloc(host_in.nbytes)
    devide_out = cuda.mem_alloc(host_out.nbytes)
    bindings = [int(devide_in), int(devide_out)]
    stream = cuda.Stream()

    cuda.memcpy_htod_async(devide_in, host_in, stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_out, devide_out, stream)
    stream.synchronize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", action='store_true')
    parser.add_argument("-d", action='store_true')
    args = parser.parse_args()

    if not (args.s ^ args.d):
        print(
            "arguments not right!\n"
            "python resnet50.py -s   # serialize model to plan file\n"
            "python resnet50.py -d   # deserialize plan file and run inference"
        )
        sys.exit()

    if args.s:
        APIToModel(BATCH_SIZE)
    else:
        runtime = trt.Runtime(TRT_LOGGER)
        assert runtime

        with open(ENGINE_PATH, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine

        context = engine.create_execution_context()
        assert context

        data = np.ones((BATCH_SIZE * 3 * INPUT_H * INPUT_W), dtype=np.float32)
        host_in = cuda.pagelocked_empty(BATCH_SIZE * 3 * INPUT_H * INPUT_W,
                                        dtype=np.float32)
        np.copyto(host_in, data.ravel())
        host_out = cuda.pagelocked_empty(OUTPUT_SIZE, dtype=np.float32)

        doInference(context, host_in, host_out, BATCH_SIZE)

        print(f'Output: \n{host_out[:10]}\n{host_out[-10:]}')
