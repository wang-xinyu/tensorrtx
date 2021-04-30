import os
import sys
import struct
import argparse

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

BATCH_SIZE = 1
INPUT_H = 224
INPUT_W = 224
OUTPUT_SIZE = 1000
INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"

WEIGHT_PATH = "./alexnet.wts"
ENGINE_PATH = "./alexnet.engine"

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


def create_engine(max_batch_size, builder, config, dt):
    weight_map = load_weights(WEIGHT_PATH)
    network = builder.create_network()

    data = network.add_input(INPUT_BLOB_NAME, dt, (3, INPUT_H, INPUT_W))
    assert data

    conv1 = network.add_convolution(input=data,
                                    num_output_maps=64,
                                    kernel_shape=(11, 11),
                                    kernel=weight_map["features.0.weight"],
                                    bias=weight_map["features.0.bias"])
    assert conv1
    conv1.stride = (4, 4)
    conv1.padding = (2, 2)

    relu1 = network.add_activation(conv1.get_output(0), type=trt.ActivationType.RELU)
    assert relu1

    pool1 = network.add_pooling(input=relu1.get_output(0),
                                type=trt.PoolingType.MAX,
                                window_size=trt.DimsHW(3, 3))
    assert pool1
    pool1.stride_nd = (2, 2)

    conv2 = network.add_convolution(input=pool1.get_output(0),
                                    num_output_maps=192,
                                    kernel_shape=(5, 5),
                                    kernel=weight_map["features.3.weight"],
                                    bias=weight_map["features.3.bias"])
    assert conv2
    conv2.padding = (2, 2)

    relu2 = network.add_activation(conv2.get_output(0), type=trt.ActivationType.RELU)
    assert relu2

    pool2 = network.add_pooling(input=relu2.get_output(0),
                                type=trt.PoolingType.MAX,
                                window_size=trt.DimsHW(3, 3))
    assert pool2
    pool2.stride_nd = (2, 2)

    conv3 = network.add_convolution(input=pool2.get_output(0),
                                    num_output_maps=384,
                                    kernel_shape=(3, 3),
                                    kernel=weight_map["features.6.weight"],
                                    bias=weight_map["features.6.bias"])
    assert conv3
    conv3.padding = (1, 1)

    relu3 = network.add_activation(conv3.get_output(0), type=trt.ActivationType.RELU)
    assert relu3

    conv4 = network.add_convolution(input=relu3.get_output(0),
                                    num_output_maps=256,
                                    kernel_shape=(3, 3),
                                    kernel=weight_map["features.8.weight"],
                                    bias=weight_map["features.8.bias"])
    assert conv4
    conv4.padding = (1, 1)

    relu4 = network.add_activation(conv4.get_output(0), type=trt.ActivationType.RELU)
    assert relu4

    conv5 = network.add_convolution(input=relu4.get_output(0),
                                    num_output_maps=256,
                                    kernel_shape=(3, 3),
                                    kernel=weight_map["features.10.weight"],
                                    bias=weight_map["features.10.bias"])
    assert conv5
    conv5.padding = (1, 1)

    relu5 = network.add_activation(conv5.get_output(0), type=trt.ActivationType.RELU)
    assert relu5

    pool3 = network.add_pooling(input=relu5.get_output(0),
                                type=trt.PoolingType.MAX,
                                window_size=trt.DimsHW(3, 3))
    assert pool3
    pool3.stride_nd = (2, 2)

    fc1 = network.add_fully_connected(input=pool3.get_output(0),
                                      num_outputs=4096,
                                      kernel=weight_map["classifier.1.weight"],
                                      bias=weight_map["classifier.1.bias"])
    assert fc1

    relu6 = network.add_activation(fc1.get_output(0), type=trt.ActivationType.RELU)
    assert relu6

    fc2 = network.add_fully_connected(input=relu6.get_output(0),
                                      num_outputs=4096,
                                      kernel=weight_map["classifier.4.weight"],
                                      bias=weight_map["classifier.4.bias"])
    assert fc2

    relu7 = network.add_activation(fc2.get_output(0), type=trt.ActivationType.RELU)
    assert relu7

    fc3 = network.add_fully_connected(input=relu7.get_output(0),
                                      num_outputs=1000,
                                      kernel=weight_map["classifier.6.weight"],
                                      bias=weight_map["classifier.6.bias"])
    assert fc3

    fc3.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(fc3.get_output(0))

    # Build Engine
    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = 1 << 20
    engine = builder.build_engine(network, config)

    del network
    del weight_map

    return engine


def API_to_model(max_batch_size):
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    engine = create_engine(max_batch_size, builder, config, trt.float32)
    assert engine
    with open(ENGINE_PATH, "wb") as f:
        f.write(engine.serialize())

    del engine
    del builder
    del config


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", action='store_true')
    parser.add_argument("-d", action='store_true')
    args = parser.parse_args()

    if not (args.s ^ args.d):
        print(
            "arguments not right!\n"
            "python alexnet.py -s   # serialize model to plan file\n"
            "python alexnet.py -d   # deserialize plan file and run inference"
        )
        sys.exit()

    if args.s:
        API_to_model(BATCH_SIZE)
    else:
        runtime = trt.Runtime(TRT_LOGGER)
        assert runtime

        with open(ENGINE_PATH, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine

        context = engine.create_execution_context()
        assert context

        data = np.ones((BATCH_SIZE * 3 * INPUT_H * INPUT_W), dtype=np.float32)
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        inputs[0].host = data

        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        print(f'Output: \n{trt_outputs[0][:10]}\n{trt_outputs[0][-10:]}')
