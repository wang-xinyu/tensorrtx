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
EPS = 1e-5

WEIGHT_PATH = "./densenet121.wts"
ENGINE_PATH = "./densenet121.engine"

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


def add_batch_norm_2d(network, weight_map, input, layer_name):
    gamma = weight_map[layer_name + ".weight"]
    beta = weight_map[layer_name + ".bias"]
    mean = weight_map[layer_name + ".running_mean"]
    var = weight_map[layer_name + ".running_var"]
    var = np.sqrt(var + EPS)

    scale = gamma / var
    shift = -mean / var * gamma + beta
    return network.add_scale(input=input,
                             mode=trt.ScaleMode.CHANNEL,
                             shift=shift,
                             scale=scale)


def add_dense_layer(network, input, weight_map, lname):
    bn1 = add_batch_norm_2d(network, weight_map, input, lname + ".norm1")

    relu1 = network.add_activation(bn1.get_output(0), type=trt.ActivationType.RELU)
    assert relu1

    conv1 = network.add_convolution(input=relu1.get_output(0),
                                    num_output_maps=128,
                                    kernel_shape=(1, 1),
                                    kernel=weight_map[lname + ".conv1.weight"],
                                    bias=trt.Weights())
    assert conv1
    conv1.stride = (1, 1)

    bn2 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + ".norm2")

    relu2 = network.add_activation(bn2.get_output(0), type=trt.ActivationType.RELU)
    assert relu2

    conv2 = network.add_convolution(input=relu2.get_output(0),
                                    num_output_maps=32,
                                    kernel_shape=(3, 3),
                                    kernel=weight_map[lname + ".conv2.weight"],
                                    bias=trt.Weights())
    assert conv2
    conv2.stride = (1, 1)
    conv2.padding = (1, 1)

    return conv2


def add_transition(network, input, weight_map, outch, lname):
    bn1 = add_batch_norm_2d(network, weight_map, input, lname + ".norm")

    relu1 = network.add_activation(bn1.get_output(0), type=trt.ActivationType.RELU)
    assert relu1

    conv1 = network.add_convolution(input=relu1.get_output(0),
                                    num_output_maps=outch,
                                    kernel_shape=(1, 1),
                                    kernel=weight_map[lname + ".conv.weight"],
                                    bias=trt.Weights())
    assert conv1
    conv1.stride = (1, 1)

    pool1 = network.add_pooling(input=conv1.get_output(0),
                                type=trt.PoolingType.AVERAGE,
                                window_size=trt.DimsHW(2, 2))
    assert pool1
    pool1.stride_nd = (2, 2)
    pool1.padding_nd = (0, 0)

    return pool1


def add_dense_block(network, input, weight_map, num_dense_layers, lname):
    input_tensors = [None for _ in range(num_dense_layers+1)]
    input_tensors[0] = input
    c = add_dense_layer(network, input, weight_map, lname + ".denselayer" + str(1))
    for i in range(1, num_dense_layers):
        input_tensors[i] = c.get_output(0)
        concat = network.add_concatenation(input_tensors[:i+1])
        assert concat
        c = add_dense_layer(network, concat.get_output(0), weight_map, lname + ".denselayer" + str(i+1))

    input_tensors[num_dense_layers] = c.get_output(0)
    concat = network.add_concatenation(input_tensors)
    assert concat

    return concat


def create_engine(max_batch_size, builder, config, dt):
    weight_map = load_weights(WEIGHT_PATH)
    network = builder.create_network()

    data = network.add_input(INPUT_BLOB_NAME, dt, (3, INPUT_H, INPUT_W))
    assert data

    conv0 = network.add_convolution(input=data,
                                    num_output_maps=64,
                                    kernel_shape=(7, 7),
                                    kernel=weight_map["features.conv0.weight"],
                                    bias=trt.Weights())
    assert conv0
    conv0.stride = (2, 2)
    conv0.padding = (3, 3)

    bn0 = add_batch_norm_2d(network, weight_map, conv0.get_output(0), "features.norm0")

    relu0 = network.add_activation(bn0.get_output(0), type=trt.ActivationType.RELU)
    assert relu0

    pool0 = network.add_pooling(input=relu0.get_output(0),
                                type=trt.PoolingType.MAX,
                                window_size=trt.DimsHW(3, 3))
    assert pool0
    pool0.stride_nd = (2, 2)
    pool0.padding_nd = (1, 1)

    dense1 = add_dense_block(network, pool0.get_output(0), weight_map, 6, "features.denseblock1")
    transition1 = add_transition(network, dense1.get_output(0), weight_map, 128, "features.transition1")

    dense2 = add_dense_block(network, transition1.get_output(0), weight_map, 12, "features.denseblock2")
    transition2 = add_transition(network, dense2.get_output(0), weight_map, 256, "features.transition2")

    dense3 = add_dense_block(network, transition2.get_output(0), weight_map, 24, "features.denseblock3")
    transition3 = add_transition(network, dense3.get_output(0), weight_map, 512, "features.transition3")

    dense4 = add_dense_block(network, transition3.get_output(0), weight_map, 16, "features.denseblock4")

    bn5 = add_batch_norm_2d(network, weight_map, dense4.get_output(0), "features.norm5")
    relu5 = network.add_activation(bn5.get_output(0), type=trt.ActivationType.RELU)

    pool5 = network.add_pooling(relu5.get_output(0), type=trt.PoolingType.AVERAGE, window_size=trt.DimsHW(7, 7))

    fc1 = network.add_fully_connected(input=pool5.get_output(0),
                                      num_outputs=OUTPUT_SIZE,
                                      kernel=weight_map["classifier.weight"],
                                      bias=weight_map["classifier.bias"])
    assert fc1

    fc1.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(fc1.get_output(0))

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
            "python densenet121.py -s   # serialize model to plan file\n"
            "python densenet121.py -d   # deserialize plan file and run inference"
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
