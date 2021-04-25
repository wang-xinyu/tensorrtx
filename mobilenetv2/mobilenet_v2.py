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

WEIGHT_PATH = "./mobilenetv2.wts"
ENGINE_PATH = "./mobilenetv2.engine"

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


def conv_bn_relu(network, weight_map, input, outch, ksize, s, g, lname):
    p = (ksize - 1) // 2

    conv1 = network.add_convolution(input=input,
                                    num_output_maps=outch,
                                    kernel_shape=(ksize, ksize),
                                    kernel=weight_map[lname + "0.weight"],
                                    bias=trt.Weights())
    assert conv1
    conv1.stride = (s, s)
    conv1.padding = (p, p)
    conv1.num_groups = g

    bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + "1", EPS)
    assert bn1

    relu1 = network.add_activation(bn1.get_output(0), type=trt.ActivationType.RELU)
    assert relu1

    shift = np.array(-6.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    power = np.array(1.0, dtype=np.float32)
    scale1 = network.add_scale(input=bn1.get_output(0),
                               mode=trt.ScaleMode.UNIFORM,
                               shift=shift,
                               scale=scale,
                               power=power)
    assert scale1

    relu2 = network.add_activation(scale1.get_output(0), type=trt.ActivationType.RELU)
    assert relu2

    ew1 = network.add_elementwise(relu1.get_output(0), relu2.get_output(0), trt.ElementWiseOperation.SUB)
    assert ew1

    return ew1


def inverted_res(network, weight_map, input, lname, inch, outch, s, exp):
    hidden = inch * exp
    use_res_connect = (s == 1 and inch == outch)

    if exp != 1:
        ew1 = conv_bn_relu(network, weight_map, input, hidden, 1, 1, 1, lname + "conv.0.")
        ew2 = conv_bn_relu(network, weight_map, ew1.get_output(0), hidden, 3, s, hidden, lname + "conv.1.")
        conv1 = network.add_convolution(input=ew2.get_output(0),
                                        num_output_maps=outch,
                                        kernel_shape=(1, 1),
                                        kernel=weight_map[lname + "conv.2.weight"],
                                        bias=trt.Weights())
        assert conv1
        bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + "conv.3", EPS)
    else:
        ew1 = conv_bn_relu(network, weight_map, input, hidden, 3, s, hidden, lname + "conv.0.")
        conv1 = network.add_convolution(input=ew1.get_output(0),
                                        num_output_maps=outch,
                                        kernel_shape=(1, 1),
                                        kernel=weight_map[lname + "conv.1.weight"],
                                        bias=trt.Weights())
        assert conv1
        bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + "conv.2", EPS)

    if not use_res_connect:
        return bn1

    ew3 = network.add_elementwise(input, bn1.get_output(0), trt.ElementWiseOperation.SUM)
    assert ew3

    return ew3


def create_engine(max_batch_size, builder, config, dt):
    weight_map = load_weights(WEIGHT_PATH)
    network = builder.create_network()

    data = network.add_input(INPUT_BLOB_NAME, dt, (3, INPUT_H, INPUT_W))
    assert data

    ew1 = conv_bn_relu(network, weight_map, data, 32, 3, 2, 1, "features.0.")
    ir1 = inverted_res(network, weight_map, ew1.get_output(0), "features.1.", 32, 16, 1, 1)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.2.", 16, 24, 2, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.3.", 24, 24, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.4.", 24, 32, 2, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.5.", 32, 32, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.6.", 32, 32, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.7.", 32, 64, 2, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.8.", 64, 64, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.9.", 64, 64, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.10.", 64, 64, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.11.", 64, 96, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.12.", 96, 96, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.13.", 96, 96, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.14.", 96, 160, 2, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.15.", 160, 160, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.16.", 160, 160, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.17.", 160, 320, 1, 6)
    ew2 = conv_bn_relu(network, weight_map, ir1.get_output(0), 1280, 1, 1, 1, "features.18.")

    pool1 = network.add_pooling(input=ew2.get_output(0),
                                type=trt.PoolingType.AVERAGE,
                                window_size=trt.DimsHW(7, 7))
    assert pool1

    fc1 = network.add_fully_connected(input=pool1.get_output(0),
                                      num_outputs=OUTPUT_SIZE,
                                      kernel=weight_map["classifier.1.weight"],
                                      bias=weight_map["classifier.1.bias"])
    assert fc1

    fc1.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(fc1.get_output(0))

    # Build Engine
    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = 1 << 32
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
            "python mobilenet_v2.py -s   # serialize model to plan file\n"
            "python mobilenet_v2.py -d   # deserialize plan file and run inference"
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
