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
BS = 1
INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"
EPS = 1e-5

WEIGHT_PATH_SMALL = "./mobilenetv3.wts"
ENGINE_PATH = "./mobilenetv3.engine"

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


def add_h_swish(network, input):
    h_sig = network.add_activation(input, type=trt.ActivationType.HARD_SIGMOID)
    assert h_sig
    h_sig.alpha = 1.0 / 6.0
    h_sig.beta = 0.5
    hsw = network.add_elementwise(input, h_sig.get_output(0), trt.ElementWiseOperation.PROD)
    assert hsw

    return hsw


def conv_bn_h_swish(network, weight_map, input, outch, ksize, s, g, lname):
    p = (ksize - 1) // 2
    conv1 = network.add_convolution(input=input,
                                    num_output_maps=outch,
                                    kernel_shape=(ksize, ksize),
                                    kernel=weight_map[lname + "0.weight"],
                                    bias=trt.Weights()
                                    )
    assert conv1
    conv1.stride = (s, s)
    conv1.padding = (p, p)
    conv1.num_groups = g

    bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + "1", EPS)
    hsw = add_h_swish(network, bn1.get_output(0))
    assert hsw

    return hsw


def add_se_layer(network, weight_map, input, c, w, lname):
    h = w
    l1 = network.add_pooling(input=input,
                             type=trt.PoolingType.AVERAGE,
                             window_size=trt.DimsHW(w, h))
    assert l1
    l1.stride_nd = (w, h)

    l2 = network.add_fully_connected(input=l1.get_output(0),
                                     num_outputs=BS * c // 4,
                                     kernel=weight_map[lname + "fc.0.weight"],
                                     bias=weight_map[lname + "fc.0.bias"])
    relu1 = network.add_activation(l2.get_output(0), type=trt.ActivationType.RELU)
    l4 = network.add_fully_connected(input=relu1.get_output(0),
                                     num_outputs=BS * c,
                                     kernel=weight_map[lname + "fc.2.weight"],
                                     bias=weight_map[lname + "fc.2.bias"])

    se = add_h_swish(network, l4.get_output(0))

    return se


def conv_seq_1(network, weight_map, input, output, hdim, k, s, use_se, use_hs, w, lname):
    p = (k - 1) // 2
    conv1 = network.add_convolution(input=input,
                                    num_output_maps=hdim,
                                    kernel_shape=(k, k),
                                    kernel=weight_map[lname + "0.weight"],
                                    bias=trt.Weights())
    assert conv1
    conv1.stride = (s, s)
    conv1.padding = (p, p)
    conv1.num_groups = hdim

    bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + "1", EPS)

    if use_hs:
        hsw = add_h_swish(network, bn1.get_output(0))
        tensor3 = hsw.get_output(0)
    else:
        relu1 = network.add_activation(bn1.get_output(0), type=trt.ActivationType.RELU)
        tensor3 = relu1.get_output(0)

    if use_se:
        se1 = add_se_layer(network, weight_map, tensor3, hdim, w, lname + "3.")
        tensor4 = se1.get_output(0)
    else:
        tensor4 = tensor3

    conv2 = network.add_convolution(input=tensor4,
                                    num_output_maps=output,
                                    kernel_shape=(1, 1),
                                    kernel=weight_map[lname + "4.weight"],
                                    bias=trt.Weights())
    bn2 = add_batch_norm_2d(network, weight_map, conv2.get_output(0), lname + "5", EPS)
    assert bn2

    return bn2


def conv_seq_2(network, weight_map, input, output, hdim, k, s, use_se, use_hs, w, lname):
    p = (k - 1) // 2
    conv1 = network.add_convolution(input=input,
                                    num_output_maps=hdim,
                                    kernel_shape=(1, 1),
                                    kernel=weight_map[lname + "0.weight"],
                                    bias=trt.Weights())
    bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + "1", EPS)

    if use_hs:
        hsw1 = add_h_swish(network, bn1.get_output(0))
        tensor3 = hsw1.get_output(0)
    else:
        relu1 = network.add_activation(bn1.get_output(0), type=trt.ActivationType.RELU)
        tensor3 = relu1.get_output(0)

    conv2 = network.add_convolution(input=tensor3,
                                    num_output_maps=hdim,
                                    kernel_shape=(k, k),
                                    kernel=weight_map[lname + "3.weight"],
                                    bias=trt.Weights())
    conv2.stride = (s, s)
    conv2.padding = (p, p)
    conv2.num_groups = hdim
    bn2 = add_batch_norm_2d(network, weight_map, conv2.get_output(0), lname + "4", EPS)

    if use_se:
        se1 = add_se_layer(network, weight_map, bn2.get_output(0), hdim, w, lname + "5.")
        tensor6 = se1.get_output(0)
    else:
        tensor6 = bn2.get_output(0)

    if use_hs:
        hsw2 = add_h_swish(network, tensor6)
        tensor7 = hsw2.get_output(0)
    else:
        relu2 = network.add_activation(tensor6, type=trt.ActivationType.RELU)
        tensor7 = relu2.get_output(0)

    conv3 = network.add_convolution(input=tensor7,
                                    num_output_maps=output,
                                    kernel_shape=(1, 1),
                                    kernel=weight_map[lname + "7.weight"],
                                    bias=trt.Weights())
    bn3 = add_batch_norm_2d(network, weight_map, conv3.get_output(0), lname + "8", EPS)
    assert bn3

    return bn3


def inverted_res(network, weight_map, input, lname, inch, outch, s, hidden, k, use_se, use_hs, w):
    use_res_connect = (s == 1 and inch == outch)

    if inch == hidden:
        conv = conv_seq_1(network, weight_map, input, outch, hidden, k, s, use_se, use_hs, w, lname + "conv.")
    else:
        conv = conv_seq_2(network, weight_map, input, outch, hidden, k, s, use_se, use_hs, w, lname + "conv.")

    if not use_res_connect:
        return conv

    ew3 = network.add_elementwise(input, conv.get_output(0), trt.ElementWiseOperation.SUM)
    assert ew3

    return ew3


def create_engine_small(max_batch_size, builder, config, dt):
    weight_map = load_weights(WEIGHT_PATH_SMALL)
    network = builder.create_network()

    data = network.add_input(INPUT_BLOB_NAME, dt, (3, INPUT_H, INPUT_W))
    assert data

    ew1 = conv_bn_h_swish(network, weight_map, data, 16, 3, 2, 1, "features.0.")
    ir1 = inverted_res(network, weight_map, ew1.get_output(0), "features.1.", 16, 16, 2, 16, 3, 1, 0, 56)
    ir2 = inverted_res(network, weight_map, ir1.get_output(0), "features.2.", 16, 24, 2, 72, 3, 0, 0, 28)
    ir3 = inverted_res(network, weight_map, ir2.get_output(0), "features.3.", 24, 24, 1, 88, 3, 0, 0, 28)
    ir4 = inverted_res(network, weight_map, ir3.get_output(0), "features.4.", 24, 40, 2, 96, 5, 1, 1, 14)
    ir5 = inverted_res(network, weight_map, ir4.get_output(0), "features.5.", 40, 40, 1, 240, 5, 1, 1, 14)
    ir6 = inverted_res(network, weight_map, ir5.get_output(0), "features.6.", 40, 40, 1, 240, 5, 1, 1, 14)
    ir7 = inverted_res(network, weight_map, ir6.get_output(0), "features.7.", 40, 48, 1, 120, 5, 1, 1, 14)
    ir8 = inverted_res(network, weight_map, ir7.get_output(0), "features.8.", 48, 48, 1, 144, 5, 1, 1, 14)
    ir9 = inverted_res(network, weight_map, ir8.get_output(0), "features.9.", 48, 96, 2, 288, 5, 1, 1, 7)
    ir10 = inverted_res(network, weight_map, ir9.get_output(0), "features.10.", 96, 96, 1, 576, 5, 1, 1, 7)
    ir11 = inverted_res(network, weight_map, ir10.get_output(0), "features.11.", 96, 96, 1, 576, 5, 1, 1, 7)
    ew2 = conv_bn_h_swish(network, weight_map, ir11.get_output(0), 576, 1, 1, 1, "conv.0.")
    se1 = add_se_layer(network, weight_map, ew2.get_output(0), 576, 7, "conv.1.")

    pool1 = network.add_pooling(input=se1.get_output(0),
                                type=trt.PoolingType.AVERAGE,
                                window_size=trt.DimsHW(7, 7))
    assert pool1
    pool1.stride_nd = (7, 7)
    sw1 = add_h_swish(network, pool1.get_output(0))

    fc1 = network.add_fully_connected(input=sw1.get_output(0),
                                      num_outputs=1280,
                                      kernel=weight_map["classifier.0.weight"],
                                      bias=weight_map["classifier.0.bias"])
    assert fc1
    bn1 = add_batch_norm_2d(network, weight_map, fc1.get_output(0), "classifier.1", EPS)
    sw2 = add_h_swish(network, bn1.get_output(0))

    fc2 = network.add_fully_connected(input=sw2.get_output(0),
                                      num_outputs=OUTPUT_SIZE,
                                      kernel=weight_map["classifier.3.weight"],
                                      bias=weight_map["classifier.3.bias"])
    bn2 = add_batch_norm_2d(network, weight_map, fc2.get_output(0), "classifier.4", EPS)
    sw3 = add_h_swish(network, bn2.get_output(0))

    sw3.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(sw3.get_output(0))

    # Build Engine
    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = 1 << 20
    engine = builder.build_engine(network, config)

    del network
    del weight_map

    return engine


def create_engine_large(max_batch_size, builder, config, dt):
    weight_map = load_weights(WEIGHT_PATH_SMALL)
    network = builder.create_network()

    data = network.add_input(INPUT_BLOB_NAME, dt, (3, INPUT_H, INPUT_W))
    assert data

    ew1 = conv_bn_h_swish(network, weight_map, data, 16, 3, 2, 1, "features.0.")
    ir1 = inverted_res(network, weight_map, ew1.get_output(0), "features.1.", 16, 16, 1, 16, 3, 0, 0, 112)
    ir2 = inverted_res(network, weight_map, ir1.get_output(0), "features.2.", 16, 24, 2, 64, 3, 0, 0, 56)
    ir3 = inverted_res(network, weight_map, ir2.get_output(0), "features.3.", 24, 24, 1, 72, 3, 0, 0, 56)
    ir4 = inverted_res(network, weight_map, ir3.get_output(0), "features.4.", 24, 40, 2, 72, 5, 1, 0, 28)
    ir5 = inverted_res(network, weight_map, ir4.get_output(0), "features.5.", 40, 40, 1, 120, 5, 1, 0, 28)
    ir6 = inverted_res(network, weight_map, ir5.get_output(0), "features.6.", 40, 40, 1, 120, 5, 1, 0, 28)
    ir7 = inverted_res(network, weight_map, ir6.get_output(0), "features.7.", 40, 80, 2, 240, 3, 0, 1, 14)
    ir8 = inverted_res(network, weight_map, ir7.get_output(0), "features.8.", 80, 80, 1, 200, 3, 0, 1, 14)
    ir9 = inverted_res(network, weight_map, ir8.get_output(0), "features.9.", 80, 80, 1, 184, 3, 0, 1, 14)
    ir10 = inverted_res(network, weight_map, ir9.get_output(0), "features.10.", 80, 80, 1, 184, 3, 0, 1, 14)
    ir11 = inverted_res(network, weight_map, ir10.get_output(0), "features.11.", 80, 112, 1, 480, 3, 1, 1, 14)
    ir12 = inverted_res(network, weight_map, ir11.get_output(0), "features.12.", 112, 112, 1, 672, 3, 1, 1, 14)
    ir13 = inverted_res(network, weight_map, ir12.get_output(0), "features.13.", 112, 160, 1, 672, 5, 1, 1, 14)
    ir14 = inverted_res(network, weight_map, ir13.get_output(0), "features.14.", 160, 160, 2, 672, 5, 1, 1, 7)
    ir15 = inverted_res(network, weight_map, ir14.get_output(0), "features.15.", 160, 160, 1, 960, 5, 1, 1, 7)
    ew2 = conv_bn_h_swish(network, weight_map, ir15.get_output(0), 960, 1, 1, 1, "conv.0.")

    pool1 = network.add_pooling(input=ew2.get_output(0),
                                type=trt.PoolingType.AVERAGE,
                                window_size=trt.DimsHW(7, 7))
    assert pool1
    pool1.stride_nd = (7, 7)
    sw1 = add_h_swish(network, pool1.get_output(0))

    fc1 = network.add_fully_connected(input=sw1.get_output(0),
                                      num_outputs=1280,
                                      kernel=weight_map["classifier.0.weight"],
                                      bias=weight_map["classifier.0.bias"])
    assert fc1
    sw2 = add_h_swish(network, fc1.get_output(0))

    fc2 = network.add_fully_connected(input=sw2.get_output(0),
                                      num_outputs=OUTPUT_SIZE,
                                      kernel=weight_map["classifier.3.weight"],
                                      bias=weight_map["classifier.3.bias"])

    fc2.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(fc2.get_output(0))

    # Build Engine
    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = 1 << 20
    engine = builder.build_engine(network, config)

    del network
    del weight_map

    return engine


def API_to_model(max_batch_size, model_type):
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    if model_type == "small":
        engine = create_engine_small(max_batch_size, builder, config, trt.float32)
        assert engine
    else:
        engine = create_engine_large(max_batch_size, builder, config, trt.float32)
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
    parser.add_argument("-t", help='indicate small or large model')
    args = parser.parse_args()

    if not (args.s ^ args.d):
        print(
            "arguments not right!\n"
            "python mobilenet_v2.py -s   # serialize model to plan file\n"
            "python mobilenet_v2.py -d   # deserialize plan file and run inference"
        )
        sys.exit()

    if args.s:
        API_to_model(BATCH_SIZE, args.t)
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
