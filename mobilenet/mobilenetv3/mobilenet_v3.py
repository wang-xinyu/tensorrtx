import os
import sys
import struct
import argparse

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import tensorrt as trt

BATCH_SIZE = 1
INPUT_H = 224
INPUT_W = 224
OUTPUT_SIZE = 1000
BS = 1
INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"
EPS = 1e-3

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def preprocess_image(img_path, h=INPUT_H, w=INPUT_W):
    """
    Returns np.float32 array of shape (1, 3, H, W), normalized like ImageNet.
    Falls back to simple [0,1] if Pillow/OpenCV not available.
    """
    try:
        import cv2
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
    except Exception:
        # Fallback to PIL
        from PIL import Image
        img = Image.open(img_path).convert("RGB").resize((w, h))
        img = np.asarray(img, dtype=np.float32) / 255.0

    # Imagenet normalization (same as common MobileNetV3 training)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # HWC->CHW and add batch dim
    img = np.transpose(img, (2, 0, 1))            # (3, H, W)
    img = np.expand_dims(img, axis=0).copy()      # (1, 3, H, W)
    return img


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
    conv1 = network.add_convolution_nd(input=input,
                                       num_output_maps=outch,
                                       kernel_shape=(ksize, ksize),
                                       kernel=weight_map[lname + "0.weight"],
                                       bias=trt.Weights())
    assert conv1
    conv1.stride_nd = (s, s)
    conv1.padding_nd = (p, p)
    conv1.num_groups = g

    bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + "1", EPS)
    hsw = add_h_swish(network, bn1.get_output(0))
    assert hsw

    return hsw


def add_se_layer(network, weight_map, input, c, w, lname):
    # Global average pool to (BS, C, 1, 1)
    pool = network.add_pooling_nd(input=input,
                                  type=trt.PoolingType.AVERAGE,
                                  window_size=trt.DimsHW(w, w))
    pool.stride_nd = (w, w)

    # Infer SE reduction channels from weights (do NOT assume c//4)
    se_c = int(weight_map[lname + "fc1.bias"].size)

    # Flatten pooled tensor to (BS, C)
    sh1 = network.add_shuffle(pool.get_output(0))
    sh1.reshape_dims = (BS, c)

    # fc1: (BS, C) x (C, se_c) + b -> (BS, se_c)
    w1 = weight_map[lname + "fc1.weight"].reshape(se_c, c, 1, 1).reshape(se_c, c)
    b1 = weight_map[lname + "fc1.bias"].reshape(1, se_c)
    k1 = network.add_constant((se_c, c), w1)
    mm1 = network.add_matrix_multiply(sh1.get_output(0), trt.MatrixOperation.NONE,
                                      k1.get_output(0), trt.MatrixOperation.TRANSPOSE)
    add1 = network.add_elementwise(
        mm1.get_output(0),
        network.add_constant((1, se_c), b1).get_output(0),
        trt.ElementWiseOperation.SUM
    )
    relu = network.add_activation(add1.get_output(0), trt.ActivationType.RELU)

    # fc2: (BS, se_c) x (se_c, C) + b -> (BS, C)
    w2 = weight_map[lname + "fc2.weight"].reshape(c, se_c, 1, 1).reshape(c, se_c)
    b2 = weight_map[lname + "fc2.bias"].reshape(1, c)
    k2 = network.add_constant((c, se_c), w2)
    mm2 = network.add_matrix_multiply(relu.get_output(0), trt.MatrixOperation.NONE,
                                      k2.get_output(0), trt.MatrixOperation.TRANSPOSE)
    add2 = network.add_elementwise(
        mm2.get_output(0),
        network.add_constant((1, c), b2).get_output(0),
        trt.ElementWiseOperation.SUM
    )

    # hard-sigmoid, then reshape to (BS, C, 1, 1) for channel-wise scaling
    hsig = network.add_activation(add2.get_output(0), trt.ActivationType.HARD_SIGMOID)
    hsig.alpha = 1.0 / 6.0
    hsig.beta = 0.5
    sh2 = network.add_shuffle(hsig.get_output(0))
    sh2.reshape_dims = (BS, c, 1, 1)

    return network.add_elementwise(input, sh2.get_output(0), trt.ElementWiseOperation.PROD)


def conv_seq_1(network, weight_map, input, output, hdim, k, s, use_se, use_hs, w, lname):
    p = (k - 1) // 2
    conv1 = network.add_convolution_nd(input=input,
                                       num_output_maps=hdim,
                                       kernel_shape=(k, k),
                                       kernel=weight_map[lname + "0.0.weight"],
                                       bias=trt.Weights())
    assert conv1
    conv1.stride_nd = (s, s)
    conv1.padding_nd = (p, p)
    conv1.num_groups = hdim

    bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + "0.1", EPS)

    if use_hs:
        hsw = add_h_swish(network, bn1.get_output(0))
        tensor3 = hsw.get_output(0)
    else:
        relu1 = network.add_activation(bn1.get_output(0), type=trt.ActivationType.RELU)
        tensor3 = relu1.get_output(0)

    if use_se:
        se1 = add_se_layer(network, weight_map, tensor3, hdim, w, lname + "1.")
        tensor4 = se1.get_output(0)
    else:
        tensor4 = tensor3

    conv2 = network.add_convolution_nd(input=tensor4,
                                       num_output_maps=output,
                                       kernel_shape=(1, 1),
                                       kernel=weight_map[lname + "2.0.weight"],
                                       bias=trt.Weights())
    bn2 = add_batch_norm_2d(network, weight_map, conv2.get_output(0), lname + "2.1", EPS)
    assert bn2

    return bn2


def conv_seq_2(network, weight_map, input, output, hdim, k, s, use_se, use_hs, w, lname):
    p = (k - 1) // 2
    conv1 = network.add_convolution_nd(input=input, num_output_maps=hdim,
                                       kernel_shape=(1, 1),
                                       kernel=weight_map[lname + "0.0.weight"],
                                       bias=trt.Weights())
    bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + "0.1", EPS)
    x = add_h_swish(network, bn1.get_output(0)).get_output(0) if use_hs else \
        network.add_activation(bn1.get_output(0), trt.ActivationType.RELU).get_output(0)

    conv2 = network.add_convolution_nd(input=x, num_output_maps=hdim,
                                       kernel_shape=(k, k),
                                       kernel=weight_map[lname + "1.0.weight"],
                                       bias=trt.Weights())
    conv2.stride_nd = (s, s)
    conv2.padding_nd = (p, p)
    conv2.num_groups = hdim
    bn2 = add_batch_norm_2d(network, weight_map, conv2.get_output(0), lname + "1.1", EPS)

    x = add_h_swish(network, bn2.get_output(0)).get_output(0) if use_hs else \
        network.add_activation(bn2.get_output(0), trt.ActivationType.RELU).get_output(0)

    if use_se:
        x = add_se_layer(network, weight_map, x, hdim, w, lname + "2.").get_output(0)

    proj_idx = "3." if use_se else "2."
    conv3 = network.add_convolution_nd(input=x, num_output_maps=output,
                                       kernel_shape=(1, 1),
                                       kernel=weight_map[lname + proj_idx + "0.weight"],
                                       bias=trt.Weights())
    bn3 = add_batch_norm_2d(network, weight_map, conv3.get_output(0), lname + proj_idx + "1", EPS)
    return bn3


def conv_seq_0(network, weight_map, input, output, hdim, k, s, use_se, use_hs, w, lname):
    # NO expansion, NO SE (e.g., MobileNetV3-Large features.1)
    # lname is "...features.X.block."
    p = (k - 1) // 2

    # Depthwise conv + BN + act
    conv1 = network.add_convolution_nd(
        input=input,
        num_output_maps=hdim,
        kernel_shape=(k, k),
        kernel=weight_map[lname + "0.0.weight"],
        bias=trt.Weights(),
    )
    conv1.stride_nd = (s, s)
    conv1.padding_nd = (p, p)
    conv1.num_groups = hdim

    bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + "0.1", EPS)

    if use_hs:
        x = add_h_swish(network, bn1.get_output(0)).get_output(0)
    else:
        x = network.add_activation(bn1.get_output(0), type=trt.ActivationType.RELU).get_output(0)

    # Pointwise projection + BN
    conv2 = network.add_convolution_nd(
        input=x,
        num_output_maps=output,
        kernel_shape=(1, 1),
        kernel=weight_map[lname + "1.0.weight"],
        bias=trt.Weights(),
    )
    bn2 = add_batch_norm_2d(network, weight_map, conv2.get_output(0), lname + "1.1", EPS)
    assert bn2

    return bn2


def inverted_res(network, weight_map, input, lname, inch, outch, s, hidden, k, use_se, use_hs, w):
    use_res_connect = (s == 1 and inch == outch)

    if inch == hidden:
        if use_se:
            # depthwise (+ optional SE) -> projection
            conv = conv_seq_1(network, weight_map, input, outch, hidden, k, s, use_se, use_hs, w, lname + "block.")
        else:
            # NO expansion, NO SE  (MobileNetV3-Large features.1 etc.)
            conv = conv_seq_0(network, weight_map, input, outch, hidden, k, s, use_se, use_hs, w, lname + "block.")
    else:
        # expansion -> depthwise (+ optional SE) -> projection
        conv = conv_seq_2(network, weight_map, input, outch, hidden, k, s, use_se, use_hs, w, lname + "block.")

    if not use_res_connect:
        return conv

    ew3 = network.add_elementwise(input, conv.get_output(0), trt.ElementWiseOperation.SUM)
    assert ew3
    return ew3


def create_engine_small(max_batch_size, builder, config, dt):
    weight_map = load_weights("./mbv3_small.wts")

    # TensorRT 10: Use explicit batch dimension
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # TensorRT 10: Input with explicit batch dimension
    data = network.add_input(INPUT_BLOB_NAME, dt, (max_batch_size, 3, INPUT_H, INPUT_W))
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
    ew2 = conv_bn_h_swish(network, weight_map, ir11.get_output(0), 576, 1, 1, 1, "features.12.")

    pool1 = network.add_pooling_nd(input=ew2.get_output(0),
                                   type=trt.PoolingType.AVERAGE,
                                   window_size=trt.DimsHW(7, 7))
    # Flatten to (N, 576)
    reshape1 = network.add_shuffle(pool1.get_output(0))
    reshape1.reshape_dims = (max_batch_size, 576)

    # FCs via MatrixMultiply + bias (like C++)
    fc1_w = weight_map["classifier.0.weight"].reshape(1024, 576)
    fc1_b = weight_map["classifier.0.bias"].reshape(1, 1024)
    k1 = network.add_constant((1024, 576), fc1_w)
    mm1 = network.add_matrix_multiply(reshape1.get_output(0), trt.MatrixOperation.NONE,
                                      k1.get_output(0), trt.MatrixOperation.TRANSPOSE)
    b1 = network.add_constant((1, 1024), fc1_b)
    add1 = network.add_elementwise(mm1.get_output(0), b1.get_output(0), trt.ElementWiseOperation.SUM)

    hsw = add_h_swish(network, add1.get_output(0))

    fc2_w = weight_map["classifier.3.weight"].reshape(1000, 1024)
    fc2_b = weight_map["classifier.3.bias"].reshape(1, 1000)
    k2 = network.add_constant((1000, 1024), fc2_w)
    mm2 = network.add_matrix_multiply(hsw.get_output(0), trt.MatrixOperation.NONE,
                                      k2.get_output(0), trt.MatrixOperation.TRANSPOSE)
    b2 = network.add_constant((1, 1000), fc2_b)
    add2 = network.add_elementwise(mm2.get_output(0), b2.get_output(0), trt.ElementWiseOperation.SUM)

    add2.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(add2.get_output(0))

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)

    serialized_engine = builder.build_serialized_network(network, config)

    del network
    del weight_map

    return serialized_engine


def create_engine_large(max_batch_size, builder, config, dt):
    weight_map = load_weights("./mbv3_large.wts")

    # TensorRT 10: Use explicit batch dimension
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # TensorRT 10: Input with explicit batch dimension
    data = network.add_input(INPUT_BLOB_NAME, dt, (max_batch_size, 3, INPUT_H, INPUT_W))
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
    ir13 = inverted_res(network, weight_map, ir12.get_output(0), "features.13.", 112, 160, 2, 672, 5, 1, 1, 7)
    ir14 = inverted_res(network, weight_map, ir13.get_output(0), "features.14.", 160, 160, 1, 960, 5, 1, 1, 7)
    ir15 = inverted_res(network, weight_map, ir14.get_output(0), "features.15.", 160, 160, 1, 960, 5, 1, 1, 7)
    ew2 = conv_bn_h_swish(network, weight_map, ir15.get_output(0), 960, 1, 1, 1, "features.16.")

    pool1 = network.add_pooling_nd(input=ew2.get_output(0),
                                   type=trt.PoolingType.AVERAGE,
                                   window_size=trt.DimsHW(7, 7))
    assert pool1
    pool1.stride_nd = trt.DimsHW(7, 7)

    # TensorRT 10: Alternative approach using MatrixMultiply for fully connected layer
    # Reshape pooled output to 2D
    input_reshape = network.add_shuffle(pool1.get_output(0))
    input_reshape.reshape_dims = (max_batch_size, 960)

    # Add weight as constant layer
    fc_weights = weight_map["classifier.0.weight"].reshape(1280, 960)
    filter_const = network.add_constant((1280, 960), fc_weights)

    # Matrix multiplication
    mm = network.add_matrix_multiply(input_reshape.get_output(0), trt.MatrixOperation.NONE,
                                     filter_const.get_output(0), trt.MatrixOperation.TRANSPOSE)

    # Add bias
    bias_weights = weight_map["classifier.0.bias"].reshape(1, 1280)
    bias_const = network.add_constant((1, 1280), bias_weights)

    fc1 = network.add_elementwise(mm.get_output(0), bias_const.get_output(0), trt.ElementWiseOperation.SUM)
    assert fc1

    sw2 = add_h_swish(network, fc1.get_output(0))

    # Second fully connected layer
    input_reshape2 = network.add_shuffle(sw2.get_output(0))
    input_reshape2.reshape_dims = (max_batch_size, 1280)

    fc2_weights = weight_map["classifier.3.weight"].reshape(OUTPUT_SIZE, 1280)
    filter_const2 = network.add_constant((OUTPUT_SIZE, 1280), fc2_weights)

    mm2 = network.add_matrix_multiply(input_reshape2.get_output(0), trt.MatrixOperation.NONE,
                                      filter_const2.get_output(0), trt.MatrixOperation.TRANSPOSE)

    bias2_weights = weight_map["classifier.3.bias"].reshape(1, OUTPUT_SIZE)
    bias_const2 = network.add_constant((1, OUTPUT_SIZE), bias2_weights)

    fc2 = network.add_elementwise(mm2.get_output(0), bias_const2.get_output(0), trt.ElementWiseOperation.SUM)
    assert fc2

    fc2.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(fc2.get_output(0))

    # TensorRT 10: Updated builder configuration
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)

    # TensorRT 10: Use serialize_network instead of build_engine
    serialized_engine = builder.build_serialized_network(network, config)

    del network
    del weight_map

    return serialized_engine


def API_to_model(max_batch_size, model_type):
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()

    if model_type == "small":
        serialized_engine = create_engine_small(max_batch_size, builder, config, trt.float32)
        engine_path = "./mobilenetv3_small.plan"
    else:
        serialized_engine = create_engine_large(max_batch_size, builder, config, trt.float32)
        engine_path = "./mobilenetv3_large.plan"

    assert serialized_engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"Model serialized successfully to {engine_path}")

    del serialized_engine
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


def allocate_buffers(engine, context):
    """TensorRT 10 compatible buffer allocation"""
    inputs = []
    outputs = []
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_mode = engine.get_tensor_mode(tensor_name)
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        tensor_shape = context.get_tensor_shape(tensor_name)

        size = trt.volume(tensor_shape)
        dtype = trt.nptype(tensor_dtype)

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Set tensor address
        context.set_tensor_address(tensor_name, int(device_mem))

        # Append to the appropriate list
        if tensor_mode == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, stream


def do_inference(context, inputs, outputs, stream):
    """TensorRT 10 compatible inference"""
    # Transfer input data to the GPU
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # Run inference using TensorRT 10 API
    context.execute_async_v3(stream_handle=stream.handle)

    # Transfer predictions back from the GPU
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

    # Synchronize the stream
    stream.synchronize()

    # Return only the host outputs
    return [out.host for out in outputs]


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", action='store_true', help="serialize model to plan file")
    parser.add_argument("-d", action='store_true', help="deserialize plan file and run inference")
    parser.add_argument("model_type", choices=["small", "large"], help="Model variant (small or large)")
    parser.add_argument("--image", default=None, type=str, help="Image to run inference.")
    args = parser.parse_args()

    if not (args.s ^ args.d):
        print(
            "Usage: python mobilenet_v3.py <-s|-d> <small|large>\n"
            "  -s small/large  : serialize model to plan file\n"
            "  -d small/large  : deserialize plan file and run inference\n"
            "\n"
            "Examples:\n"
            "  python mobilenet_v3.py -s large  # Build MobileNetV3-Large engine\n"
            "  python mobilenet_v3.py -d large  # Run inference with all-ones test\n"
            "  python mobilenet_v3.py -d small  # Run inference with MobileNetV3-Small\n"
            "  python mobilenet_v3.py -d small  --image ../image.jpg # Run inference with MobileNetV3-Small on a image"

        )
        sys.exit()

    if args.s:
        API_to_model(BATCH_SIZE, args.model_type)
    else:
        engine_path = f"./mobilenetv3_{args.model_type}.plan"

        if not os.path.exists(engine_path):
            print(f"Engine file {engine_path} not found. Please run with -s first to build the engine.")
            sys.exit(1)

        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine

        context = engine.create_execution_context()
        assert context

        # load image or fall back to ones
        if "--image" in sys.argv or any(a.startswith("--image") for a in sys.argv):
            # arg already declared in argparse; use it
            if args.image and os.path.exists(args.image):
                host_input = preprocess_image(args.image)  # (1,3,H,W)
                print(f"Using image: {args.image}")
            else:
                print("Image not found, using all-ones test data")
                host_input = np.ones((BATCH_SIZE, 3, INPUT_H, INPUT_W), dtype=np.float32)
        else:
            host_input = np.ones((BATCH_SIZE, 3, INPUT_H, INPUT_W), dtype=np.float32)
            print("No image provided, using all-ones test data")

        context.set_input_shape(INPUT_BLOB_NAME, (BATCH_SIZE, 3, INPUT_H, INPUT_W))

        # allocate host/device buffers and bind them to the context
        inputs, outputs, stream = allocate_buffers(engine, context)

        # populate input host buffer (flatten to match binding size)
        inputs[0].host[:] = host_input.ravel()

        # run inference (this does H2D, enqueue_v3, D2H, sync)
        trt_outputs = do_inference(context, inputs=inputs, outputs=outputs, stream=stream)

        logits = trt_outputs[0].reshape(BATCH_SIZE, OUTPUT_SIZE)[0]
        probs = softmax(logits)

        if args.image:
            top5_idx = probs.argsort()[-5:][::-1]
            print("\nTop-5 classes:")
            for i in top5_idx:
                print(f"{i:4d}: {probs[i]:.4f}")
        else:
            print(f'Output: \n{trt_outputs[0][:10]}\n{trt_outputs[0][-10:]}')

        # Cleanup
        del context
        del engine
        del runtime
