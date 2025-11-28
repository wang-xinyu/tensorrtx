#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa: E501
# pylint: disable=no-member
"""
YOLOv1 TensorRT Inference Script

This script provides functions to:

- Load YOLOv1 weights and shapes from text and binary files.
- Build a TensorRT serialized engine for YOLOv1.
- Perform inference on an input image using the serialized engine.
- Postprocess predictions and draw bounding boxes on the image.

The script can be run in two modes:

1. Serialize the YOLOv1 model to a TensorRT engine file:
    python yolov1.py -s

2. Deserialize the engine and run inference on a test image:
    python yolov1.py -d
"""

import argparse
import os
import sys
import ast
import cv2

import numpy as np
import pycuda.autoinit  # noqa: F401 pylint::disable=unused-import
import pycuda.driver as cuda
import tensorrt as trt

from util import preprocess, pred2xywhcc, draw_bbox

BATCH_SIZE = 1
INPUT_H = 448
INPUT_W = 448
OUTPUT_SIZE = (7, 7, 30)
INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"

WEIGHT_PATH = "./models/yolov1.wts"
ENGINE_PATH = "./models/yolov1.engine"

S = 7
B = 2
NUM_CLASSES = 20
CONF_THRESH = 0.1
IOU_THRESH = 0.3
class_names = [
    "person",
    "bird",
    "cat",
    "cow",
    "dog",
    "horse",
    "sheep",
    "aeroplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "motorbike",
    "train",
    "bottle",
    "chair",
    "dining table",
    "potted plant",
    "sofa",
    "tvmonitor",
]

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_bin_memmap(path, shape):
    """
    Load a binary weight file as a NumPy memmap array.

    Args:
        path (str): Filename of the binary weight file (without .bin extension)
        shape (tuple): Shape of the weight array

    Returns:
        np.memmap: Memory-mapped NumPy array of the weights
    """
    path = "./models/" + path + ".bin"
    return np.memmap(path, dtype=np.float32, mode="r", shape=shape)


def load_name_and_shape_from_txt(txt_path):
    """
    Load layer names and their shapes from a text file.

    Args:
        txt_path (str): Path to the text file containing layer names and shapes

    Returns:
        dict: Dictionary mapping sanitized layer names to their shapes
    """
    result = {}

    with open(txt_path, "r", encoding="utf-8") as txt_file:
        for line in txt_file:
            line = line.strip()
            if not line:
                continue

            name, shape_str = line.split("\t")
            name = name.replace(".", "_")
            shape = ast.literal_eval(shape_str)

            result[name] = shape

    return result


def add_batchnorm_2d(network, weight_map, lname, bn_input):
    """
    Add a 2D Batch Normalization layer to the network.

    Args:
        network (trt.INetworkDefinition): TensorRT network
        weight_map (dict): Mapping from layer names to their shapes
        lname (str): Base name of the batchnorm layer
        bn_input (trt.ITensor): Input tensor for batchnorm

    Returns:
        trt.IScaleLayer: BatchNorm layer
    """
    gamma_name = (lname + ".weight").replace(".", "_")
    beta_name = (lname + ".bias").replace(".", "_")
    mean_name = (lname + ".running_mean").replace(".", "_")
    var_name = (lname + ".running_var").replace(".", "_")

    gamma = load_bin_memmap(gamma_name, weight_map[gamma_name])
    beta = load_bin_memmap(beta_name, weight_map[beta_name])
    mean = load_bin_memmap(mean_name, weight_map[mean_name])
    var = load_bin_memmap(var_name, weight_map[var_name])

    eps = 1e-5
    scale = gamma / np.sqrt(var + eps)
    shift = beta - mean * scale
    power = np.ones_like(scale)

    bn = network.add_scale(
        input=bn_input,
        mode=trt.ScaleMode.CHANNEL,
        shift=shift,
        scale=scale,
        power=power,
    )
    return bn


def conv_bn_relu(
    network,
    weight_map,
    conv_bn_relu_input,
    conv_mid_name,
    bn_mid_name,
    out_ch,
    ksize=3,
    stride=1,
    padding=1,
):
    """
    Add a convolution layer followed by batchnorm and LeakyReLU activation.

    Args:
        network (trt.INetworkDefinition): TensorRT network
        weight_map (dict): Mapping from layer names to their shapes
        conv_bn_relu_input (trt.ITensor): Input tensor
        conv_mid_name (str): Convolution layer mid name
        bn_mid_name (str): BatchNorm layer mid name
        out_ch (int): Number of output channels
        ksize (int, optional): Kernel size. Defaults to 3.
        stride (int, optional): Stride. Defaults to 1.
        padding (int, optional): Padding. Defaults to 1.

    Returns:
        trt.IActivationLayer: Activated output tensor
    """
    conv = network.add_convolution_nd(
        input=conv_bn_relu_input,
        num_output_maps=out_ch,
        kernel_shape=(ksize, ksize),
        kernel=load_bin_memmap(
            "conv_layers_" + str(conv_mid_name) + "_weight",
            weight_map["conv_layers_" + str(conv_mid_name) + "_weight"],
        ),
        bias=load_bin_memmap(
            "conv_layers_" + str(conv_mid_name) + "_bias",
            weight_map["conv_layers_" + str(conv_mid_name) + "_bias"],
        ),
    )
    conv.stride_nd = (stride, stride)
    conv.padding_nd = (padding, padding)

    bn = add_batchnorm_2d(
        network, weight_map, "conv_layers." + bn_mid_name, conv.get_output(0)
    )

    relu = network.add_activation(bn.get_output(0), trt.ActivationType.LEAKY_RELU)
    relu.alpha = 0.1

    return relu


def fully_connected(
    network,
    fully_connection_input,
    weight_map,
    weight_name,
    bias_name,
    input_channel,
    output_channel,
):
    """
    Add a fully connected (dense) layer to the network.

    Args:
        network (trt.INetworkDefinition): TensorRT network
        fully_connection_input (trt.ITensor): Input tensor
        weight_map (dict): Mapping from layer names to their shapes
        weight_name (str): Weight layer name
        bias_name (str): Bias layer name
        input_channel (int): Number of input channel
        output_channel (int): Number of output channel

    Returns:
        trt.IElementWiseLayer: Output tensor after fully connected layer
    """

    # flatten
    flatten = network.add_shuffle(fully_connection_input.get_output(0))
    flatten.reshape_dims = (1, input_channel)

    fc_weight = load_bin_memmap(weight_name, weight_map[weight_name]).reshape(
        output_channel, input_channel
    )
    fc_weight_tensor = network.add_constant(
        fc_weight.shape, fc_weight.astype(np.float32)
    )

    # ===== MatrixMultiply (FC) =====
    fc_mm = network.add_matrix_multiply(
        input0=flatten.get_output(0),
        op0=trt.MatrixOperation.NONE,
        input1=fc_weight_tensor.get_output(0),
        op1=trt.MatrixOperation.TRANSPOSE,
    )

    # ===== Bias =====
    fc_bias = load_bin_memmap(bias_name, weight_map[bias_name]).reshape(
        1, output_channel
    )
    fc_bias_tensor = network.add_constant(fc_bias.shape, fc_bias.astype(np.float32))

    fc = network.add_elementwise(
        fc_mm.get_output(0), fc_bias_tensor.get_output(0), trt.ElementWiseOperation.SUM
    )

    return fc


def create_yolov1_engine(builder, config, dt):
    """
    Build and serialize a YOLOv1 TensorRT engine.

    Args:
        builder (trt.IBuilder): TensorRT builder
        config (trt.IBuilderConfig): Builder configuration
        dt (trt.DataType): Data type (e.g., trt.float32)

    Returns:
        bytes: Serialized engine
    """
    weight_map = load_name_and_shape_from_txt("./models/weight_name_and_shape.txt")

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    data = network.add_input(INPUT_BLOB_NAME, dt, (1, 3, INPUT_H, INPUT_W))

    print("The input data shape is :", data.shape)

    stage1 = conv_bn_relu(
        network,
        weight_map,
        data,
        conv_mid_name="0",
        bn_mid_name="1",
        out_ch=192,
        ksize=7,
        stride=2,
    )

    pool1 = network.add_pooling_nd(
        input=stage1.get_output(0),
        type=trt.PoolingType.MAX,
        window_size=trt.DimsHW(2, 2),
    )
    pool1.stride_nd = (2, 2)

    stage2 = conv_bn_relu(
        network,
        weight_map,
        pool1.get_output(0),
        conv_mid_name="4",
        bn_mid_name="5",
        out_ch=256,
    )
    pool2 = network.add_pooling_nd(
        input=stage2.get_output(0),
        type=trt.PoolingType.MAX,
        window_size=trt.DimsHW(2, 2),
    )
    pool2.stride_nd = (2, 2)

    stage3 = conv_bn_relu(
        network,
        weight_map,
        pool2.get_output(0),
        conv_mid_name="8",
        bn_mid_name="9",
        out_ch=512,
    )
    pool3 = network.add_pooling_nd(
        input=stage3.get_output(0),
        type=trt.PoolingType.MAX,
        window_size=trt.DimsHW(2, 2),
    )
    pool3.stride_nd = (2, 2)

    stage4 = conv_bn_relu(
        network,
        weight_map,
        pool3.get_output(0),
        conv_mid_name="12",
        bn_mid_name="13",
        out_ch=1024,
    )
    pool4 = network.add_pooling_nd(
        input=stage4.get_output(0),
        type=trt.PoolingType.MAX,
        window_size=trt.DimsHW(2, 2),
    )
    pool4.stride_nd = (2, 2)

    stage5 = conv_bn_relu(
        network,
        weight_map,
        pool4.get_output(0),
        conv_mid_name="16",
        bn_mid_name="17",
        out_ch=1024,
        stride=2,
    )

    stage6 = conv_bn_relu(
        network,
        weight_map,
        stage5.get_output(0),
        conv_mid_name="19",
        bn_mid_name="20",
        out_ch=1024,
    )

    fc1 = fully_connected(
        network,
        stage6,
        weight_map,
        "fc_layers_0_weight",
        "fc_layers_0_bias",
        50176,
        4096,
    )

    fc1_relu = network.add_activation(fc1.get_output(0), trt.ActivationType.LEAKY_RELU)
    fc1_relu.alpha = 0.1

    fc2 = fully_connected(
        network,
        fc1_relu,
        weight_map,
        "fc_layers_3_weight",
        "fc_layers_3_bias",
        4096,
        1470,
    )
    fc2_sig = network.add_activation(fc2.get_output(0), type=trt.ActivationType.SIGMOID)

    # reshape
    out = network.add_shuffle(fc2_sig.get_output(0))
    out.reshape_dims = (1, 7, 7, 30)

    out.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(out.get_output(0))

    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)

    return serialized_engine


def api_to_model():
    """
    Build YOLOv1 engine and save it to a file.

    This function constructs the TensorRT engine and writes the serialized engine
    to ENGINE_PATH.
    """
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    yolov1_engine = create_yolov1_engine(builder, config, trt.float32)
    assert yolov1_engine
    with open(ENGINE_PATH, "wb") as engine_file:
        engine_file.write(yolov1_engine)


def do_inference(exec_context, host_input, host_output):
    """
    Run inference on input data using a TensorRT execution context.

    Args:
        exec_context (trt.IExecutionContext): TensorRT execution context
        host_input (np.ndarray): Input array to be copied to device
        host_output (np.ndarray): Output array to store device results
    """
    trt_engine = exec_context.engine
    print("num io tensors is:", trt_engine.num_io_tensors)

    input_name = trt_engine.get_tensor_name(0)
    output_name = trt_engine.get_tensor_name(1)

    print("Input tensor name is :", input_name)
    print("output tensor name is :", output_name)

    device_in = cuda.mem_alloc(host_input.nbytes)
    device_out = cuda.mem_alloc(host_output.nbytes)

    stream = cuda.Stream()

    cuda.memcpy_htod_async(device_in, host_input, stream)

    exec_context.set_tensor_address(input_name, int(device_in))
    exec_context.set_tensor_address(output_name, int(device_out))

    exec_context.execute_async_v3(stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(host_output, device_out, stream)

    stream.synchronize()


def main():
    """
    Main script entry point.

    Handles command line arguments:
        - -s: Serialize YOLOv1 model to TensorRT engine
        - -d: Deserialize engine and run inference on test.jpg
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", action="store_true")
    parser.add_argument("-d", action="store_true")
    args = parser.parse_args()

    if not args.s ^ args.d:
        print("arguments not right!")
        print("python lenet.py -s   # serialize model to plan file")
        print("python lenet.py -d   # deserialize plan file and run inference")
        sys.exit()

    if args.s:
        api_to_model()
    else:
        runtime = trt.Runtime(TRT_LOGGER)
        assert runtime

        with open(ENGINE_PATH, "rb") as fp:
            loaded_engine = runtime.deserialize_cuda_engine(fp.read())
        assert loaded_engine

        context = loaded_engine.create_execution_context()
        assert context

        img = cv2.imread("test.jpg")
        preprocessed_data = preprocess(img, INPUT_H)

        host_in = cuda.pagelocked_empty(preprocessed_data.size, dtype=np.float32)
        np.copyto(host_in, preprocessed_data.ravel())
        host_out = cuda.pagelocked_empty(OUTPUT_SIZE, dtype=np.float32)

        do_inference(context, host_in, host_out)

        xywhcc = pred2xywhcc(host_out, S, NUM_CLASSES, CONF_THRESH, IOU_THRESH)

        if xywhcc.shape[0] != 0:
            img = draw_bbox(img, xywhcc, class_names)
            # save output img
            img_name = "output.jpg"
            cv2.imwrite(os.path.join("./", img_name), img)


if __name__ == "__main__":
    main()
