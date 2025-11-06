import argparse
import os
import struct
import sys

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

INPUT_H = 32
INPUT_W = 32
OUTPUT_SIZE = 10
INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"

weight_path = "./lenet5.wts"
engine_path = "./lenet5.engine"

gLogger = trt.Logger(trt.Logger.INFO)


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


def createLenetEngine(maxBatchSize, builder, config, dt):
    weight_map = load_weights(weight_path)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    data = network.add_input(INPUT_BLOB_NAME, dt, (1, 1, INPUT_H, INPUT_W))
    assert data

    conv1 = network.add_convolution_nd(input=data,
                                       num_output_maps=6,
                                       kernel_shape=(5, 5),
                                       kernel=weight_map["conv1.weight"],
                                       bias=weight_map["conv1.bias"])
    assert conv1
    conv1.stride_nd = (1, 1)

    relu1 = network.add_activation(conv1.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu1

    pool1 = network.add_pooling_nd(input=relu1.get_output(0),
                                   window_size=trt.DimsHW(2, 2),
                                   type=trt.PoolingType.AVERAGE)
    assert pool1
    pool1.stride_nd = (2, 2)

    conv2 = network.add_convolution_nd(pool1.get_output(0), 16, trt.DimsHW(5, 5),
                                       weight_map["conv2.weight"],
                                       weight_map["conv2.bias"])
    assert conv2
    conv2.stride_nd = (1, 1)

    relu2 = network.add_activation(conv2.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu2

    pool2 = network.add_pooling_nd(input=relu2.get_output(0),
                                   window_size=trt.DimsHW(2, 2),
                                   type=trt.PoolingType.AVERAGE)
    assert pool2
    pool2.stride_nd = (2, 2)

    # flatten pool2
    flatten_layer = network.add_shuffle(pool2.get_output(0))
    flatten_layer.reshape_dims = (1, 16 * 5 * 5,)
    flatten_output = flatten_layer.get_output(0)

    weight_map['fc1.weight'] = weight_map['fc1.weight'].reshape(120, 400)
    weight_tensor = network.add_constant(weight_map['fc1.weight'].shape, weight_map['fc1.weight'].astype(np.float32))

    fc1_matrix_multiply = network.add_matrix_multiply(input0=flatten_output,
                                                      op0=trt.MatrixOperation.NONE,
                                                      input1=weight_tensor.get_output(0),
                                                      op1=trt.MatrixOperation.TRANSPOSE)
    assert fc1_matrix_multiply

    weight_map['fc1.bias'] = weight_map['fc1.bias'].reshape(1, 120)
    bias_tensor = network.add_constant(weight_map['fc1.bias'].shape, weight_map['fc1.bias'].astype(np.float32))
    fc1 = network.add_elementwise(fc1_matrix_multiply.get_output(0),
                                  bias_tensor.get_output(0),
                                  trt.ElementWiseOperation.SUM)
    assert fc1

    relu3 = network.add_activation(fc1.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu3

    weight_map['fc2.weight'] = weight_map['fc2.weight'].reshape(84, 120)
    fc2_weight_tensor = network.add_constant(weight_map['fc2.weight'].shape,
                                             weight_map['fc2.weight'].astype(np.float32))

    fc2_matrix_multiply = network.add_matrix_multiply(input0=relu3.get_output(0),
                                                      op0=trt.MatrixOperation.NONE,
                                                      input1=fc2_weight_tensor.get_output(0),
                                                      op1=trt.MatrixOperation.TRANSPOSE)
    assert fc2_matrix_multiply

    weight_map['fc2.bias'] = weight_map['fc2.bias'].reshape(1, 84)
    fc2_bias_tensor = network.add_constant(weight_map['fc2.bias'].shape, weight_map['fc2.bias'].astype(np.float32))
    fc2 = network.add_elementwise(fc2_matrix_multiply.get_output(0),
                                  fc2_bias_tensor.get_output(0),
                                  trt.ElementWiseOperation.SUM)
    assert fc2

    relu4 = network.add_activation(fc2.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu4

    weight_map['fc3.weight'] = weight_map['fc3.weight'].reshape(10, 84)
    fc3_weight_tensor = network.add_constant(weight_map['fc3.weight'].shape,
                                             weight_map['fc3.weight'].astype(np.float32))

    fc3_matrix_multiply = network.add_matrix_multiply(input0=relu4.get_output(0),
                                                      op0=trt.MatrixOperation.NONE,
                                                      input1=fc3_weight_tensor.get_output(0),
                                                      op1=trt.MatrixOperation.TRANSPOSE)
    assert fc3_matrix_multiply

    weight_map['fc3.bias'] = weight_map['fc3.bias'].reshape(1, 10)
    fc3_bias_tensor = network.add_constant(weight_map['fc3.bias'].shape, weight_map['fc3.bias'].astype(np.float32))
    fc3 = network.add_elementwise(fc3_matrix_multiply.get_output(0),
                                  fc3_bias_tensor.get_output(0),
                                  trt.ElementWiseOperation.SUM)
    assert fc3

    prob = network.add_softmax(fc3.get_output(0))
    prob.axes = 1 << 1    # dim=1
    assert prob

    prob.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(prob.get_output(0))

    # Build engine
    engine = builder.build_serialized_network(network, config)

    return engine


def APIToModel(maxBatchSize):
    builder = trt.Builder(gLogger)
    config = builder.create_builder_config()
    engine = createLenetEngine(maxBatchSize, builder, config, trt.float32)
    assert engine
    with open(engine_path, "wb") as f:
        f.write(engine)

    del engine
    del builder


def doInference(context, host_in, host_out, batchSize):
    engine = context.engine
    assert engine.num_io_tensors == 2

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    device_in = cuda.mem_alloc(host_in.nbytes)
    device_out = cuda.mem_alloc(host_out.nbytes)
    stream = cuda.Stream()

    cuda.memcpy_htod_async(device_in, host_in, stream)

    context.set_tensor_address(input_name, int(device_in))
    context.set_tensor_address(output_name, int(device_out))

    context.execute_async_v3(stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(host_out, device_out, stream)

    stream.synchronize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", action='store_true')
    parser.add_argument("-d", action='store_true')
    args = parser.parse_args()

    if not (args.s ^ args.d):
        print("arguments not right!")
        print("python lenet.py -s   # serialize model to plan file")
        print("python lenet.py -d   # deserialize plan file and run inference")
        sys.exit()

    if args.s:
        APIToModel(1)
    else:
        runtime = trt.Runtime(gLogger)
        assert runtime

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine

        context = engine.create_execution_context()
        assert context

        data = np.ones((INPUT_H * INPUT_W), dtype=np.float32)
        host_in = cuda.pagelocked_empty(INPUT_H * INPUT_W, dtype=np.float32)
        np.copyto(host_in, data.ravel())
        host_out = cuda.pagelocked_empty(OUTPUT_SIZE, dtype=np.float32)

        doInference(context, host_in, host_out, 1)

        print(f'Output: {host_out}')
