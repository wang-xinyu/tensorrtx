import argparse
import os
import numpy as np
import struct

# required for the model creation
import tensorrt as trt

# required for the inference using TRT engine
import pycuda.driver as cuda

# Sizes of input and output for TensorRT model
INPUT_SIZE = 1
OUTPUT_SIZE = 1

# path of .wts (weight file) and .engine (model file)
WEIGHT_PATH = "./mlp.wts"
ENGINE_PATH = "./mlp.engine"

# input and output names are must for the TRT model
INPUT_BLOB_NAME = 'data'
OUTPUT_BLOB_NAME = 'out'

# A logger provided by NVIDIA-TRT
gLogger = trt.Logger(trt.Logger.INFO)


################################
# DEPLOYMENT RELATED ###########
################################
def load_weights(file_path):
    """
    Parse the .wts file and store weights in dict format
    :param file_path:
    :return weight_map: dictionary containing weights and their values
    """
    print(f"[INFO]: Loading weights: {file_path}")
    assert os.path.exists(file_path), '[ERROR]: Unable to load weight file.'

    weight_map = {}
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f]

    # count for total # of weights
    count = int(lines[0])
    assert count == len(lines) - 1

    # Loop through counts and get the exact num of values against weights
    for i in range(1, count + 1):
        splits = lines[i].split(" ")
        name = splits[0]
        cur_count = int(splits[1])

        # len of splits must be greater than current weight counts
        assert cur_count + 2 == len(splits)

        # loop through all weights and unpack from the hexadecimal values
        values = []
        for j in range(2, len(splits)):
            # hex string to bytes to float
            values.append(struct.unpack(">f", bytes.fromhex(splits[j])))

        # store in format of { 'weight.name': [weights_val0, weight_val1, ..] }
        weight_map[name] = np.array(values, dtype=np.float32)

    return weight_map


def create_mlp_engine(max_batch_size, builder, config, dt):
    """
    Create Multi-Layer Perceptron using the TRT Builder and Configurations
    :param max_batch_size: batch size for built TRT model
    :param builder: to build engine and networks
    :param config: configuration related to Hardware
    :param dt: datatype for model layers
    :return engine: TRT model
    """
    print("[INFO]: Creating MLP using TensorRT...")
    # load weight maps from the file
    weight_map = load_weights(WEIGHT_PATH)

    # build an empty network using builder
    network = builder.create_network()

    # add an input to network using the *input-name
    data = network.add_input(INPUT_BLOB_NAME, dt, (1, INPUT_SIZE))
    assert data

    # add the layer with output-size (number of outputs)
    # 1. Perform matrix multiplication
    weight_tensor = network.add_constant(
        weight_map['linear.weight'].shape, weight_map['linear.weight'].astype(np.float32))
    matrix_multiply = network.add_matrix_multiply(input0=data,
                                                  op0=trt.MatrixOperation.NONE,
                                                  input1=weight_tensor.get_output(0),
                                                  op1=trt.MatrixOperation.NONE)
    assert matrix_multiply

    # 2. Add the bias as elementwise addition (W * X + b)
    bias_tensor = network.add_constant(weight_map['linear.bias'].shape, weight_map['linear.bias'].astype(np.float32))
    bias_layer = network.add_elementwise(matrix_multiply.get_output(0),
                                         bias_tensor.get_output(0),
                                         trt.ElementWiseOperation.SUM)
    assert bias_layer

    # set the name for output layer
    bias_layer.get_output(0).name = OUTPUT_BLOB_NAME

    # mark this layer as final output layer
    network.mark_output(bias_layer.get_output(0))

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("Failed to build serialized TensorRT engine")

    print("[INFO]: MLP Engine built successfully!")
    return engine


def api_to_model(max_batch_size):
    """
    Create engine using TensorRT APIs
    :param max_batch_size: for the deployed model configs
    :return:
    """
    # Create Builder with logger provided by TRT
    builder = trt.Builder(gLogger)

    # Create configurations from Engine Builder
    config = builder.create_builder_config()

    # Create MLP Engine
    engine = create_mlp_engine(max_batch_size, builder, config, trt.float32)
    assert engine

    # Write the engine into binary file
    print("[INFO]: Writing engine into binary...")
    with open(ENGINE_PATH, "wb") as f:
        # write serialized model in file
        f.write(engine)


################################
# INFERENCE RELATED ############
################################
def perform_inference(input_val):
    """
    Get inference using the pre-trained model
    :param input_val: a number as an input
    :return:
    """

    def do_inference(inf_context, inf_host_in, inf_host_out):
        """
        Perform inference using the CUDA context
        :param inf_context: context created by engine
        :param inf_host_in: input from the host
        :param inf_host_out: output to save on host
        :return:
        """

        inference_engine = inf_context.engine
        # Input and output bindings are required for inference
        assert inference_engine.num_io_tensors == 2

        # allocate memory in GPU using CUDA bindings
        device_in = cuda.mem_alloc(inf_host_in.nbytes)
        device_out = cuda.mem_alloc(inf_host_out.nbytes)

        # create bindings for input and output
        bindings = [int(device_in), int(device_out)]

        # create CUDA stream for simultaneous CUDA operations
        stream = cuda.Stream()

        # copy input from host (CPU) to device (GPU)  in stream
        cuda.memcpy_htod_async(device_in, inf_host_in, stream)

        # Set the tensor addresses for input and output
        for i, binding in enumerate(bindings):
            # Use the tensor name obtained from the engine to set the address
            inf_context.set_tensor_address(inference_engine.get_tensor_name(i), binding)

        # execute inference using context provided by engine
        inf_context.execute_async_v3(stream_handle=stream.handle)

        # copy output back from device (GPU) to host (CPU)
        cuda.memcpy_dtoh_async(inf_host_out, device_out, stream)

        # synchronize the stream to prevent issues
        #       (block CUDA and wait for CUDA operations to be completed)
        stream.synchronize()

    # create a runtime (required for deserialization of model) with NVIDIA's logger
    runtime = trt.Runtime(gLogger)
    assert runtime

    # read and deserialize engine for inference
    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine

    # create execution context -- required for inference executions
    context = engine.create_execution_context()
    assert context

    # create input as array
    data = np.array([input_val], dtype=np.float32)

    # capture free memory for input in GPU
    host_in = cuda.pagelocked_empty((INPUT_SIZE), dtype=np.float32)

    # copy input-array from CPU to Flatten array in GPU
    np.copyto(host_in, data.ravel())

    # capture free memory for output in GPU
    host_out = cuda.pagelocked_empty(OUTPUT_SIZE, dtype=np.float32)

    # do inference using required parameters
    do_inference(context, host_in, host_out)

    print(f'\n[INFO]: Predictions using pre-trained model..\n\tInput:\t{input_val}\n\tOutput:\t{host_out[0]:.4f}')


def get_args():
    """
    Parse command line arguments
    :return arguments: parsed arguments
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-s', action='store_true')
    arg_parser.add_argument('-d', action='store_true')
    arguments = vars(arg_parser.parse_args())
    # check for the arguments
    if not (arguments['s'] ^ arguments['d']):
        print("[ERROR]: Arguments not right!\n")
        print("\tpython mlp.py -s   # serialize model to engine file")
        print("\tpython mlp.py -d   # deserialize engine file and run inference")
        exit()

    return arguments


if __name__ == "__main__":
    args = get_args()
    if args['s']:
        api_to_model(max_batch_size=1)
        print("[INFO]: Successfully created TensorRT engine...")
        print("\n\tRun inference using `python mlp.py -d`\n")
    else:
        perform_inference(input_val=4.0)
