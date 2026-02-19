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
INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"
EPS = 1e-5

WEIGHT_PATH = "./mobilenet.wts"
ENGINE_PATH = "./mobilenetv2.plan"

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


def conv_bn_relu(network, weight_map, input, outch, ksize, s, g, lname):
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
    assert bn1

    relu1 = network.add_activation(bn1.get_output(0), type=trt.ActivationType.RELU)
    assert relu1

    shift = np.array([-6.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    power = np.array([1.0], dtype=np.float32)
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
        conv1 = network.add_convolution_nd(input=ew2.get_output(0),
                                           num_output_maps=outch,
                                           kernel_shape=(1, 1),
                                           kernel=weight_map[lname + "conv.2.weight"],
                                           bias=trt.Weights())
        assert conv1
        bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + "conv.3", EPS)
    else:
        ew1 = conv_bn_relu(network, weight_map, input, hidden, 3, s, hidden, lname + "conv.0.")
        conv1 = network.add_convolution_nd(input=ew1.get_output(0),
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
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    data = network.add_input(INPUT_BLOB_NAME, dt, (max_batch_size, 3, INPUT_H, INPUT_W))
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

    pool1 = network.add_pooling_nd(input=ew2.get_output(0),
                                   type=trt.PoolingType.AVERAGE,
                                   window_size=trt.DimsHW(7, 7))
    assert pool1

    input_reshape = network.add_shuffle(pool1.get_output(0))
    input_reshape.reshape_dims = (max_batch_size, 1280)

    # Add weight as constant layer
    fc_weights = weight_map["classifier.1.weight"].reshape(OUTPUT_SIZE, 1280)
    filter_const = network.add_constant((OUTPUT_SIZE, 1280), fc_weights)

    # Matrix multiplication
    mm = network.add_matrix_multiply(input_reshape.get_output(0), trt.MatrixOperation.NONE,
                                     filter_const.get_output(0), trt.MatrixOperation.TRANSPOSE)
    assert mm

    # Add bias
    bias_weights = weight_map["classifier.1.bias"].reshape(1, OUTPUT_SIZE)
    bias_const = network.add_constant((1, OUTPUT_SIZE), bias_weights)

    add1 = network.add_elementwise(mm.get_output(0), bias_const.get_output(0), trt.ElementWiseOperation.SUM)
    assert add1

    add1.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(add1.get_output(0))

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    serialized_engine = builder.build_serialized_network(network, config)

    del network
    del weight_map

    return serialized_engine


def API_to_model(max_batch_size):
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    serialized_engine = create_engine(max_batch_size, builder, config, trt.float32)
    assert serialized_engine

    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized_engine)

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
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v3(stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", action='store_true')
    parser.add_argument("-d", action='store_true')
    parser.add_argument("--image", default=None, type=str, help="Image to run inference.")
    args = parser.parse_args()

    if not (args.s ^ args.d):
        print(
            "arguments not right!\n"
            "python mobilenet_v2.py -s                      # serialize model to plan file\n"
            "python mobilenet_v2.py -d                      # deserialize plan file and run inference\n"
            "python mobilenet_v2.py -d --image image.jpg    # deserialize plan file and run inference"
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

        if "--image" in sys.argv or any(a.startswith("--image") for a in sys.argv):
            # arg already declared in argparse; use it
            if args.image and os.path.exists(args.image):
                data = preprocess_image(args.image)  # (1,3,H,W)
                print(f"Using image: {args.image}")
            else:
                print("Image not found, using all-ones test data")
                data = np.ones((BATCH_SIZE, 3, INPUT_H, INPUT_W), dtype=np.float32)
        else:
            data = np.ones((BATCH_SIZE, 3, INPUT_H, INPUT_W), dtype=np.float32)
            print("No image provided, using all-ones test data")

        inputs, outputs, stream = allocate_buffers(engine, context)

        inputs[0].host = data.ravel()

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

        del context
        del engine
        del runtime
