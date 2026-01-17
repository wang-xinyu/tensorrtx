import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import ctypes
import os
import sys

def load_imagenet_labels(label_file="imagenet_classes.txt"):
    """Load ImageNet class labels"""
    if not os.path.exists(label_file):
        return None
    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def main(engine_path, img_path, label_file="imagenet_classes.txt"):
    # Load plugin library
    so_file = os.path.abspath("./build/liblayernorm_plugin.so")
    if not os.path.exists(so_file):
        print(f"Plugin library not found: {so_file}")
        return
    
    ctypes.CDLL(so_file)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    if not os.path.exists(engine_path):
        print(f"Engine file not found: {engine_path}")
        return

    with open(engine_path, "rb") as f:
        serialized_engine = f.read()
        
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if not engine:
        print("Failed to deserialize engine.")
        return

    context = engine.create_execution_context()
    
    # Get Input Shape from Engine
    input_shape = (224, 224) # Default
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            shape = engine.get_binding_shape(i)
            # shape is usually (N, C, H, W) or (C, H, W)
            # We assume explicit batch if N is present
            if len(shape) == 4:
                input_shape = (shape[2], shape[3])
            elif len(shape) == 3:
                input_shape = (shape[1], shape[2])
            break

    # Prepare input
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[1], input_shape[0])) # cv2.resize takes (W, H)
    img = img.astype(np.float32) / 255.0
    
    # ImageNet Mean/Std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    img = img.transpose(2, 0, 1) # HWC -> CHW
    img = np.expand_dims(img, axis=0) # CHW -> NCHW
    img = np.ascontiguousarray(img)

    # Allocate buffers
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    
    for i in range(engine.num_bindings):
        tensor_name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        shape = engine.get_binding_shape(i)
        
        # Handle dynamic shape or fixed
        # Check if input or output
        is_input = engine.binding_is_input(i)
        
        # Since we use explicit batch, shape[0] might be -1 or 1
        # If -1, we set context binding shape
        if shape[0] == -1:
             shape = (1,) + shape[1:]
             context.set_binding_shape(i, shape)
        
        size = trt.volume(shape) * np.dtype(dtype).itemsize
        
        # Host memory
        host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype)
        # Device memory
        device_mem = cuda.mem_alloc(size)
        
        bindings.append(int(device_mem))
        
        if is_input:
            inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            # Copy input data to host buffer
            np.copyto(host_mem, img.ravel())
        else:
            outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})

    # Inference
    # Transfer input data to the GPU.
    for inp in inputs:
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
        
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    
    # Transfer predictions back from the GPU.
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        
    # Synchronize the stream
    stream.synchronize()
    
    # Process output
    labels = load_imagenet_labels(label_file)
    for out in outputs:
        output_data = out['host']
        max_idx = np.argmax(output_data)
        max_val = output_data[max_idx]
        if labels:
            print(f"Predicted Class: {max_idx} - {labels[max_idx]} (Score: {max_val})")
        else:
            print(f"Predicted Class: {max_idx} (Score: {max_val})")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(f"Usage: python {sys.argv[0]} <engine_path> <image_path> [label_file]")
        print(f"Example: python {sys.argv[0]} convnextv2.engine images/test.jpg")
        print(f"         python {sys.argv[0]} convnextv2.engine images/test.jpg custom_labels.txt")
        sys.exit(1)
    
    engine_path = sys.argv[1]
    img_path = sys.argv[2]
    label_file = sys.argv[3] if len(sys.argv) == 4 else "imagenet_classes.txt"
    main(engine_path, img_path, label_file)
