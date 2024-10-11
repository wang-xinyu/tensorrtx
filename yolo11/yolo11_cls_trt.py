"""
An example that uses TensorRT's Python api to make inferences.
"""
import os
import shutil
import sys
import threading
import time
import cv2
import numpy as np
import torch
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt


def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret


with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]


class YoLo11TRT(object):
    """
    description: A YOLO11 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        for binding in engine:
            print('binding:', binding, engine.get_binding_shape(binding))
            self.batch_size = engine.get_binding_shape(binding)[0]
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def infer(self, raw_image_generator):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_image_raw = []
        batch_input_image = np.empty(
            shape=[self.batch_size, 3, self.input_h, self.input_w])
        for i, image_raw in enumerate(raw_image_generator):
            batch_image_raw.append(image_raw)
            input_image = self.preprocess_cls_image(image_raw)
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        for i in range(self.batch_size):
            classes_ls, predicted_conf_ls, category_id_ls = self.postprocess_cls(
                output)
            cv2.putText(batch_image_raw[i], str(
                classes_ls), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            print(classes_ls, predicted_conf_ls)
        return batch_image_raw, end - start

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)

    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_cls_image(self, raw_bgr_image, dst_width=224, dst_height=224):

        """
            description: Convert BGR image to RGB,
                         crop the center square frame,
                         resize it to target size, normalize to [0,1],
                         transform to NCHW format.
            param:
                raw_bgr_image: numpy array, raw BGR image
                dst_width: int, target image width
                dst_height: int, target image height
            return:
                image:  the processed image
                image_raw: the original image
                h: original height
                w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        # Crop the center square frame
        m = min(h, w)
        top = (h - m) // 2
        left = (w - m) // 2
        image = raw_bgr_image[top:top + m, left:left + m]

        # Resize the image with target size while maintaining ratio
        image = cv2.resize(image, (dst_width, dst_height), interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to [0,1]
        image = image.astype(np.float32) / 255.0

        # HWC to CHW format
        image = image.transpose(2, 0, 1)

        # CHW to NCHW format (add batch dimension)
        image = np.expand_dims(image, axis=0)

        # Convert the image to row-major order, also known as "C order"
        image = np.ascontiguousarray(image)

        batch_data = np.expand_dims(image, axis=0)

        return batch_data

    def postprocess_cls(self, output_data):
        classes_ls = []
        predicted_conf_ls = []
        category_id_ls = []
        output_data = output_data.reshape(self.batch_size, -1)
        output_data = torch.Tensor(output_data)
        p = torch.nn.functional.softmax(output_data, dim=1)
        score, index = torch.topk(p, 3)
        for ind in range(index.shape[0]):
            input_category_id = index[ind][0].item()  # 716
            category_id_ls.append(input_category_id)
            predicted_confidence = score[ind][0].item()
            predicted_conf_ls.append(predicted_confidence)
            classes_ls.append(classes[input_category_id])
        return classes_ls, predicted_conf_ls, category_id_ls


class inferThread(threading.Thread):
    def __init__(self, yolo11_wrapper, image_path_batch):
        threading.Thread.__init__(self)
        self.yolo11_wrapper = yolo11_wrapper
        self.image_path_batch = image_path_batch

    def run(self):
        batch_image_raw, use_time = self.yolo11_wrapper.infer(
            self.yolo11_wrapper.get_raw_image(self.image_path_batch))
        for i, img_path in enumerate(self.image_path_batch):
            parent, filename = os.path.split(img_path)
            save_name = os.path.join('output', filename)
            # Save image
            cv2.imwrite(save_name, batch_image_raw[i])
        print('input->{}, time->{:.2f}ms, saving into output/'.format(
            self.image_path_batch, use_time * 1000))


class warmUpThread(threading.Thread):
    def __init__(self, yolo11_wrapper):
        threading.Thread.__init__(self)
        self.yolo11_wrapper = yolo11_wrapper

    def run(self):
        batch_image_raw, use_time = self.yolo11_wrapper.infer(
            self.yolo11_wrapper.get_raw_image_zeros())
        print(
            'warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))


if __name__ == "__main__":
    # load custom plugin and engine
    engine_file_path = "./yolo11x-cls-fp32.engine"

    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]

    if os.path.exists('output/'):
        shutil.rmtree('output/')
    os.makedirs('output/')
    # a YoLo11TRT instance
    yolo11_wrapper = YoLo11TRT(engine_file_path)
    try:
        print('batch size is', yolo11_wrapper.batch_size)

        image_dir = "images/"
        image_path_batches = get_img_path_batches(
            yolo11_wrapper.batch_size, image_dir)

        for i in range(10):
            # create a new thread to do warm_up
            thread1 = warmUpThread(yolo11_wrapper)
            thread1.start()
            thread1.join()
        for batch in image_path_batches:
            # create a new thread to do inference
            thread1 = inferThread(yolo11_wrapper, batch)
            thread1.start()
            thread1.join()
    finally:
        # destroy the instance
        yolo11_wrapper.destroy()
