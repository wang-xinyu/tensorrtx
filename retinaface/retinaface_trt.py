"""
Use TensorRT's Python api to make inferences.
"""
# -*- coding: utf-8 -*
import ctypes
import os
import random
import sys
import threading
import time

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision

INPUT_H = 480  #defined in decode.h
INPUT_W = 640
CONF_THRESH = 0.75
IOU_THRESHOLD = 0.4
np.set_printoptions(threshold=np.inf)

def plot_one_box(x, landmark,img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,

    param:
        x:     a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
            line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness

    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    cv2.circle(img, (int(landmark[0]), int(landmark[1])), 1, (0, 0, 255), 4)
    cv2.circle(img, (int(landmark[2]), int(landmark[3])), 1, (0, 255, 255), 4)
    cv2.circle(img, (int(landmark[4]), int(landmark[5])), 1, (255, 0, 255), 4)
    cv2.circle(img, (int(landmark[6]), int(landmark[7])), 1, (0, 255, 0), 4)
    cv2.circle(img, (int(landmark[8]), int(landmark[9])), 1, (255, 0, 0), 4)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class Retinaface_trt(object):
    """
    description: A Retineface class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.cfx = cuda.Device(0).make_context()
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

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
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

    def infer(self, input_image_path):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.

        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(
            input_image_path
        )
        a = time.time()
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]

        # Do postprocess
        result_boxes, result_scores, result_landmark = self.post_process(
            output, origin_h, origin_w
        )
        b = time.time()-a
        print(b)

        # Draw rectangles and labels on the original image

        # Save image
        for i in range(len(result_boxes)):
            box = result_boxes[i]
            landmark = result_landmark[i]
            plot_one_box(
                box,
                landmark,
                image_raw,
                label="{}:{:.2f}".format( 'Face', result_scores[i]))
        parent, filename = os.path.split(input_image_path)
        save_name = os.path.join(parent, "output_" + filename)

        cv2.imwrite(save_name, image_raw)

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

    def preprocess_image(self, input_image_path):
        """
        description: Read an image from image path, resize and pad it to target size,
                     normalize to [0,1],transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = cv2.imread(input_image_path)
        h, w, c = image_raw.shape

        # Calculate widht and height and paddings
        r_w = INPUT_W / w
        r_h = INPUT_H / h
        if r_h > r_w:
            tw = INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((INPUT_H - th) / 2)
            ty2 = INPUT_H - th - ty1
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx1 = int((INPUT_W - tw) / 2)
            tx2 = INPUT_W - tw - tx1
            ty1 = ty2 = 0

        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image_raw, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)

        # HWC to CHW format:
        image -= (104, 117, 123)
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x,landmark):

        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

        r_w = INPUT_W / origin_w
        r_h = INPUT_H / origin_h

        if r_h > r_w:
            y[:, 0] = x[:, 0] / r_w
            y[:, 2] = x[:, 2] / r_w
            y[:, 1] = (x[:, 1] - (INPUT_H - r_w * origin_h) / 2) / r_w
            y[:, 3] = (x[:, 3] - (INPUT_H - r_w * origin_h) / 2) / r_w
            
            landmark[:,0] = landmark[:,0]/r_w
            landmark[:,1] = (landmark[:,1] - (INPUT_H - r_w * origin_h) / 2)/r_w
            landmark[:,2] = landmark[:,2]/r_w
            landmark[:,3] = (landmark[:,3] - (INPUT_H - r_w * origin_h) / 2)/r_w
            landmark[:,4] = landmark[:,4]/r_w
            landmark[:,5] = (landmark[:,5] - (INPUT_H - r_w * origin_h) / 2)/r_w
            landmark[:,6] = landmark[:,6]/r_w
            landmark[:,7] = (landmark[:,7] - (INPUT_H - r_w * origin_h) / 2)/r_w
            landmark[:,8] = landmark[:,8]/r_w
            landmark[:,9] = (landmark[:,9] - (INPUT_H - r_w * origin_h) / 2)/r_w
        else:
            y[:, 0] = (x[:, 0] - (INPUT_W - r_h * origin_w) / 2) / r_h
            y[:, 2] = (x[:, 2] - (INPUT_W - r_h * origin_w) / 2) / r_h
            y[:, 1] = x[:, 1] /r_h
            y[:, 3] = x[:, 3] /r_h

            landmark[:,0] = (landmark[:,0] - (INPUT_W - r_h * origin_w) / 2)/r_h
            landmark[:,1] = landmark[:,1]/ r_h
            landmark[:,2] = (landmark[:,2] - (INPUT_W - r_h * origin_w) / 2)/r_h
            landmark[:,3] = landmark[:,3]/ r_h
            landmark[:,4] = (landmark[:,4] - (INPUT_W - r_h * origin_w) / 2)/r_h
            landmark[:,5] = landmark[:,5]/ r_h
            landmark[:,6] = (landmark[:,6] - (INPUT_W - r_h * origin_w) / 2)/r_h
            landmark[:,7] = landmark[:,7]/ r_h
            landmark[:,8] = (landmark[:,8] - (INPUT_W - r_h * origin_w) / 2)/r_h
            landmark[:,9] = landmark[:,9]/ r_h

        return y, landmark

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,x1,y1,x2,y2,conf,landmark_x1,landmark_y1,
            landmark_x2,landmark_y2,...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 15))[:num, :]
        # to  torch Tensor
        pred = torch.Tensor(pred).cuda()
        # Get the boxes
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4]
        # Get the landmark
        landmark = pred[:,5:15]
        # Choose those boxes that score > CONF_THRESH
        si = scores > CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si]

        landmark = landmark[si,:]

        # Get boxes and landmark
        boxes,landmark = self.xywh2xyxy(origin_h, origin_w, boxes,landmark)
        # Do nms
        indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_landmark = landmark[indices].cpu()
        return result_boxes, result_scores, result_landmark

class myThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)

if __name__ == "__main__":
    # load custom plugins,make sure it has been generated
    PLUGIN_LIBRARY = "build/libdecodeplugin.so"
    ctypes.CDLL(PLUGIN_LIBRARY)
    engine_file_path = "build/retina_r50.engine"

    retinaface = Retinaface_trt(engine_file_path)
    input_image_paths = ["zidane.jpg"]
    for i in range(10):
        for input_image_path in input_image_paths:
            # create a new thread to do inference
            thread = myThread(retinaface.infer, [input_image_path])
            thread.start()
            thread.join()

    # destroy the instance
    retinaface.destroy()
