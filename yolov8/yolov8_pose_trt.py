"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4
POSE_NUM = 17 * 3
DET_NUM = 6
SEG_NUM = 32
OBB_NUM = 1
keypoint_pairs = [
    (0, 1), (0, 2), (0, 5), (0, 6), (1, 2),
    (1, 3), (2, 4), (5, 6), (5, 7), (5, 11),
    (6, 8), (6, 12), (7, 9), (8, 10), (11, 12),
    (11, 13), (12, 14), (13, 15), (14, 16)
]


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


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov8 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
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


class YoLov8TRT(object):
    """
    description: A YOLOv8 class that warps TensorRT ops, preprocess and postprocess ops.
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

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
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
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
        self.det_output_size = host_outputs[0].shape[0]

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
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        for i, image_raw in enumerate(raw_image_generator):
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i],
                      input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
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

            result_boxes, result_scores, result_classid, keypoints = self.post_process(
                output[i * (self.det_output_size): (i + 1) * (self.det_output_size)],
                batch_origin_h[i], batch_origin_w[i]
            )

            # Draw rectangles and labels on the original image
            for j in range(len(result_boxes)):
                box = result_boxes[j]
                plot_one_box(
                    box,
                    batch_image_raw[i],
                    label="{}:{:.2f}".format(
                        categories[int(result_classid[j])], result_scores[j]
                    ),
                )

                num_keypoints = len(keypoints[j]) // 3
                points = []
                for k in range(num_keypoints):
                    x = keypoints[j][k * 3]
                    y = keypoints[j][k * 3 + 1]
                    confidence = keypoints[j][k * 3 + 2]
                    if confidence > 0:
                        points.append((int(x), int(y)))
                    else:
                        points.append(None)

                # 根据关键点索引对绘制线条
                for pair in keypoint_pairs:
                    partA, partB = pair
                    if points[partA] and points[partB]:
                        cv2.line(batch_image_raw[i], points[partA], points[partB], (0, 255, 0), 2)

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

    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy_with_keypoints(self, origin_h, origin_w, boxes, keypoints):

        n = len(boxes)
        box_array = np.zeros_like(boxes)
        keypoint_array = np.zeros_like(keypoints)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        for i in range(n):
            if r_h > r_w:
                box = boxes[i]
                lmk = keypoints[i]
                box_array[i, 0] = box[0] / r_w
                box_array[i, 2] = box[2] / r_w
                box_array[i, 1] = (box[1] - (self.input_h - r_w * origin_h) / 2) / r_w
                box_array[i, 3] = (box[3] - (self.input_h - r_w * origin_h) / 2) / r_w

                for j in range(0, len(lmk), 3):
                    keypoint_array[i, j] = lmk[j] / r_w
                    keypoint_array[i, j + 1] = (lmk[j + 1] - (self.input_h - r_w * origin_h) / 2) / r_w
                    keypoint_array[i, j + 2] = lmk[j + 2]
            else:

                box = boxes[i]
                lmk = keypoints[i]

                box_array[i, 0] = (box[0] - (self.input_w - r_h * origin_w) / 2) / r_h
                box_array[i, 2] = (box[2] - (self.input_w - r_h * origin_w) / 2) / r_h
                box_array[i, 1] = box[1] / r_h
                box_array[i, 3] = box[3] / r_h

                for j in range(0, len(lmk), 3):
                    keypoint_array[i, j] = (lmk[j] - (self.input_w - r_h * origin_w) / 2) / r_h
                    keypoint_array[i, j + 1] = lmk[j + 1] / r_h
                    keypoint_array[i, j + 2] = lmk[j + 2]

        return box_array, keypoint_array

    def post_process(self, output, origin_h, origin_w):
        """
        description: Post-process the prediction to include pose keypoints
        param:
            output:     A numpy array like [num_boxes, cx, cy, w, h, conf,
            cls_id, px1, py1, pconf1,...px17, py17, pconf17] where p denotes pose keypoint
            origin_h:   Height of original image
            origin_w:   Width of original image
        return:
            result_boxes:    Final boxes, a numpy array, each row is a box [x1, y1, x2, y2]
            result_scores:   Final scores, a numpy array, each element is the score corresponding to box
            result_classid:  Final classID, a numpy array, each element is the classid corresponding to box
            result_keypoints: Final keypoints, a list of numpy arrays,
            each element represents keypoints for a box, shaped as (#keypoints, 3)
        """
        # Number of values per detection: 38 base values + 17 keypoints * 3 values each + angle
        num_values_per_detection = DET_NUM + SEG_NUM + POSE_NUM + OBB_NUM
        # Get the number of boxes detected
        num = int(output[0])
        # Reshape to a two-dimensional ndarray with the full detection shape
        pred = np.reshape(output[1:], (-1, num_values_per_detection))[:num, :]

        # Perform non-maximum suppression to filter the detections
        boxes = self.non_max_suppression(
            pred[:, :num_values_per_detection], origin_h, origin_w,
            conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)

        # Extract the bounding boxes, confidence scores, and class IDs
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        result_keypoints = boxes[:, -POSE_NUM-1:-1] if len(boxes) else np.array([])

        # Return the post-processed results including keypoints
        return result_boxes, result_scores, result_classid, result_keypoints

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(
            inter_rect_x2 - inter_rect_x1 + 1, 0, None) * np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        res_array = np.copy(boxes)
        box_pred_deep_copy = np.copy(boxes[:, :4])
        keypoints_pred_deep_copy = np.copy(boxes[:, -POSE_NUM-1:-1])
        res_box, res_keypoints = self.xywh2xyxy_with_keypoints(
            origin_h, origin_w, box_pred_deep_copy, keypoints_pred_deep_copy)
        res_array[:, :4] = res_box
        res_array[:, -POSE_NUM-1:-1] = res_keypoints
        # clip the coordinates
        res_array[:, 0] = np.clip(res_array[:, 0], 0, origin_w - 1)
        res_array[:, 2] = np.clip(res_array[:, 2], 0, origin_w - 1)
        res_array[:, 1] = np.clip(res_array[:, 1], 0, origin_h - 1)
        res_array[:, 3] = np.clip(res_array[:, 3], 0, origin_h - 1)
        # Object confidence
        confs = res_array[:, 4]
        # Sort by the confs
        res_array = res_array[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_res_array = []
        while res_array.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(res_array[0, :4], 0), res_array[:, :4]) > nms_thres
            label_match = res_array[0, 5] == res_array[:, 5]
            invalid = large_overlap & label_match
            keep_res_array.append(res_array[0])
            res_array = res_array[~invalid]

        res_array = np.stack(keep_res_array, 0) if len(keep_res_array) else np.array([])
        return res_array


class inferThread(threading.Thread):
    def __init__(self, yolov8_wrapper, image_path_batch):
        threading.Thread.__init__(self)
        self.yolov8_wrapper = yolov8_wrapper
        self.image_path_batch = image_path_batch

    def run(self):
        batch_image_raw, use_time = self.yolov8_wrapper.infer(self.yolov8_wrapper.get_raw_image(self.image_path_batch))
        for i, img_path in enumerate(self.image_path_batch):
            parent, filename = os.path.split(img_path)
            save_name = os.path.join('output', filename)
            # Save image

            cv2.imwrite(save_name, batch_image_raw[i])
        print('input->{}, time->{:.2f}ms, saving into output/'.format(self.image_path_batch, use_time * 1000))


class warmUpThread(threading.Thread):
    def __init__(self, yolov8_wrapper):
        threading.Thread.__init__(self)
        self.yolov8_wrapper = yolov8_wrapper

    def run(self):
        batch_image_raw, use_time = self.yolov8_wrapper.infer(self.yolov8_wrapper.get_raw_image_zeros())
        print('warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))


if __name__ == "__main__":
    # load custom plugin and engine
    PLUGIN_LIBRARY = "./build/libmyplugins.so"
    engine_file_path = "yolov8n-pose.engine"

    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    ctypes.CDLL(PLUGIN_LIBRARY)

    # load coco labels

    categories = ["person"]

    if os.path.exists('output/'):
        shutil.rmtree('output/')
    os.makedirs('output/')
    # a YoLov8TRT instance
    yolov8_wrapper = YoLov8TRT(engine_file_path)
    try:
        print('batch size is', yolov8_wrapper.batch_size)

        image_dir = "images/"
        image_path_batches = get_img_path_batches(yolov8_wrapper.batch_size, image_dir)

        for i in range(10):
            # create a new thread to do warm_up
            thread1 = warmUpThread(yolov8_wrapper)
            thread1.start()
            thread1.join()
        for batch in image_path_batches:
            # create a new thread to do inference
            thread1 = inferThread(yolov8_wrapper, batch)
            thread1.start()
            thread1.join()
    finally:
        # destroy the instance
        yolov8_wrapper.destroy()
