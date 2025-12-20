"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import shutil
import sys
import threading
import time
import cv2
import math
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

INPUT_W = 1024
INPUT_H = 1024


class Detection:
    def __init__(self, bbox, score, class_id, angle):
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.angle = angle


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


def get_corner(img, box: Detection):
    """
    description: Get the four corner points of the rotated bounding box
    param:
        img:    an opencv image object (numpy array)
        box:    a Detection object containing bbox [cx,cy,w,h] and angle (radians)
    return:
        corners: four corner points of the rotated bounding box as numpy array [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    # Extract box parameters
    cx, cy, w, h = box.bbox
    angle = box.angle * 180.0 / math.pi  # Convert radians to degrees

    # Swap width and height if height >= width
    if h >= w:
        w, h = h, w
        angle = (angle + 90.0) % 180.0  # Adjust angle

    # Ensure angle is between 0 and 180 degrees
    if angle < 0:
        angle += 360.0
    if angle > 180.0:
        angle -= 180.0

    # Convert to normalized angle (0-180)
    normal_angle = angle % 180.0
    if normal_angle < 0:
        normal_angle += 180.0

    # Convert back to radians for calculation
    angle_rad = angle * math.pi / 180.0
    cos_val = math.cos(angle_rad)
    sin_val = math.sin(angle_rad)

    # Calculate boundaries
    l_x = cx - w / 2
    r_x = cx + w / 2
    t_y = cy - h / 2
    b_y = cy + h / 2

    # Scale coordinates using get_rect_obb (matching C++ version)
    bbox = [l_x, t_y, r_x, b_y]
    rect = get_rect_obb(img, bbox)

    # Calculate center and dimensions of scaled box
    x_ = (rect[0] + rect[0] + rect[2]) / 2  # rect.x + rect.width/2
    y_ = (rect[1] + rect[1] + rect[3]) / 2  # rect.y + rect.height/2
    width = rect[2]
    height = rect[3]

    # Calculate vectors
    vec1x = width / 2 * cos_val
    vec1y = width / 2 * sin_val
    vec2x = -height / 2 * sin_val
    vec2y = height / 2 * cos_val

    # Calculate four corners
    corners = np.array([
        [int(round(x_ + vec1x + vec2x)), int(round(y_ + vec1y + vec2y))],  # Top-left
        [int(round(x_ + vec1x - vec2x)), int(round(y_ + vec1y - vec2y))],  # Top-right
        [int(round(x_ - vec1x - vec2x)), int(round(y_ - vec1y - vec2y))],  # Bottom-right
        [int(round(x_ - vec1x + vec2x)), int(round(y_ - vec1y + vec2y))]  # Bottom-left
    ], dtype=np.int32)

    # Clip to image boundaries
    h, w = img.shape[:2]
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)

    return corners


def get_rect_obb(img, bbox):
    """
    Scale coordinates according to image resize ratio (matching C++ version)
    param:
        img: OpenCV image (numpy array)
        bbox: [left, top, right, bottom]
    return:
        [x, y, width, height]
    """
    l_x, t_y, r_x, b_y = bbox
    r_w = INPUT_W / img.shape[1]  # INPUT_W should be your model input width
    r_h = INPUT_H / img.shape[0]  # INPUT_H should be your model input height

    if r_h > r_w:
        l_x = l_x
        r_x = r_x
        t_y = t_y - (INPUT_H - r_w * img.shape[0]) / 2
        b_y = b_y - (INPUT_H - r_w * img.shape[0]) / 2
        l_x = l_x / r_w
        r_x = r_x / r_w
        t_y = t_y / r_w
        b_y = b_y / r_w
    else:
        l_x = l_x - (INPUT_W - r_h * img.shape[1]) / 2
        r_x = r_x - (INPUT_W - r_h * img.shape[1]) / 2
        t_y = t_y
        b_y = b_y
        l_x = l_x / r_h
        r_x = r_x / r_h
        t_y = t_y / r_h
        b_y = b_y / r_h

    l_x = max(0.0, l_x)
    t_y = max(0.0, t_y)
    width = max(0, min(int(round(r_x - l_x)), img.shape[1] - int(round(l_x))))
    height = max(0, min(int(round(b_y - t_y)), img.shape[0] - int(round(t_y))))

    return [int(round(l_x)), int(round(t_y)), width, height]


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one rotated bounding box on image img
    param:
        x:      a box in [cx, cy, w, h, angle] format
        img:    an opencv image object
        color:  color to draw rectangle
        label:  str
        line_thickness: int
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1

    # Get four corner points
    corners = get_corner(img, x)
    corners = corners.astype(int)

    # Draw the rotated rectangle
    cv2.polylines(img, [corners], isClosed=True, color=color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        # Use first corner point for label placement
        p1 = tuple(corners[0])
        w, h = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        outside = p1[1] - h >= 3
        p2 = (p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3)

        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA
        )


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
        input_binding_names = []
        output_binding_names = []

        for binding_name in engine:
            shape = engine.get_tensor_shape(binding_name)
            print('binding_name:', binding_name, shape)
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            # Append to the appropriate list.
            if engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
                input_binding_names.append(binding_name)
                global INPUT_W, INPUT_H
                self.input_w = shape[-1]
                INPUT_W = self.input_w
                self.input_h = shape[-2]
                INPUT_H = self.input_h
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            elif engine.get_tensor_mode(binding_name) == trt.TensorIOMode.OUTPUT:
                output_binding_names.append(binding_name)
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
            else:
                print('unknow:', binding_name)

        # Store
        self.stream = stream
        self.context = context
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.input_binding_names = input_binding_names
        self.output_binding_names = output_binding_names
        self.batch_size = engine.get_tensor_shape(input_binding_names[0])[0]
        self.det_output_length = host_outputs[0].shape[0]
        print('batch_size:', self.batch_size)

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
        input_binding_names = self.input_binding_names
        output_binding_names = self.output_binding_names
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
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.set_tensor_address(input_binding_names[0], cuda_inputs[0])
        context.set_tensor_address(output_binding_names[0], cuda_outputs[0])
        context.execute_async_v3(stream_handle=stream.handle)
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
            keep = self.post_process(
                output[i * self.det_output_length: (i + 1) * self.det_output_length], batch_origin_h[i],
                batch_origin_w[i]
            )
            # Draw rectangles and labels on the original image
            for j in range(len(keep)):
                box = keep[j]  # type: Detection
                np.random.seed(int(keep[j].class_id))
                color = [np.random.randint(0, 255) for _ in range(3)]
                plot_one_box(
                    box,
                    batch_image_raw[i],
                    label="{}:{:.2f}".format(
                        categories[int(keep[j].class_id)], keep[j].score
                    ),
                    color=color,
                    line_thickness=1
                )
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

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0]
            y[:, 2] = x[:, 2]
            y[:, 1] = x[:, 1] - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 3] - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 2] - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1]
            y[:, 3] = x[:, 3]
            y /= r_h

        return y

    def covariance_matrix(self, res: Detection):
        """
        description: Generating covariance matrix from obbs.
        param:
            box (np.ndarray): A numpy array representing rotated bounding box, with xywhr format.

        return:
            tuple: (a, b, c) values of covariance matrix
        """
        w = res.bbox[2]
        h = res.bbox[3]
        angle = res.angle

        a = w * w / 12.0
        b = h * h / 12.0
        c = angle

        cos_r = math.cos(c)
        sin_r = math.sin(c)

        cos_r2 = cos_r * cos_r
        sin_r2 = sin_r * sin_r

        a_val = a * cos_r2 + b * sin_r2
        b_val = a * sin_r2 + b * cos_r2
        c_val = (a - b) * cos_r * sin_r

        return a_val, b_val, c_val

    def probiou(self, box1: Detection, box2: Detection, eps=1e-7):
        """
        description: Calculate the prob IoU between oriented bounding boxes.
        param:
            box1 (np.ndarray): First box in xywhr format
            box2 (np.ndarray): Second box in xywhr format
            eps (float): Small value to avoid division by zero
        return:
            float: 1 - hd where hd is the Bhattacharyya distance
        """
        a1, b1, c1 = self.covariance_matrix(box1)
        a2, b2, c2 = self.covariance_matrix(box2)

        x1, y1 = box1.bbox[0], box1.bbox[1]
        x2, y2 = box2.bbox[0], box2.bbox[1]

        t1 = ((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / \
             ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)
        t1 *= 0.25

        t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / \
             ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)
        t2 *= 0.5

        t3 = ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2) / \
             (4 * math.sqrt(max(a1 * b1 - c1 * c1, 0.0)) *
              math.sqrt(max(a2 * b2 - c2 * c2, 0.0)) + eps)
        t3 = math.log(t3 + eps) * 0.5

        bd = max(min(t1 + t2 + t3, 100.0), eps)
        hd = math.sqrt(1.0 - math.exp(-bd) + eps)

        return 1 - hd

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id,angle cx,cy,w,h,conf,cls_id,angle ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2, angle]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        num_values_per_detection = DET_NUM + SEG_NUM + POSE_NUM + OBB_NUM
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, num_values_per_detection))[:num, :]

        # Filter by confidence threshold
        mask = pred[:, 4] >= CONF_THRESH
        pred = pred[mask]

        if len(pred) == 0:
            return []

        m_map = {}
        for i in range(len(pred)):
            class_id = int(pred[i][5])
            if class_id not in m_map:
                m_map[class_id] = []
            m_map[class_id].append(Detection(pred[i][:4], pred[i][4], class_id, pred[i][89]))

        res = []
        for it in m_map:
            dets = m_map[it]
            dets = sorted(dets, key=lambda x: x.score, reverse=True)
            for m in range(len(dets)):
                if dets[m].score == 0.0:
                    continue
                item = dets[m]
                res.append(item)
                for n in range(m + 1, len(dets)):
                    if dets[n].score == 0.0:
                        continue
                    if self.probiou(item, dets[n]) > IOU_THRESHOLD:
                        dets[n].score = 0.0

        keep = []
        for i in range(len(res)):
            if res[i].score > CONF_THRESH:
                keep.append(res[i])

        return keep


class inferThread(threading.Thread):
    def __init__(self, yolo11_wrapper, image_path_batch):
        threading.Thread.__init__(self)
        self.yolo11_wrapper = yolo11_wrapper
        self.image_path_batch = image_path_batch

    def run(self):
        batch_image_raw, use_time = self.yolo11_wrapper.infer(self.yolo11_wrapper.get_raw_image(self.image_path_batch))
        for i, img_path in enumerate(self.image_path_batch):
            parent, filename = os.path.split(img_path)
            save_name = os.path.join('output', filename)
            # Save image
            cv2.imwrite(save_name, batch_image_raw[i])
        print('input->{}, time->{:.2f}ms, saving into output/'.format(self.image_path_batch, use_time * 1000))


class warmUpThread(threading.Thread):
    def __init__(self, yolo11_wrapper):
        threading.Thread.__init__(self)
        self.yolo11_wrapper = yolo11_wrapper

    def run(self):
        batch_image_raw, use_time = self.yolo11_wrapper.infer(self.yolo11_wrapper.get_raw_image_zeros())
        print('warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))


if __name__ == "__main__":
    # load custom plugin and engine
    PLUGIN_LIBRARY = "./build/libmyplugins.so"
    engine_file_path = "yolo11n-obb.engine"

    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    ctypes.CDLL(PLUGIN_LIBRARY)

    # load DOTAV 1.5 labels

    categories = ["plane", "ship", "storage tank", "baseball diamond", "tennis court",
                  "basketball court", "ground track field", "harbor",
                  "bridge", "large vehicle", "small vehicle", "helicopter",
                  "roundabout", "soccer ball field", "swimming pool", "container crane"]

    if os.path.exists('output/'):
        shutil.rmtree('output/')
    os.makedirs('output/')
    # a YoLo11TRT instance
    yolo11_wrapper = YoLo11TRT(engine_file_path)
    try:
        print('batch size is', yolo11_wrapper.batch_size)

        image_dir = "images/"
        image_path_batches = get_img_path_batches(yolo11_wrapper.batch_size, image_dir)

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
