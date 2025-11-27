import cv2
import random
import numpy as np

COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]


def preprocess(img, input_size):
    """
    Preprocess input image for YOLOv1:
        1. BGR -> RGB
        2. Resize
        3. HWC -> CHW
        4. Normalize to 0~1
        5. Add batch dimension
        6. Make contiguous
    """
    # 1. BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Resize
    img = cv2.resize(img, (input_size, input_size))

    # 3. HWC -> CHW
    img = img.transpose(2, 0, 1)

    # 4. float32 + normalize 0~1
    img = img.astype(np.float32) / 255.0

    # 5. Add batch
    img = np.expand_dims(img, axis=0)

    # 6. Contiguous
    return np.ascontiguousarray(img)


def pred2xywhcc(pred, S, B, num_classes, conf_thresh, iou_thresh):
    """
    Convert YOLOv1 grid output (S, S, 30) to a list of bounding boxes.

    Input:
        pred: raw YOLOv1 prediction, shape (S, S, 30)
        Format per grid:
            [B1_x, B1_y, B1_w, B1_h, B1_conf,
             B2_x, B2_y, B2_w, B2_h, B2_conf,
             class_1, ..., class_20]

    Steps:
        - For every grid cell, keep the bbox (B1 or B2) with higher confidence.
        - Keep class probabilities unchanged.
        - Apply NMS over all selected bounding boxes.

    Output:
        array of [x, y, w, h, score, class_id]
    """

    # pred : shape = (S, S, 30)
    pred = np.array(pred, dtype=np.float32)

    # Final bboxes after selecting one bbox per grid cell
    bboxes = np.zeros((S * S, 5 + num_classes), dtype=np.float32)

    for x in range(S):
        for y in range(S):

            conf1, conf2 = pred[x, y, 4], pred[x, y, 9]

            if conf1 > conf2:
                # B1: pred[x,y,0:4] = [x,y,w,h]
                bboxes[x * S + y, 0:4] = pred[x, y, 0:4]
                bboxes[x * S + y, 4] = conf1
                bboxes[x * S + y, 5:] = pred[x, y, 10:]
            else:
                # B2: pred[x,y,5:9]
                bboxes[x * S + y, 0:4] = pred[x, y, 5:9]
                bboxes[x * S + y, 4] = conf2
                bboxes[x * S + y, 5:] = pred[x, y, 10:]

    xywhcc = nms(bboxes, num_classes, conf_thresh, iou_thresh)
    return xywhcc


def nms(bboxs, num_classes, conf_thresh=0.1, iou_thresh=0.3):
    """
    Non-Maximum Suppression (class-specific) for YOLOv1 results.

    Input:
        bboxs: (N, 5 + num_classes)
               [x, y, w, h, conf, class_scores...]

        conf: YOLOv1 object confidence
        class_scores: conditional class probabilities P(class | object)

        final score = conf * class_prob

    Output: (M, 6)
        [x, y, w, h, max_score, class_id]
    """

    if len(bboxs) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    bboxs = np.array(bboxs, dtype=np.float32)

    # Step 1: class-specific confidence = conf * P(class | object)
    bbox_prob = bboxs[:, 5:]  # N x num_classes
    bbox_conf = bboxs[:, 4:5]  # N x 1
    bbox_cls_spec_conf = bbox_conf * bbox_prob

    bbox_cls_spec_conf[bbox_cls_spec_conf <= conf_thresh] = 0

    # Step 2: NMS per class
    for c in range(num_classes):
        scores = bbox_cls_spec_conf[:, c].copy()
        order = scores.argsort()[::-1]

        for i in range(len(order)):
            if scores[order[i]] == 0:
                continue

            for j in range(i + 1, len(order)):
                if scores[order[j]] == 0:
                    continue
                iou = calculate_iou(bboxs[order[i], 0:4], bboxs[order[j], 0:4])
                if iou > iou_thresh:
                    scores[order[j]] = 0

            bbox_cls_spec_conf[:, c] = scores

    # Step 3: keep boxes that have at least one class score > 0
    keep_idx = np.max(bbox_cls_spec_conf, axis=1) > 0
    if np.sum(keep_idx) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    bboxs_keep = bboxs[keep_idx]
    cls_conf_keep = bbox_cls_spec_conf[keep_idx]

    # Step 4: Format output
    ret = np.ones((bboxs_keep.shape[0], 6), dtype=np.float32)
    ret[:, 0:4] = bboxs_keep[:, 0:4]
    ret[:, 4] = np.max(cls_conf_keep, axis=1)
    ret[:, 5] = np.argmax(bboxs_keep[:, 5:], axis=1)

    return ret


def calculate_iou(bbox1, bbox2):
    """
    Compute IoU for two boxes given as:
        [x_center, y_center, w, h]
    """

    bbox1 = np.array(bbox1, dtype=np.float32)
    bbox2 = np.array(bbox2, dtype=np.float32)

    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]

    left1 = bbox1[0] - bbox1[2] / 2
    right1 = bbox1[0] + bbox1[2] / 2
    top1 = bbox1[1] - bbox1[3] / 2
    bottom1 = bbox1[1] + bbox1[3] / 2

    left2 = bbox2[0] - bbox2[2] / 2
    right2 = bbox2[0] + bbox2[2] / 2
    top2 = bbox2[1] - bbox2[3] / 2
    bottom2 = bbox2[1] + bbox2[3] / 2

    max_left = max(left1, left2)
    min_right = min(right1, right2)
    max_top = max(top1, top2)
    min_bottom = min(bottom1, bottom2)

    if max_left >= min_right or max_top >= min_bottom:
        return 0.0

    intersect = (min_right - max_left) * (min_bottom - max_top)
    iou = intersect / (area1 + area2 - intersect)
    return iou


def draw_bbox(img, bboxs, class_names):
    h, w = img.shape[0:2]
    n = bboxs.shape[0]
    for i in range(n):
        p1 = (int((bboxs[i, 0] - bboxs[i, 2] / 2) * w), int((bboxs[i, 1] - bboxs[i, 3] / 2) * h))
        p2 = (int((bboxs[i, 0] + bboxs[i, 2] / 2) * w), int((bboxs[i, 1] + bboxs[i, 3] / 2) * h))
        class_name = class_names[int(bboxs[i, 5])]
        # confidence = bboxs[i, 4]
        cv2.rectangle(img, p1, p2, color=COLORS[int(bboxs[i, 5])], thickness=2)
        cv2.putText(img, class_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[int(bboxs[i, 5])])
    return img
