#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa: E501
# pylint: disable=no-member

"""
Utility functions for YOLOv1 inference.

This module includes:
    - Image preprocessing
    - YOLOv1 prediction decoding
    - Non-maximum suppression (NMS)
    - IoU calculation
    - Bounding box drawing utilities
"""

import random
import cv2
import numpy as np

COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]


def preprocess(img, input_size):
    """
    Preprocess an input image for YOLOv1 model.

    Steps:
        1. Convert BGR → RGB
        2. Resize to (input_size, input_size)
        3. Convert from HWC to CHW
        4. Normalize pixel values to [0, 1]
        5. Add batch dimension
        6. Return contiguous array

    Args:
        img (np.ndarray): BGR source image.
        input_size (int): Model input resolution.

    Returns:
        np.ndarray: Preprocessed image ready for model inference.
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


def pred2xywhcc(pred, s, num_classes, conf_thresh, iou_thresh):
    """
    Convert YOLOv1 (S, S, 30) grid predictions into bounding boxes.

    Select one bbox (B1 or B2) per grid cell, compute confidence,
    and apply NMS across all boxes.

    Args:
        pred (np.ndarray): YOLOv1 raw model output.
        s (int): Grid size (e.g., 7).
        num_classes (int): Number of classes.
        conf_thresh (float): Confidence threshold.
        iou_thresh (float): NMS IoU threshold.

    Returns:
        np.ndarray: Array of boxes [x, y, w, h, score, class_id].
    """

    # pred : shape = (s, s, 30)
    pred = np.array(pred, dtype=np.float32)

    # Final bboxes after selecting one bbox per grid cell
    bboxes = np.zeros((s * s, 5 + num_classes), dtype=np.float32)

    for x in range(s):
        for y in range(s):

            conf1, conf2 = pred[x, y, 4], pred[x, y, 9]

            if conf1 > conf2:
                # B1: pred[x,y,0:4] = [x,y,w,h]
                bboxes[x * s + y, 0:4] = pred[x, y, 0:4]
                bboxes[x * s + y, 4] = conf1
                bboxes[x * s + y, 5:] = pred[x, y, 10:]
            else:
                # B2: pred[x,y,5:9]
                bboxes[x * s + y, 0:4] = pred[x, y, 5:9]
                bboxes[x * s + y, 4] = conf2
                bboxes[x * s + y, 5:] = pred[x, y, 10:]

    xywhcc = nms(bboxes, num_classes, conf_thresh, iou_thresh)
    return xywhcc


def nms(bboxs, num_classes, conf_thresh=0.1, iou_thresh=0.3):
    """
    Perform class-specific Non-Maximum Suppression (NMS).

    Computes:
        - class-specific confidence = conf * class_prob
        - threshold scores
        - per-class NMS

    Args:
        bboxs (np.ndarray): Boxes of shape (N, 5 + num_classes).
        num_classes (int): Number of classes.
        conf_thresh (float): Confidence threshold.
        iou_thresh (float): IoU threshold for NMS.

    Returns:
        np.ndarray: Suppressed boxes, shape (M, 6).
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

        for i_idx, i in enumerate(order):
            if scores[i] == 0:
                continue

            for _, j in enumerate(order[i_idx + 1 :], start=i_idx + 1):
                if scores[j] == 0:
                    continue

                iou = calculate_iou(bboxs[i, 0:4], bboxs[j, 0:4])
                if iou > iou_thresh:
                    scores[j] = 0

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
    Compute IoU between two boxes.

    Args:
        bbox1 (array-like): [x_center, y_center, w, h]
        bbox2 (array-like): [x_center, y_center, w, h]

    Returns:
        float: Intersection-over-Union score.
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
    """
    Draw bounding boxes on an image.

    Args:
        img (np.ndarray): Target image.
        bboxs (np.ndarray): Array of [x, y, w, h, score, class_id].
        class_names (list[str]): Class name list.

    Returns:
        np.ndarray: Image with drawn bounding boxes.
    """
    h, w = img.shape[0:2]
    n = bboxs.shape[0]
    for i in range(n):
        p1 = (
            int((bboxs[i, 0] - bboxs[i, 2] / 2) * w),
            int((bboxs[i, 1] - bboxs[i, 3] / 2) * h),
        )
        p2 = (
            int((bboxs[i, 0] + bboxs[i, 2] / 2) * w),
            int((bboxs[i, 1] + bboxs[i, 3] / 2) * h),
        )
        class_name = class_names[int(bboxs[i, 5])]
        # confidence = bboxs[i, 4]
        cv2.rectangle(img, p1, p2, color=COLORS[int(bboxs[i, 5])], thickness=2)
        cv2.putText(
            img, class_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[int(bboxs[i, 5])]
        )
    return img
