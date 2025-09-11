import argparse
import os

import cv2
import numpy as np
import nvtripy as tp
import time
from constants import IMAGE_H, IMAGE_W

CURDIR = os.path.realpath(os.path.dirname(__file__))


def load_image(path):
    return cv2.imread(path)


def preprocess(image):
    h, w, _ = image.shape
    # Crop the center square frame
    m = min(h, w)
    top = (h - m) // 2
    left = (w - m) // 2
    image = image[top:top + m, left:left + m]

    # Resize the image with target size while maintaining ratio
    image = cv2.resize(image, (IMAGE_H, IMAGE_W), interpolation=cv2.INTER_LINEAR)

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

    return image


def main():
    parser = argparse.ArgumentParser(description="Classify an image using a YOLO11 classifier model.")
    parser.add_argument("images", help="Images to classify", nargs="+")
    parser.add_argument(
        "--model-path",
        help="Path to the compiled model",
        default=os.path.join(CURDIR, "yolo11-cls.tpymodel"),
    )

    parser.add_argument(
        "--imagenet-classes-file",
        help="Path to the ImageNet classes file (imagenet_classes.txt)",
        default=os.path.join(CURDIR, "imagenet_classes.txt"),
    )

    args, _ = parser.parse_known_args()

    with open(args.imagenet_classes_file) as f:
        CLASSES = [line.strip() for line in f.readlines()]

    print(f"Loading model: {args.model_path}...")

    model = tp.Executable.load(args.model_path)

    input_info = model.input_infos["batch"]
    dtype = input_info.dtype

    if input_info.shape_bounds.max[0] < len(args.images):
        raise ValueError(
            f"Model was compiled for a maximum of {input_info.shape_bounds.max[0]} image(s) "
            f"per batch, but {len(args.images)} were provided."
            f"\nPlease recompile the model with a larger maximum batch size using the "
            f"`--max-images` argument in `compile_classifier.py`."
        )

    images = [preprocess(load_image(path)) for path in args.images]
    batch = tp.Tensor(np.concatenate(images, axis=0))

    # Warm up the model:
    _, _ = model(tp.zeros_like(batch, dtype=dtype).eval())

    # Cast the input based on the model type.
    # Note that the result will be in GPU memory, so we don't need an explicit copy.
    batch = tp.cast(batch, dtype).eval()

    start = time.perf_counter()
    batch_scores, batch_preds = model(batch)
    end = time.perf_counter()

    print(f"Inference + Postprocessing took: {(end - start) * 1000:.3f} ms")

    # Copy the scores back to CPU memory and convert to numpy:
    batch_scores = np.from_dlpack(tp.copy(batch_scores, device=tp.device("cpu")))
    batch_preds = np.from_dlpack(tp.copy(batch_preds, device=tp.device("cpu")))

    for path, scores, preds in zip(args.images, batch_scores, batch_preds):
        print(f"Top {len(preds)} predictions for:", path)
        for idx, (score, pred) in enumerate(zip(scores, preds)):
            print(f"    {idx + 1}. (confidence: {score:.3f}) {CLASSES[pred]}")
        print()


if __name__ == "__main__":
    main()
