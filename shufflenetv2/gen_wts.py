import struct

import cv2
import numpy as np
import torch
from torchvision.models.shufflenetv2 import (
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)


def read_imagenet_labels() -> dict[int, str]:
    """
    read ImageNet 1000 labels

    Returns:
        dict[int, str]: labels dict
    """
    clsid2label = {}
    with open("../assets/imagenet1000_clsidx_to_labels.txt", "r") as f:
        for i in f.readlines():
            k, v = i.split(": ")
            clsid2label.setdefault(int(k), v[1:-3])
    return clsid2label


def preprocess(img: np.array) -> torch.Tensor:
    """
    a preprocess method align with ImageNet dataset

    Args:
        img (np.array): input image

    Returns:
        torch.Tensor: preprocessed image in `NCHW` layout
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)[None, ...]
    return torch.from_numpy(img)


if __name__ == "__main__":
    labels = read_imagenet_labels()
    img = cv2.imread("../assets/cats.jpg", cv2.IMREAD_COLOR)
    img = preprocess(img)

    """
    NOTE: comment out the model you don't want
    """
    models = [
        ("shufflenet_v2_x0_5", shufflenet_v2_x0_5(pretrained=True)),
        ("shufflenet_v2_x1_0", shufflenet_v2_x1_0(pretrained=True)),
        ("shufflenet_v2_x1_5", shufflenet_v2_x1_5(pretrained=True)),
        ("shufflenet_v2_x2_0", shufflenet_v2_x2_0(pretrained=True)),
    ]

    for name, model in models:
        model.eval()
        with torch.inference_mode():
            output = model(img)
        print(f"{name} result:")
        for i, batch in enumerate(torch.topk(output, k=3).indices):
            for j, idx in enumerate(batch):
                print(f"\tBatch: {i}, Top: {j}, logits: {output[i][idx]:.4f}, label: {labels[int(idx)]}")
        print(f"{'=' * 32}")

        with open(f"../models/{name}.wts", "w") as f:
            f.write("{}\n".format(len(model.state_dict().keys())))
            for k, v in model.state_dict().items():
                print("key: ", k)
                print("value: ", v.shape)
                vr = v.reshape(-1).cpu().numpy()
                f.write("{} {}".format(k, len(vr)))
                for vv in vr:
                    f.write(" ")
                    f.write(struct.pack(">f", float(vv)).hex())
                f.write("\n")
