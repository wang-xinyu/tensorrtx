import struct

import cv2
import numpy as np
import torch
from torchvision.models import alexnet


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
    img = cv2.imread("../assets/cats.jpg", cv2.IMREAD_COLOR)
    img = preprocess(img)
    model = alexnet(pretrained=True)
    model.eval()
    output = model(img)
    labels = read_imagenet_labels()
    for batch in torch.topk(output, k=3).indices:
        for i, j in enumerate(batch, 1):
            print(f"top: {i:<2}, confidence: {float(output[0, j]):.4f}, label: {labels[int(j)]}")

    print("writing alexnet wts")
    with open("../models/alexnet.wts", "w") as f:
        f.write("{}\n".format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            print(f"key: {k}\tvalue: {v.shape}")
            vr = v.reshape(-1).cpu().numpy()
            f.write("{} {}".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")
