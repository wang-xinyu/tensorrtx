import struct

import cv2
import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification


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


USE_HF_PREPROCESS = False

if __name__ == "__main__":
    hub_model_id = "google/vit-base-patch16-224"
    config = AutoConfig.from_pretrained(hub_model_id)
    config._attn_implementation = "eager"
    model = AutoModelForImageClassification.from_pretrained(
        hub_model_id,
        ignore_mismatched_sizes=False,
        config=config,
    )

    model.eval()

    img = cv2.imread("../assets/cats.jpg", cv2.IMREAD_COLOR)

    if USE_HF_PREPROCESS:
        image_processor = AutoImageProcessor.from_pretrained(hub_model_id)
        img = image_processor(img, return_tensors="pt")
        img = img["pixel_values"]
    else:
        img: np.array = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 255.0 - np.array([0.5, 0.5, 0.5])) / np.array([0.5, 0.5, 0.5])
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))[None, ...])

    output = model(img)
    labels = read_imagenet_labels()
    for i, j in enumerate(torch.topk(output.logits[0], k=3).indices):
        print(f"Top: {i} is {labels[int(j)]}")

    f = open("../models/vit.wts", "w")
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
