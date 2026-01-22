import struct
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)

        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)

        y = y.view(y.shape[0], -1)

        y = self.fc1(y)
        y = self.relu3(y)

        y = self.fc2(y)
        y = self.relu4(y)

        y = self.fc3(y)
        return y


def reformat_state_dict(state: OrderedDict) -> OrderedDict:
    mapping: dict[str, str] = {
        "layer1.0.weight": "conv1.weight",
        "layer1.0.bias": "conv1.bias",
        "layer1.3.weight": "conv2.weight",
        "layer1.3.bias": "conv2.bias",
        "layer2.0.weight": "fc1.weight",
        "layer2.0.bias": "fc1.bias",
        "layer2.2.weight": "fc2.weight",
        "layer2.2.bias": "fc2.bias",
        "layer2.4.weight": "fc3.weight",
        "layer2.4.bias": "fc3.bias",
    }
    for i, j in mapping.items():
        state.setdefault(j, state.pop(i))
    return state


def main():
    model = LeNet()
    model.eval()
    with torch.inference_mode():
        img = cv2.imread("../assets/6.pgm", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
        img = (((img / 255.0) - 0.1307) / 0.3081).astype(np.float32)
        state = torch.load("../models/model.pt", weights_only=False)
        state = reformat_state_dict(state["state_dict"])
        model.load_state_dict(state)
        input = torch.from_numpy(img)[None, None, ...]
        out = model(input)
        print(f"lenet output shape: {out.shape}")
        print(f"lenet output: {out}")
        print(f"inference result for MNIST data: {int(torch.argmax(out, 1))}")

    # save to wts
    print("Writing into lenet.wts")
    with open("../models/lenet.wts", "w") as f:
        f.write("{}\n".format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write("{} {} ".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")


if __name__ == "__main__":
    main()
