"""
model codes are borrowed from:
`https://github.com/xuexingyu24/License_Plate_Detection_Pytorch/blob/master/LPRNet/model/LPRNET.py`

check `.pth` model here:
`https://github.com/xuexingyu24/License_Plate_Detection_Pytorch/blob/master/LPRNet/weights/Final_LPRNet_model.pth`

"""

import struct

import cv2
import numpy as np
import torch
import torch.nn as nn

CHARS = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZIO-"


def preprocess(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (94, 24), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    image = image / 255.0 - 0.5  # still HxWx3, BGR
    image = image.transpose(2, 0, 1)[None, ...]
    image = torch.from_numpy(image)
    return image


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),  # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),  # 4
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),  # 11
            nn.BatchNorm2d(num_features=256),  # 12
            nn.ReLU(),  # 13
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),  # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # 22
        )
        self.container = nn.Sequential(
            nn.Conv2d(
                in_channels=256 + class_num + 128 + 64, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)
            )
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # [2, 4, 8, 11, 22]
                print(self.backbone[i])
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits


if __name__ == "__main__":
    model_path = "../models/Final_LPRNet_model.pth"

    model = LPRNet(class_num=len(CHARS), dropout_rate=0)
    print("loading pretrained model from %s" % model_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))

    img = preprocess("../assets/car_plate.jpg")
    model.eval()
    print(model)
    with torch.inference_mode():
        preds = model(img)
        res = "".join(CHARS[i] for i in torch.argmax(preds[0], dim=0).tolist())
        res = res.replace("-", "")

    with open("../models/LPRNet.wts", "w") as f:
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

    print(f"inference result: {res}")
