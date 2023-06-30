import sys
import argparse
import os
import struct
import torch

pt_file = "./weights/yolov8s.pt"
wts_file = "./weights/yolov8s.wts"

# Initialize
device = 'cpu'

# Load model
model = torch.load(pt_file, map_location=device)['model'].float()  # load to FP32

anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]

delattr(model.model[-1], 'anchors')

model.to(device).eval()

with open(wts_file, 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')
