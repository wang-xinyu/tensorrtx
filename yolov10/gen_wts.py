# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2024/7/22 下午9:17
  @version V1.0
"""
import sys  # noqa: F401
import argparse
import os
import struct
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-w', '--weights', default='./weights/yolov10n.pt',
                        help='Input weights (.pt) file path (required)')
    parser.add_argument(
        '-o', '--output', help='Output (.wts) file path (optional)')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input file')
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.wts'
    elif os.path.isdir(args.output):
        args.output = os.path.join(
            args.output,
            os.path.splitext(os.path.basename(args.weights))[0] + '.wts')
    return args.weights, args.output


pt_file, wts_file = parse_args()

# Load model
print(f'Loading {pt_file}')

# Initialize
device = 'cpu'

# Load model
model = torch.load(pt_file, map_location=device)['model'].float()  # load to FP32
# If the training is not finished, the model will be interrupted.
# model = torch.load(pt_file, map_location=device)['ema'].float()  # load to FP32

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
print(f'success {wts_file}!!!')
