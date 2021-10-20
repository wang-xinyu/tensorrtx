import sys
import argparse
import numpy as np
import struct
import torch
from pathlib import Path
from utils.torch_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-of', '--official', required=True, help='Input weights(.pt) file path of official pretrained model (required)')
    parser.add_argument('-o', '--output', help='Output (.wts) file path (optional)')
    args = parser.parse_args()
    weights_path = Path(args.weights)
    official_path = Path(args.official)
    if not (weights_path.is_file() and official_path.is_file()):
        raise SystemExit('Invalid input file(s)')
    if not args.output:
        args.output = weights_path.stem + '.wts'
    else:
        output_path = Path(args.output)
        if output_path.is_dir():
            output_path = output_path / weights_path.stem + '.wts'
    return weights_path, official_path, output_path


pt_file, official_pt_file, wts_file = parse_args()

# Initialize
device = select_device('cpu')
# Load model
official_model = torch.load(str(official_pt_file), map_location=device)['model'].float()
model = torch.load(str(pt_file), map_location=device)['model'].float()  # load to FP32
model.to(device).eval()

with wts_file.open(mode = w+) as f:
    f.write('{}\n'.format(len(official_model.state_dict().keys())))
    for k, v in official_model.state_dict().items():
        if k in model.state_dict():
            v = model.state_dict()[k]
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f' ,float(vv)).hex())
        f.write('\n')
