import sys
import argparse
import os
import struct
import torch
from utils.torch_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('pt_file', help='Input (.pt) file (required)')
    parser.add_argument('wts_file', nargs="?", help='Output (.wts) file path (optional)')
    args = parser.parse_args()
    if not os.path.isfile(args.pt_file):
        raise SystemExit('Invalid input file')
    if args.wts_file is None:
        args.wts_file = os.path.splitext(args.pt_file)[0] + '.wts'
    elif os.path.isdir(args.wts_file):
        args.wts_file = os.path.join(
            args.wts_file,
            os.path.splitext(os.path.basename(args.pt_file))[0] + '.wts')
    return args.pt_file, args.wts_file


pt_file, wts_file = parse_args()

# Initialize
device = select_device('cpu')
# Load model
model = torch.load(pt_file, map_location=device)['model'].float()  # load to FP32
model.to(device).eval()

with open(wts_file, 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f' ,float(vv)).hex())
        f.write('\n')
