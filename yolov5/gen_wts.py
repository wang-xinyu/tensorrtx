import sys
import argparse
import os
import struct
import torch
from utils.torch_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-w', '--weights', required=True,
                        help='Input weights (.pt) file path (required)')
    parser.add_argument(
        '-o', '--output', help='Output (.wts) file path (optional)')
    parser.add_argument(
        '-t', '--type', type=str, default='detect', choices=['detect', 'cls', 'seg'],
        help='determines the model is detection/classification')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input file')
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.wts'
    elif os.path.isdir(args.output):
        args.output = os.path.join(
            args.output,
            os.path.splitext(os.path.basename(args.weights))[0] + '.wts')
    return args.weights, args.output, args.type


pt_file, wts_file, m_type = parse_args()
print(f'Generating .wts for {m_type} model')

# Initialize
device = select_device('cpu')
# Load model
print(f'Loading {pt_file}')
model = torch.load(pt_file, map_location=device)  # load to FP32
model = model['ema' if model.get('ema') else 'model'].float()

if m_type in ['detect', 'seg']:
    # update anchor_grid info
    anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]
    # model.model[-1].anchor_grid = anchor_grid
    delattr(model.model[-1], 'anchor_grid')  # model.model[-1] is detect layer
    # The parameters are saved in the OrderDict through the "register_buffer" method, and then saved to the weight.
    model.model[-1].register_buffer("anchor_grid", anchor_grid)
    model.model[-1].register_buffer("strides", model.model[-1].stride)

model.to(device).eval()

print(f'Writing into {wts_file}')
with open(wts_file, 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')
