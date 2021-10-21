import sys
import argparse
import os
import struct
import torch
from utils.torch_utils import select_device
import numpy as np

anchor_grid={
    'model.24.anchor_grid':[
        [10,13, 16,30, 33,23],      # P3/8
        [30,61, 62,45, 59,119],     # P4/16
        [116,90, 156,198, 373,326]  # P5/32
    ],
    'model.33.anchor_grid':[
        [19,27,  44,40,  38,94],       # P3/8
        [96,68,  86,152,  180,137],    # P4/16
        [140,301,  303,264,  238,542], # P5/32
        [436,615,  739,380,  925,792]  # P6/64
    ]
}

def verify_model(model):
    '''
    Verify that the anchor_grid layer exists
    '''
    module_key=None
    model_keys=model.state_dict().keys()
    if not("model.24.anchor_grid" in  model_keys  or "model.33.anchor_grid" in  model_keys):
        if "model.24.anchors" in  model_keys:
            module_key="model.24.anchor_grid"
        elif "model.33.anchors" in  model_keys:
            module_key="model.33.anchor_grid"
        else:
            raise NotImplementedError
    return  module_key


def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-o', '--output', help='Output (.wts) file path (optional)')
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

# Initialize
device = select_device('cpu')
# Load model
model = torch.load(pt_file, map_location=device)['model'].float()  # load to FP32
model.to(device).eval()
module_key=verify_model(model)

with open(wts_file, 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys()) if not module_key else  len(model.state_dict().keys())+1))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f' ,float(vv)).hex())
        f.write('\n')

        # insert anchor_grid
        if module_key and 'anchors' in  k:
            k=module_key
            vr=anchor_grid[k]
            vr=np.array(vr).flatten()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f' ,float(vv)).hex())
            f.write('\n')
