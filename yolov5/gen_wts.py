import torch
import struct
import sys
import os
from utils.torch_utils import select_device


# Initialize
device = select_device('cpu')
pt_file = sys.argv[1]
# Load model
model = torch.load(pt_file, map_location=device)['model'].float()  # load to FP32
model.to(device).eval()

if len(sys.argv) > 2:
    if os.path.isdir(sys.argv[2]):
        wts_file = os.path.join(
            sys.argv[2],
            os.path.splitext(os.path.basename(pt_file))[0] + '.wts')
    else:
        wts_file = sys.argv[2]
else:
    wts_file = os.path.splitext(pt_file)[0] + '.wts'


with open(wts_file, 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')
