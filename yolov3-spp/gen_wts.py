import struct
import sys
import os
import torch
from models import Darknet
from utils.utils import torch_utils


model = Darknet('cfg/yolov3-spp.cfg', (416, 416))
weights_file = sys.argv[1]
dev = '0'
device = torch_utils.select_device(dev)
model.load_state_dict(torch.load(weights_file, map_location=device)['model'])

if len(sys.argv) > 2:
    if os.path.isdir(sys.argv[2]):
        wts_file = os.path.join(
            sys.argv[2],
            os.path.splitext(os.path.basename(weights_file))[0] + '.wts')
    else:
        wts_file = sys.argv[2]
else:
    wts_file = os.path.splitext(weights_file)[0] + '.wts'


with open(wts_file, 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')

