import struct
import sys
import os
import torch
from models import Darknet, load_darknet_weights
from utils import torch_utils


model = Darknet('cfg/yolov3.cfg', (608, 608))
weights_file = sys.argv[1]
device = torch_utils.select_device('0')
if weights_file.endswith('.pt'):  # pytorch format
    model.load_state_dict(torch.load(weights_file, map_location=device)['model'])
else:  # darknet format
    load_darknet_weights(model, weights_file)
model = model.eval()

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

