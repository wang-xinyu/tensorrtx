import struct
import torch
import sys
from models import *  # noqa: F403
from utils.utils import *  # noqa: F403

model = Darknet('cfg/yolov3-tiny.cfg', (608, 608))  # noqa: F405
weights = sys.argv[1]
device = torch_utils.select_device('0')  # noqa: F405
if weights.endswith('.pt'):  # pytorch format
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=False)['model'])
else:  # darknet format
    load_darknet_weights(model, weights)  # noqa: F405
model = model.eval()

with open('yolov3-tiny.wts', 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')
