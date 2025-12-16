import struct
import sys
import torch
from models import *  # noqa: F403
from utils.utils import *  # noqa: F403

model = Darknet('cfg/yolov3-spp.cfg', (416, 416))  # noqa: F405
weights = sys.argv[1]
dev = '0'
device = torch_utils.select_device(dev)  # noqa: F405
model.load_state_dict(torch.load(weights, map_location=device, weights_only=False)['model'])


with open('yolov3-spp_ultralytics68.wts', 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')
