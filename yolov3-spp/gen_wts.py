import struct
import sys
from models import *
from utils.utils import *

model = Darknet('cfg/yolov3-spp.cfg', (416, 416))
weights = sys.argv[1]
dev = '0'
device = torch_utils.select_device(dev)
model.load_state_dict(torch.load(weights, map_location=device)['model'])


with open('yolov3-spp_ultralytics68.wts', 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')

