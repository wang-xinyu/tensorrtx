import os, sys
import torch
import struct

# TODO: BASE_DIR 是 YOLOP 的根目录
#BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = "/home/jetson/tmp/YOLOP"

sys.path.append(BASE_DIR)
from lib.models import get_net
from lib.config import cfg


# Initialize
device = torch.device('cpu')
# Load model
model = get_net(cfg)
checkpoint = torch.load(BASE_DIR + '/weights/End-to-end.pth', map_location=device)
model.load_state_dict(checkpoint['state_dict'])
# load to FP32
model.float()
model.to(device).eval()

f = open('yolop.wts', 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')

f.close()
