import os, sys
import torch
import struct

# TODO: YOLOP_BASE_DIR is the root of YOLOP
print("[WARN] Please download/clone YOLOP, then set YOLOP_BASE_DIR to the root of YOLOP")

#YOLOP_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YOLOP_BASE_DIR = "/home/user/jetson/tmp/YOLOP"

sys.path.append(YOLOP_BASE_DIR)
from lib.models import get_net
from lib.config import cfg


# Initialize
device = torch.device('cpu')
# Load model
model = get_net(cfg)
checkpoint = torch.load(YOLOP_BASE_DIR + '/weights/End-to-end.pth', map_location=device)
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

print("save as yolop.wts")