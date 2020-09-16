import torch
import struct
#import models.crnn as crnn
from model.model import parsingNet

# Initialize
model = parsingNet(pretrained = False, backbone='18', cls_dim = (101, 56, 4), use_aux=False).cuda()
device = 'cpu'
# Load model
state_dict = torch.load('tusimple_18.pth', map_location='cpu')['model']
model.to(device).eval()

f = open('lane.wts', 'w')
f.write('{}\n'.format(len(state_dict.keys())))
for k, v in state_dict.items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')
