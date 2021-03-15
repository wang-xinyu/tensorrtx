import torch
import torch.nn as nn
import struct
from models.refinedet import build_refinedet



num_classes = 25
path_model = "/data_2/project_2021/pytorch_refinedet/2021/20210308.pth"
path_save_wts = "./refinedet0312.wts"
input_size = 320

net = build_refinedet('test', input_size, num_classes)  # initialize net
net.load_state_dict(torch.load(path_model))
net.eval()


f = open(path_save_wts, 'w')
f.write('{}\n'.format(len(net.state_dict().keys())))
for k, v in net.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')

print("success generate wts!")