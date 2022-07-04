import torch
import struct
from model import SuperPointNet

model_name = "superpoint_v1"

net = SuperPointNet()
net.load_state_dict(torch.load("superpoint_v1.pth"))
net = net.cuda()
net.eval()

f = open(model_name + ".wts", "w")
f.write("{}\n".format(len(net.state_dict().keys())))
for k, v in net.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write("{} {}".format(k, len(vr)))
    for vv in vr:
        f.write(" ")
        f.write(struct.pack(">f", float(vv)).hex())
    f.write("\n")