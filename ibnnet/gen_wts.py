import torch
import os
import sys
import struct


assert sys.argv[1] == "a" or sys.argv[1] == "b"
model_name = "resnet50_ibn_" + sys.argv[1]

net = torch.hub.load('XingangPan/IBN-Net', model_name, pretrained=True).to('cuda:0').eval()

#verify
#input = torch.ones(1, 3, 224, 224).to('cuda:0')
#pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to('cuda:0')
#pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to('cuda:0')
#input.sub_(pixel_mean).div_(pixel_std)
#out = net(input)
#print(out)

f = open(model_name + ".wts", 'w')
f.write("{}\n".format(len(net.state_dict().keys())))
for k,v in net.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write("{} {}".format(k, len(vr)))
    for vv in vr:
        f.write(" ")
        f.write(struct.pack(">f", float(vv)).hex())
    f.write("\n")


