import struct
import torch

state_dict = torch.load("yolov7.pt", map_location="cpu")
state_dict = state_dict["model"].state_dict()

with open("yolov7.wts", 'w') as f:
    f.write("{}\n".format(len(state_dict.keys())))

    for k, v in state_dict.items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')
