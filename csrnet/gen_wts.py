from torch.nn.modules import module
from model import CSRNet
import torch
import os
import struct


save_path = os.path.join(os.path.dirname(
    __file__), "output", os.path.basename(__file__).split('.')[0])
os.makedirs(save_path, exist_ok=True)
wts_file = os.path.join(save_path, "csrnet.wts")


# load model
model_path = "partBmodel_best.pth.tar"
model = CSRNet()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])


# save to wts
print(f'Writing into {wts_file}')
with open(wts_file, 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')