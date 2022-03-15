import torch
import struct
import sys

# Initialize
pt_file = sys.argv[1]
# Load model
model = torch.load(pt_file, map_location=torch.device('cpu'))['model'].float()  # load to FP32
model.to(device).eval()

with open(pt_file.split('.')[0] + '.wts', 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')
