import torch
from torch.autograd import Variable

from LPRNet.model import LPRNET
import struct

model_path = './weights/Final_LPRNet_model.pth'
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]
model = LPRNET.LPRNet(class_num=len(CHARS), dropout_rate=0)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

image = torch.ones(1, 3, 24, 94)
if torch.cuda.is_available():
    image = image.cuda()

model.eval()
print(model)
print('image shape ', image.shape)
preds = model(image)

f = open("LPRNet.wts", 'w')
f.write("{}\n".format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    print('key: ', k)
    print('value: ', v.shape)
    vr = v.reshape(-1).cpu().numpy()
    f.write("{} {}".format(k, len(vr)))
    for vv in vr:
        f.write(" ")
        f.write(struct.pack(">f", float(vv)).hex())
    f.write("\n")
