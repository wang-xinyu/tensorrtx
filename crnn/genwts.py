import torch
from torch.autograd import Variable
import utils
import models.crnn as crnn
import struct

model_path = './data/crnn.pth'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

image = torch.ones(1, 1, 32, 100)
if torch.cuda.is_available():
    image = image.cuda()

model.eval()
print(model)
print('image shape ', image.shape)
preds = model(image)

f = open("crnn.wts", 'w')
f.write("{}\n".format(len(model.state_dict().keys())))
for k,v in model.state_dict().items():
    print('key: ', k)
    print('value: ', v.shape)
    vr = v.reshape(-1).cpu().numpy()
    f.write("{} {}".format(k, len(vr)))
    for vv in vr:
        f.write(" ")
        f.write(struct.pack(">f", float(vv)).hex())
    f.write("\n")

