from detectron2.layers import Conv2d
from torch import nn
import torch
import numpy as np
import struct
def fuse_conv_and_bn(conv):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    bn = conv.norm
    # init
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def fuse_bn(model):
    for child_name, child in model.named_children():
        if isinstance(child, Conv2d) and child.norm is not None:
            setattr(model, child_name, fuse_conv_and_bn(child))
        else:
            fuse_bn(child)

def gen_wts(model, filename):
    f = open('./' + filename + '.wts', 'w')
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')
    f.close()

# construct model
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
cfg = get_cfg()
cfg.merge_from_file('./configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml')
cfg.MODEL.WEIGHTS = './model_final_721ade.pkl'
cfg.MODEL.DEVICE = 'cpu'
model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval()
fuse_bn(model)
gen_wts(model, 'faster')

# test data
# from detectron2.data.detection_utils import read_image
# from detectron2.data import transforms as T
# import cv2
# original_image = cv2.imread('./demo.jpg')
# original_image = original_image.astype('float32')

# transform_gen = T.ResizeShortestEdge(
#             [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
#         )
# height, width = original_image.shape[:2]

# image = transform_gen.get_transform(original_image).apply_image(original_image)
# image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

# # model test
# inputs = {"image": image, "height": height, "width": width}

# with torch.no_grad():
#     predictions = model([inputs])[0]
# print (predictions)
