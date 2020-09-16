import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.onnx as torch_onnx
from model.model import parsingNet

MODELPATH = "tusimple_18.pth"

net = parsingNet(pretrained = False, backbone='18', cls_dim = (101, 56, 4), use_aux=False).cuda()

state_dict = torch.load(MODELPATH, map_location='cpu')['model']

net.train(False)

x = torch.randn(1, 3, 288, 800).cuda()

torch_onnx.export(net, x, "lane.onnx", verbose=True, input_names=["input"], output_names=["output"],opset_version=11)
