import torch
from torch import nn
import torchvision
import os
import struct
from torchsummary import summary
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite

def export_as_weights(net, path_to_wts="models/ssdmobilenet.wts"):
    """ save the model weights """
    f = open(path_to_wts, 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
    print("Saved weights at ", path_to_wts)

def main():
    print('cuda device count: ', torch.cuda.device_count())
    DEVICE = 'cuda:0'
    class_names = [name.strip() for name in open('models/voc-model-labels.txt').readlines()]

    image = torch.ones(1, 3, 300, 300).to(DEVICE)

    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
    net.load('models/mb2-ssd-lite-mp-0_686.pth')
    net = net.to(DEVICE)

    net = net.eval()
    scores, boxes = net(image)

    print("Input shape ", image.shape)
    print("Scores shape ", scores.shape)
    print("Boxes shape ", boxes.shape)

    export_as_weights(net)

if __name__ == '__main__':
    main()
