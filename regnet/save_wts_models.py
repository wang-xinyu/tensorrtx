import torch
import struct
from torchsummary import summary


def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torch.load('./models/regnet_x_400mf.pth')
    net = net.to('cuda:0')
    net.eval()

    tmp = torch.ones(1, 3, 224, 224).to('cuda:0')
    out = net(tmp)

    print('output:', out)

    summary(net, (3, 224, 224))

    f = open("./models/regnet_x_400mf.wts", 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k, v in net.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")


if __name__ == '__main__':
    main()
