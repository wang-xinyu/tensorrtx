import os
import torch
import torchvision
import contextlib
from torchinfo import summary

torch.set_printoptions(threshold=10000)


def main():
    print('cuda device count:', torch.cuda.device_count())

    net = torchvision.models.regnet_x_400mf(pretrained=True)
    net.eval()
    net = net.to('cuda:0')

    os.makedirs("./models", exist_ok=True)

    # summary(net, input_size=(2, 3, 224, 224))
    with open("./models/regnet_x_400mf_summary.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            summary(
                net,
                input_size=(2, 3, 224, 224),
                depth=5,
                col_names=("input_size", "output_size", "num_params"),
                verbose=2
            )

    print(net)

    torch.save(net, "./models/regnet_x_400mf.pth")
    print("regnet_x_400mf.pth saved successfully!")


if __name__ == '__main__':
    main()
