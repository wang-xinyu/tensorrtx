import argparse
import struct

import _init_paths
import models
import torch
from config import config, update_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")

    parser.add_argument("--cfg", help="experiment configure file name", type=str)
    parser.add_argument("--ckpt_path", help="checkpoint path", required=True, type=str)
    parser.add_argument("--save_path", help=".wts path", required=True, type=str)

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    model = eval("models." + config.MODEL.NAME + ".get_seg_model")(config)

    print("=> loading model from {}".format(args.ckpt_path))
    pretrained_dict = torch.load(args.ckpt_path, map_location="cpu")
    model_dict = model.state_dict()
    pretrained_dict = {
        k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()
    }
    for k, _ in pretrained_dict.items():
        print("=> loading {} from pretrained model".format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print("=> saving {} ".format(args.save_path))
    f = open(args.save_path, "w")
    f.write("{}\n".format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {} ".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
    f.close()


if __name__ == "__main__":
    main()
