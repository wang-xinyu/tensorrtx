import argparse
import struct

import torch


def main(args):
    # Load model
    state_dict = torch.load(args.weight)
    with open(args.save_path, "w") as f:
        f.write("{}\n".format(len(state_dict.keys())))
        for k, v in state_dict.items():
            vr = v.reshape(-1).cpu().numpy()
            f.write("{} {} ".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--weight",
        type=str,
        required=True,
        help="RepVGG model weight path",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=True,
        help="generated wts path",
    )
    args = parser.parse_args()
    main(args)