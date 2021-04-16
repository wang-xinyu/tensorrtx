import argparse
import struct

import torch
import numpy as np


def write_one_weight(writer, name, weight):
    assert isinstance(weight, np.ndarray)
    values = weight.reshape(-1)
    writer.write('{} {}'.format(name, len(values)))
    for value in values:
        writer.write(' ')
        # float to bytes to hex_string
        writer.write(struct.pack('>f', float(value)).hex())
    writer.write('\n')


def convert_name(name):
    return name.replace("module.", "").replace("base_model.", "").\
        replace("net.", "").replace("new_fc", "fc").replace("backbone.", "").\
        replace("cls_head.fc_cls", "fc").replace(".conv.", ".").\
        replace("conv1.bn", "bn1").replace("conv2.bn", "bn2").\
        replace("conv3.bn", "bn3").replace("downsample.bn", "downsample.1").\
        replace("downsample.weight", "downsample.0.weight")


def main(args):
    ckpt = torch.load(args.checkpoint)['state_dict']
    ckpt = {k: v for k, v in ckpt.items() if 'num_batches_tracked' not in k}
    with open(args.out_filename, "w") as f:
        f.write(f"{len(ckpt)}\n")
        for k, v in ckpt.items():
            key = convert_name(k)
            write_one_weight(f, key, v.cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--out-filename",
                        type=str,
                        default="tsm_r50.wts",
                        help="Path to converted wegiths file")
    args = parser.parse_args()
    main(args)
