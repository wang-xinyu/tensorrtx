#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa: E501
# pylint: disable=no-name-in-module

"""
YOLOv1 weight export utilities.

This script provides functions to export a YOLOv1 PyTorch model's weights
to a .wts text file and to binary .bin files for each tensor, which can be
used for TensorRT engine building.

Functions:
    save_name_and_shape_to_txt(weight_dict, txt_path)
        Save weight names and their shapes to a txt file.
    export_wts(cfg_path, weight_path, output_wts="yolov1.wts")
        Export all model weights to a .wts file.
    export_bin(cfg_path, weight_path, output_dir="yolov1_bin")
        Export each tensor in model state_dict to individual .bin files.
    main()
        Main script entry point to generate .wts and .bin files from a model.
"""

import os
import sys
import struct

from utils import parse_cfg, build_model


def save_name_and_shape_to_txt(weight_dict, txt_path):
    """
    save weight name and shape as txt file.

    args:
        weight_dict: dict, key = weight name (str), value = numpy
        txt_path: path that save txt file
    """
    with open(txt_path, "w", encoding="utf-8") as f:
        for name, arr in weight_dict.items():
            f.write(f"{name}\t{tuple(arr.shape)}\n")

    print(f"Saved weight shapes to {txt_path}")


def export_wts(cfg_path, weight_path, output_wts="yolov1.wts"):
    """
    Export YOLOv1 model weights to a .wts text file.

    Args:
        cfg_path (str): Path to YOLOv1 config yaml
        weight_path (str): Path to PyTorch model weights (.pth)
        output_wts (str): Output .wts file path
    """
    # 1. load cfg
    cfg = parse_cfg(cfg_path)
    s, b, num_classes = cfg["S"], cfg["B"], cfg["num_classes"]

    # 2. load model
    print("[INFO] Loading YOLOv1 model...")
    model = build_model(weight_path, s, b, num_classes)
    model = model.eval()

    # 3. export state_dict
    state_dict = model.state_dict()
    print(f"[INFO] Total weights: {len(state_dict.keys())}")

    os.makedirs("./yolov1_bin", exist_ok=True)

    save_name_and_shape_to_txt(state_dict, "./yolov1_bin/weight_name_and_shape.txt")

    with open(output_wts, "w", encoding="utf-8") as f:
        # first line: number of tensors
        # f.write("{}\n".format(len(state_dict.keys())))
        f.write(f"{len(state_dict.keys())}\n")

        for k, v in state_dict.items():
            print(f"[INFO] Processing: {k}, shape: {v.shape}")

            vr = v.reshape(-1).cpu().numpy()
            # f.write("{} {}".format(k, len(vr)))
            f.write(f"{k} {len(vr)}")

            for vv in vr:
                # convert float → hex string
                hex_str = struct.pack(">f", float(vv)).hex()
                f.write(" " + hex_str)

            f.write("\n")

    print(f"[INFO] .wts file saved to: {output_wts}")


def export_bin(cfg_path, weight_path, output_dir="yolov1_bin"):
    """
    Export each tensor in model state_dict to separate binary .bin files.

    Args:
        cfg_path (str): Path to YOLOv1 config yaml
        weight_path (str): Path to PyTorch model weights (.pth)
        output_dir (str): Directory to save .bin files
    """
    # 1. load cfg
    cfg = parse_cfg(cfg_path)
    s, b, num_classes = cfg["S"], cfg["B"], cfg["num_classes"]

    # 2. load model
    print("[INFO] Loading YOLOv1 model...")
    model = build_model(weight_path, s, b, num_classes)
    model = model.eval()

    # 3. export
    state_dict = model.state_dict()
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Total tensors: {len(state_dict)}")

    for k, v in state_dict.items():
        vr = v.reshape(-1).cpu().numpy().astype("float32")

        out_path = os.path.join(output_dir, k.replace(".", "_") + ".bin")

        print(f"[INFO] Saving: {out_path}  shape={v.shape}  bytes={vr.nbytes}")

        # save as pure binary float32
        vr.tofile(out_path)

    print(f"[INFO] Export complete. All .bin files saved to: {output_dir}")


def main():
    """
    Main script entry point.

    Usage:
        # gene wts and bin file.
        python gen_wts_bin_files.py cfg/yolov1.yaml weights/yolov1.pth
    """
    if len(sys.argv) < 3:
        print("Usage: python yolov1_export_wts.py <cfg> <weights> [output.wts]")
        sys.exit(0)

    cfg_path = sys.argv[1]
    weight_path = sys.argv[2]
    output_wts = sys.argv[3] if len(sys.argv) > 3 else "yolov1.wts"

    export_wts(cfg_path, weight_path, output_wts)
    export_bin(cfg_path, weight_path)


if __name__ == "__main__":
    main()
