from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import pprint

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import _init_paths
import models
from config import config
from config import update_config
from utils.modelsummary import get_model_summary
from utils.utils import create_logger

import cv2
import numpy as np
from PIL import Image
import struct
import time

import torch.onnx as torch_onnx

ignore_label = -1
label_mapping = {-1: ignore_label, 0: ignore_label,
                 1: ignore_label, 2: ignore_label,
                 3: ignore_label, 4: ignore_label,
                 5: ignore_label, 6: ignore_label,
                 7: 0, 8: 1, 9: ignore_label,
                 10: ignore_label, 11: 2, 12: 3,
                 13: 4, 14: ignore_label, 15: ignore_label,
                 16: ignore_label, 17: 5, 18: ignore_label,
                 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                 25: 12, 26: 13, 27: 14, 28: 15,
                 29: ignore_label, 30: ignore_label,
                 31: 16, 32: 17, 33: 18}

def convert_label(label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in label_mapping.items():
                label[temp == k] = v
        return label

def get_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def save_pred(preds, sv_path, name):
    palette = get_palette(256)
    #preds = preds.cpu().numpy().copy()
    preds = preds.detach().cpu().numpy().copy()
    # numpy.argmax(array, axis) 用于返回一个numpy数组中最大值的索引值
    # (1 19 512 1024) 在19维度上输出每个维度最大值的索引
    # np.argmax(preds, axis=1) 返回 （1， 512， 1024）
    preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8) # argmax
    for i in range(preds.shape[0]):
        pred = convert_label(preds[i], inverse=True)
        save_img = Image.fromarray(pred)
        save_img.putpalette(palette)
        #save_img.save(os.path.join(sv_path, name[i]+'.png'))
        save_img.show()
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=r"../experiments/cityscapes/seg_hrnet_w18_small_v2_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    savewts = False
    save_onnx = False
    args = parse_args()

    logger, final_output_dir, _ = create_logger(config, args.cfg, 'demo')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # eval() 函数用来执行一个字符串表达式，并返回表达式的值。
    model = eval('models.'+config.MODEL.NAME+'.get_seg_model')(config)

    model_state_file = "../pretrained_models/hrnet_w18_small_v2_cityscapes_cls19_1024x2048_trainset.pth"
    logger.info('=> loading model from {}'.format(model_state_file))
    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if savewts:
        logger.info('=> saving {} '.format('HRNetSeg.wts'))
        f = open('HRNetSeg.wts', 'w')
        f.write('{}\n'.format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')
        exit(0)
    # load img
    #testImg = "../data/cityscapes/images/bonn_000014_000019_leftImg8bit.png"
    testImg = r"E:\Datasets\oneimg\munster_000000_000019_rightImg8bit.png"
    image = cv2.imread(testImg) #BGR 0-255 hwc
    resized_img = cv2.resize(image, (config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1]))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB) #RGB
    # normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inp_image = ((resized_img/255. - mean) / std).astype(np.float32)
    inp_image = inp_image.transpose(2, 0, 1)
    inp_image = torch.from_numpy(inp_image).unsqueeze(0) # to_tensor
    model.eval()
    if save_onnx:
        logger.info('=> saving {} '.format('hrnet.onnx'))
        torch_onnx.export(model, inp_image, "hrnet.onnx", verbose=True, input_names=["input"],
                          output_names=["output"],opset_version=12)
    start = time.time()
    output = model(inp_image)
    if output.size()[-2] != config.TRAIN.IMAGE_SIZE[0] or output.size()[-1] != config.TRAIN.IMAGE_SIZE[1]:
        output = F.upsample(output, (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0]), mode='bilinear', align_corners=True)
    end = time.time()
    print(end-start)
    save_pred(output, "./", "1")
if __name__ == "__main__":
    main()
