# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import pprint

import torch
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
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger
from core.evaluate import accuracy

import cv2
import numpy as np
from PIL import Image
import struct

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=r"E:\LearningCodes\GithubRepo\HRNet-Image-Classification\experiments\cls_hrnet_w18_small_v2_sgd_lr5e-2_wd1e-4_bs32_x100.yaml",
                        type=str)
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default=r'E:\LearningCodes\GithubRepo\HRNet-Image-Classification\hrnet_w18_small_model_v2.pth')
    parser.add_argument('--testImg',
                    help='imgs',
                    type=str,
                    default=r'E:\Datasets\tiny-imagenet-200\tiny-imagenet-200\val\images\val_41.JPEG')
    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    savewts = False
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'demo')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # eval() 函数用来执行一个字符串表达式，并返回表达式的值。
    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config)

    model.load_state_dict(torch.load(args.testModel))

    if savewts:
        f = open('HRNetClassify.wts', 'w')
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
    image = cv2.imread(args.testImg) #BGR 0-255 hwc
    #im = Image.open(args.testImg)
    #print(im.getpixel((0,0)))  ## 0-255
    #resize
    # config.MODEL.IMAGE_SIZE[0]
    resized_img = cv2.resize(image, (config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB) #RGB
    # normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inp_image = ((resized_img/255. - mean) / std).astype(np.float32) # R-0.485  B-
    inp_image = inp_image.transpose(2, 0, 1) # chw
    inp_image = torch.from_numpy(inp_image).unsqueeze(0) # to_tensor
    model.eval()
    output = model(inp_image)
    #print(output)

    _, pred = output.topk(1)
    pred = pred.t()
    print(pred)
if __name__ == "__main__":
    main()