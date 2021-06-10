import cv2 as cv
import numpy as np

import tensorrt as trt
import common

import torch
import time
from sys import argv

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def pre_process(image):
    long_size = max(image.shape)
    img = np.zeros((long_size, long_size, 3))
    img[:image.shape[0], :img.shape[1], :] = image[:]
    img = cv.resize(img, (512,512))
    inp_image = ((img / 255. - 0.5) / 0.5).astype(np.float32)
    images = inp_image.transpose(2, 0, 1)
    return images, long_size/512


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds.true_divide(width)).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind.true_divide(K)).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


if __name__ == '__main__':
    try:
        engine_path = argv[1]
        img_path = argv[2]
    except:
        print('engine path and image path are needed!')
        exit()
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            img = cv.imread('test.jpg')
            dis = img.copy()
            img, s = pre_process(img)
            # Copy to the pagelocked input buffer
            np.copyto(inputs[0].host, img.ravel())
            [hm, wh, reg] = common.do_inference(
                context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)

            [dets] = ctdet_decode(torch.from_numpy(hm.reshape(1, 80, 128, 128)), torch.from_numpy(
                wh.reshape(1, 2, 128, 128)), torch.from_numpy(reg.reshape(1, 2, 128, 128)))

            for i in dets:
                if i[-2] > 0.5:
                    i[:4] *= 4*s
                    cv.rectangle(dis, (int(i[0]), int(
                        i[1])), (int(i[2]), int(i[3])), 255, 1)
                    cv.putText(dis, '%d' %
                               int(i[-1]), (int(i[0]), int(i[1])), 1, 1, 255)

            cv.imwrite('trt_out.jpg', dis)
