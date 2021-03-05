# TensorRTx

TensorRTx aims to implement popular deep learning networks with tensorrt network definition APIs. As we know, tensorrt has builtin parsers, including caffeparser, uffparser, onnxparser, etc. But when we use these parsers, we often run into some "unsupported operations or layers" problems, especially some state-of-the-art models are using new type of layers.

So why don't we just skip all parsers? We just use TensorRT network definition APIs to build the whole network, it's not so complicated.

I wrote this project to get familiar with tensorrt API, and also to share and learn from the community.

All the models are implemented in pytorch/mxnet/tensorflown first, and export a weights file xxx.wts, and then use tensorrt to load weights, define network and do inference. Some pytorch implementations can be found in my repo [Pytorchx](https://github.com/wang-xinyu/pytorchx), the remaining are from polular open-source implementations.

## News

- `5 Mar 2021`. [chgit0214](https://github.com/chgit0214) added the LPRNet.
- `31 Jan 2021`. RepVGG added by [upczww](https://github.com/upczww).
- `29 Jan 2021`. U-Net added by [YuzhouPeng](https://github.com/YuzhouPeng).
- `24 Jan 2021`. IBN-Net added by [TCHeish](https://github.com/TCHeish), PSENet optimized, YOLOv5 v4.0 INT8, etc.
- `8 Jan 2021`. YOLOv5 s/m/l/x updated to v4.0.
- `27 Dec 2020`. HRNet-Semantic-Segmentation added by [BaofengZan](https://github.com/BaofengZan).
- `4 Dec 2020`. DBNet dynamic input shape support by [BaofengZan](https://github.com/BaofengZan), YOLOv3 int8, PSENet(tensorflow) text detection by [upczww](https://github.com/upczww).
- `19 Nov 2020`. YOLOv3-SPP supports dynamic input shape, including a dynamic plugin.
- `17 Nov 2020`. [AlfengYuan](https://github.com/AlfengYuan) added a Dockerfile.
- `7 Nov 2020`. All models migrated to trt7 API, and clean up the master branch.
- `29 Oct 2020`. First INT8 quantization implementation! Please check retinaface.
- `23 Oct 2020`. Add a .wts model zoo for quick evaluation.
- `16 Sep 2020`. [hwh-hit](https://github.com/hwh-hit) added ufld(Ultra-Fast-Lane-Detection, ECCV2020).
- `13 Sep 2020`. Add crnn, and got 1000fps on GTX1080.
- `28 Aug 2020`. [BaofengZan](https://github.com/BaofengZan) added a tutorial for compiling and running tensorrtx on windows.

## Tutorials

- [Install the dependencies.](./tutorials/install.md)
- [A guide for quickly getting started, taking lenet5 as a demo.](./tutorials/getting_started.md)
- [The .wts file content format](./tutorials/getting_started.md#the-wts-content-format)
- [Frequently Asked Questions (FAQ)](./tutorials/faq.md)
- [Migrating from TensorRT 4 to 7](./tutorials/migrating_from_tensorrt_4_to_7.md)
- [How to implement multi-GPU processing, taking YOLOv4 as example](./tutorials/multi_GPU_processing.md)
- [Check if Your GPU support FP16/INT8](./tutorials/check_fp16_int8_support.md)
- [How to Compile and Run on Windows](./tutorials/run_on_windows.md)
- [Deploy YOLOv4 with Triton Inference Server](https://github.com/isarsoft/yolov4-triton-tensorrt)
- [From pytorch to trt step by step, hrnet as example(Chinese)](./tutorials/from_pytorch_to_trt_stepbystep_hrnet.md)

## Test Environment

1. GTX1080 / Ubuntu16.04 / cuda10.0 / cudnn7.6.5 / tensorrt7.0.0 / nvinfer7.0.0 / opencv3.3

## How to run

Each folder has a readme inside, which explains how to run the models inside.

## Models

Following models are implemented.

|Name | Description |
|-|-|
|[lenet](./lenet) | the simplest, as a "hello world" of this project |
|[alexnet](./alexnet)| easy to implement, all layers are supported in tensorrt |
|[googlenet](./googlenet)| GoogLeNet (Inception v1) |
|[inception](./inceptionv3)| Inception v3 |
|[mnasnet](./mnasnet)| MNASNet with depth multiplier of 0.5 from the paper |
|[mobilenetv2](./mobilenetv2)| MobileNet V2 |
|[mobilenetv3](./mobilenetv3)| V3-small, V3-large. |
|[resnet](./resnet)| resnet-18, resnet-50 and resnext50-32x4d are implemented |
|[senet](./senet)| se-resnet50 |
|[shufflenet](./shufflenetv2)| ShuffleNetV2 with 0.5x output channels |
|[squeezenet](./squeezenet)| SqueezeNet 1.1 model |
|[vgg](./vgg)| VGG 11-layer model |
|[yolov3-tiny](./yolov3-tiny)| weights and pytorch implementation from [ultralytics/yolov3](https://github.com/ultralytics/yolov3) |
|[yolov3](./yolov3)| darknet-53, weights and pytorch implementation from [ultralytics/yolov3](https://github.com/ultralytics/yolov3) |
|[yolov3-spp](./yolov3-spp)| darknet-53, weights and pytorch implementation from [ultralytics/yolov3](https://github.com/ultralytics/yolov3) |
|[yolov4](./yolov4)| CSPDarknet53, weights from [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet#pre-trained-models), pytorch implementation from [ultralytics/yolov3](https://github.com/ultralytics/yolov3) |
|[yolov5](./yolov5)| yolov5-s/m/l/x v1.0 v2.0 v3.0 v3.1, pytorch implementation from [ultralytics/yolov5](https://github.com/ultralytics/yolov5) |
|[retinaface](./retinaface)| resnet50 and mobilnet0.25, weights from [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) |
|[arcface](./arcface)| LResNet50E-IR, weights from [deepinsight/insightface](https://github.com/deepinsight/insightface) |
|[retinafaceAntiCov](./retinafaceAntiCov)| mobilenet0.25, weights from [deepinsight/insightface](https://github.com/deepinsight/insightface), retinaface anti-COVID-19, detect face and mask attribute |
|[dbnet](./dbnet)| Scene Text Detection, weights from [BaofengZan/DBNet.pytorch](https://github.com/BaofengZan/DBNet.pytorch) |
|[crnn](./crnn)| pytorch implementation from [meijieru/crnn.pytorch](https://github.com/meijieru/crnn.pytorch) |
|[ufld](./ufld)| pytorch implementation from [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection), ECCV2020 |
|[hrnet](./hrnet)| hrnet-image-classification and hrnet-semantic-segmentation, pytorch implementation from [HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification) and [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) |
|[psenet](./psenet)| PSENet Text Detection, tensorflow implementation from [liuheng92/tensorflow_PSENet](https://github.com/liuheng92/tensorflow_PSENet) |
|[ibnnet](./ibnnet)| IBN-Net, pytorch implementation from [XingangPan/IBN-Net](https://github.com/XingangPan/IBN-Net), ECCV2018 |
|[unet](./unet)| U-Net, pytorch implementation from [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) |
|[repvgg](./repvgg)| RepVGG, pytorch implementation from [DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG) |
|[lprnet](./lprnet)| LPRNet, pytorch implementation from [xuexingyu24/License_Plate_Detection_Pytorch](https://github.com/xuexingyu24/License_Plate_Detection_Pytorch) |

## Model Zoo

The .wts files can be downloaded from model zoo for quick evaluation. But it is recommended to convert .wts from pytorch/mxnet/tensorflow model, so that you can retrain your own model.

[BaiduPan](https://pan.baidu.com/s/19s6hO8esU7-TtZEXN7G3OA) pwd: uvv2

## Tricky Operations

Some tricky operations encountered in these models, already solved, but might have better solutions.

|Name | Description |
|-|-|
|BatchNorm| Implement by a scale layer, used in resnet, googlenet, mobilenet, etc. |
|MaxPool2d(ceil_mode=True)| use a padding layer before maxpool to solve ceil_mode=True, see googlenet. |
|average pool with padding| use setAverageCountExcludesPadding() when necessary, see inception. |
|relu6| use `Relu6(x) = Relu(x) - Relu(x-6)`, see mobilenet. |
|torch.chunk()| implement the 'chunk(2, dim=C)' by tensorrt plugin, see shufflenet. |
|channel shuffle| use two shuffle layers to implement `channel_shuffle`, see shufflenet. |
|adaptive pool| use fixed input dimension, and use regular average pooling, see shufflenet. |
|leaky relu| I wrote a leaky relu plugin, but PRelu in `NvInferPlugin.h` can be used, see yolov3 in branch `trt4`. |
|yolo layer v1| yolo layer is implemented as a plugin, see yolov3 in branch `trt4`. |
|yolo layer v2| three yolo layers implemented in one plugin, see yolov3-spp. |
|upsample| replaced by a deconvolution layer, see yolov3. |
|hsigmoid| hard sigmoid is implemented as a plugin, hsigmoid and hswish are used in mobilenetv3 |
|retinaface output decode| implement a plugin to decode bbox, confidence and landmarks, see retinaface. |
|mish| mish activation is implemented as a plugin, mish is used in yolov4 |
|prelu| mxnet's prelu activation with trainable gamma is implemented as a plugin, used in arcface |
|HardSwish| hard_swish = x * hard_sigmoid, used in yolov5 v3.0 |
|LSTM| Implemented pytorch nn.LSTM() with tensorrt api |

## Speed Benchmark

| Models | Device | BatchSize | Mode | Input Shape(HxW) | FPS |
|-|-|:-:|:-:|:-:|:-:|
| YOLOv3-tiny | Xeon E5-2620/GTX1080 | 1 | FP32 | 608x608 | 333 |
| YOLOv3(darknet53) | Xeon E5-2620/GTX1080 | 1 | FP32 | 608x608 | 39.2 |
| YOLOv3(darknet53) | Xeon E5-2620/GTX1080 | 1 | INT8 | 608x608 | 71.4 |
| YOLOv3-spp(darknet53) | Xeon E5-2620/GTX1080 | 1 | FP32 | 608x608 | 38.5 |
| YOLOv4(CSPDarknet53) | Xeon E5-2620/GTX1080 | 1 | FP32 | 608x608 | 35.7 |
| YOLOv4(CSPDarknet53) | Xeon E5-2620/GTX1080 | 4 | FP32 | 608x608 | 40.9 |
| YOLOv4(CSPDarknet53) | Xeon E5-2620/GTX1080 | 8 | FP32 | 608x608 | 41.3 | 
| YOLOv5-s v3.0 | Xeon E5-2620/GTX1080 | 1 | FP32 | 608x608 | 142 |
| YOLOv5-s v3.0 | Xeon E5-2620/GTX1080 | 4 | FP32 | 608x608 | 173 |
| YOLOv5-s v3.0 | Xeon E5-2620/GTX1080 | 8 | FP32 | 608x608 | 190 |
| YOLOv5-m v3.0 | Xeon E5-2620/GTX1080 | 1 | FP32 | 608x608 | 71 |
| YOLOv5-l v3.0 | Xeon E5-2620/GTX1080 | 1 | FP32 | 608x608 | 43 |
| YOLOv5-x v3.0 | Xeon E5-2620/GTX1080 | 1 | FP32 | 608x608 | 29 |
| YOLOv5-s v4.0 | Xeon E5-2620/GTX1080 | 1 | FP32 | 608x608 | 142 |
| YOLOv5-m v4.0 | Xeon E5-2620/GTX1080 | 1 | FP32 | 608x608 | 71 |
| YOLOv5-l v4.0 | Xeon E5-2620/GTX1080 | 1 | FP32 | 608x608 | 40 |
| YOLOv5-x v4.0 | Xeon E5-2620/GTX1080 | 1 | FP32 | 608x608 | 27 |
| RetinaFace(resnet50) | Xeon E5-2620/GTX1080 | 1 | FP32 | 480x640 | 90 |
| RetinaFace(resnet50) | Xeon E5-2620/GTX1080 | 1 | INT8 | 480x640 | 204 |
| RetinaFace(mobilenet0.25) | Xeon E5-2620/GTX1080 | 1 | FP32 | 480x640 | 417 |
| ArcFace(LResNet50E-IR) | Xeon E5-2620/GTX1080 | 1 | FP32 | 112x112 | 333 |
| CRNN | Xeon E5-2620/GTX1080 | 1 | FP32 | 32x100 | 1000 |

Help wanted, if you got speed results, please add an issue or PR.

## Acknowledgments & Contact

Any contributions, questions and discussions are welcomed, contact me by following info.

E-mail: wangxinyu_es@163.com

WeChat ID: wangxinyu0375 (可加我微信进tensorrtx交流群，**备注：tensorrtx**)
