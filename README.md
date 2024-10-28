# TensorRTx

TensorRTx aims to implement popular deep learning networks with TensorRT network definition API.

Why don't we use a parser (ONNX parser, UFF parser, caffe parser, etc), but use complex APIs to build a network from scratch? I have summarized the advantages in the following aspects.
- **Flexible**, easy to modify the network, add/delete a layer or input/output tensor, replace a layer, merge layers, integrate preprocessing and postprocessing into network, etc.
- **Debuggable**, construct the entire network in an incremental development manner, easy to get middle layer results.
- **Educational**, learn about the network structure during this development, rather than treating everything as a black box.

The basic workflow of TensorRTx is:
1. Get the trained models from pytorch, mxnet or tensorflow, etc. Some pytorch models can be found in my repo [pytorchx](https://github.com/wang-xinyu/pytorchx), the remaining are from popular open-source repos.
2. Export the weights to a plain text file -- [.wts file](./tutorials/getting_started.md#the-wts-content-format).
3. Load weights in TensorRT, define the network, build a TensorRT engine.
4. Load the TensorRT engine and run inference.

## News

- `22 Oct 2024`. [lindsayshuo](https://github.com/lindsayshuo): YOLOv8-obb
- `18 Oct 2024`. [zgjja](https://github.com/zgjja): Refactor docker image.
- `11 Oct 2024`. [mpj1234](https://github.com/mpj1234): YOLO11
- `9 Oct 2024`. [Phoenix8215](https://github.com/Phoenix8215): GhostNet V1 and V2.
- `21 Aug 2024`. [Lemonononon](https://github.com/Lemonononon): real-esrgan-general-x4v3
- `29 Jul 2024`. [mpj1234](https://github.com/mpj1234): Check the YOLOv5, YOLOv8 & YOLOv10 in TensorRT 10.x API, branch → [trt10](https://github.com/wang-xinyu/tensorrtx/tree/trt10)
- `29 Jul 2024`. [mpj1234](https://github.com/mpj1234): YOLOv10
- `21 Jun 2024`. [WuxinrongY](https://github.com/WuxinrongY): YOLOv9-T, YOLOv9-S, YOLOv9-M
- `28 Apr 2024`. [lindsayshuo](https://github.com/lindsayshuo): YOLOv8-pose
- `22 Apr 2024`. [B1SH0PP](https://github.com/B1SH0PP): EfficientAd: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.
- `18 Apr 2024`. [lindsayshuo](https://github.com/lindsayshuo): YOLOv8-p2
- `12 Mar 2024`. [lindsayshuo](https://github.com/lindsayshuo): YOLOv8-cls
- `11 Mar 2024`. [WuxinrongY](https://github.com/WuxinrongY): YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information
- `7 Mar 2024`. [AadeIT](https://github.com/AadeIT): CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes
- `17 Oct 2023`. [Rex-LK](https://github.com/Rex-LK): YOLOv8-Seg

## Tutorials

- [How to make contribution](./tutorials/contribution.md)
- [Install the dependencies.](./tutorials/install.md)
- [A guide for quickly getting started, taking lenet5 as a demo.](./tutorials/getting_started.md)
- [The .wts file content format](./tutorials/getting_started.md#the-wts-content-format)
- [Frequently Asked Questions (FAQ)](./tutorials/faq.md)
- [Migration Guide](./tutorials/migration_guide.md)
- [How to implement multi-GPU processing, taking YOLOv4 as example](./tutorials/multi_GPU_processing.md)
- [Check if Your GPU support FP16/INT8](./tutorials/check_fp16_int8_support.md)
- [How to Compile and Run on Windows](./tutorials/run_on_windows.md)
- [Deploy YOLOv4 with Triton Inference Server](https://github.com/isarsoft/yolov4-triton-tensorrt)
- [From pytorch to trt step by step, hrnet as example(Chinese)](./tutorials/from_pytorch_to_trt_stepbystep_hrnet.md)

## Test Environment

1. (**NOT recommended**) TensorRT 7.x
2. (**Recommended**)TensorRT 8.x
3. (**NOT recommended**) TensorRT 10.x

### Note

1. For history reason, some of the models are limited to specific TensorRT version, please check the README.md or code for the model you want to use.
2. Currently, TensorRT 8.x has better compatibility and the most of the features  supported.

## How to run

**Note**: this project support to build each network by the `CMakeLists.txt` in its subfolder, or you can build them together by the `CMakeLists.txt` on top of this project.

* General procedures before building and running:

```bash
# 1. generate xxx.wts from https://github.com/wang-xinyu/pytorchx/tree/master/lenet
# ...

# 2. put xxx.wts on top of this folder
# ...
```

* (*Option 1*) To build a single subproject in this project, do:

```bash
## enter the subfolder
cd tensorrtx/xxx

## configure & build
cmake -S . -B build
make -C build
```

* (*Option 2*) To build many subprojects, firstly, in the top `CMakeLists.txt`, **uncomment** the project you don't want to build or not suppoted by your TensorRT version, e.g., you cannot build subprojects in `${TensorRT_8_Targets}` if your TensorRT is `7.x`. Then:

```bash
## enter the top of this project
cd tensorrtx

## configure & build
# you may use "Ninja" rather than "make" to significantly boost the build speed
cmake -G Ninja -S . -B build
ninja -C build
```

**WARNING**: This part is still under development, most subprojects are not adapted yet.

* run the generated executable, e.g.:

```bash
# serialize model to plan file i.e. 'xxx.engine'
build/xxx -s

# deserialize plan file and run inference
build/xxx -d

# (Optional) check if the output is same as pytorchx/lenet
# ...

# (Optional) customize the project
# ...
```

For more details, each subfolder may contain a `README.md` inside, which explains more.

## Models

Following models are implemented.

| Name | Description | Supported TensorRT Version |
|---------------|---------------|---------------|
|[mlp](./mlp) | the very basic model for starters, properly documented | 7.x/8.x/10.x |
|[lenet](./lenet) | the simplest, as a "hello world" of this project | 7.x/8.x/10.x |
|[alexnet](./alexnet)| easy to implement, all layers are supported in tensorrt |
|[googlenet](./googlenet)| GoogLeNet (Inception v1) |
|[inception](./inception)| Inception v3, v4 |
|[mnasnet](./mnasnet)| MNASNet with depth multiplier of 0.5 from the paper |
|[mobilenet](./mobilenet)| MobileNet v2, v3-small, v3-large |
|[resnet](./resnet)| resnet-18, resnet-50 and resnext50-32x4d are implemented |
|[senet](./senet)| se-resnet50 |
|[shufflenet](./shufflenetv2)| ShuffleNet v2 with 0.5x output channels |
|[squeezenet](./squeezenet)| SqueezeNet 1.1 model |
|[vgg](./vgg)| VGG 11-layer model |
|[yolov3-tiny](./yolov3-tiny)| weights and pytorch implementation from [ultralytics/yolov3](https://github.com/ultralytics/yolov3) |
|[yolov3](./yolov3)| darknet-53, weights and pytorch implementation from [ultralytics/yolov3](https://github.com/ultralytics/yolov3) |
|[yolov3-spp](./yolov3-spp)| darknet-53, weights and pytorch implementation from [ultralytics/yolov3](https://github.com/ultralytics/yolov3) |
|[yolov4](./yolov4)| CSPDarknet53, weights from [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet#pre-trained-models), pytorch implementation from [ultralytics/yolov3](https://github.com/ultralytics/yolov3) |
|[yolov5](./yolov5)| yolov5 v1.0-v7.0 of [ultralytics/yolov5](https://github.com/ultralytics/yolov5), detection, classification and instance segmentation |
|[yolov7](./yolov7)| yolov7 v0.1, pytorch implementation from [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) |
|[yolov8](./yolov8)| yolov8, pytorch implementation from [ultralytics](https://github.com/ultralytics/ultralytics) |
|[yolov9](./yolov9)| The Pytorch implementation is [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9). |
|[yolov10](./yolov10)| The Pytorch implementation is [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10). |
|[yolo11](./yolo11)| The Pytorch implementation is [ultralytics](https://github.com/ultralytics/ultralytics). |
|[yolop](./yolop)| yolop, pytorch implementation from [hustvl/YOLOP](https://github.com/hustvl/YOLOP) |
|[retinaface](./retinaface)| resnet50 and mobilnet0.25, weights from [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) |
|[arcface](./arcface)| LResNet50E-IR, LResNet100E-IR and MobileFaceNet, weights from [deepinsight/insightface](https://github.com/deepinsight/insightface) |
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
|[refinedet](./refinedet)| RefineDet, pytorch implementation from [luuuyi/RefineDet.PyTorch](https://github.com/luuuyi/RefineDet.PyTorch) |
|[densenet](./densenet)| DenseNet-121, from torchvision.models |
|[rcnn](./rcnn)| FasterRCNN and MaskRCNN, model from [detectron2](https://github.com/facebookresearch/detectron2) |
|[tsm](./tsm)| TSM: Temporal Shift Module for Efficient Video Understanding, ICCV2019 |
|[scaled-yolov4](./scaled-yolov4)| yolov4-csp, pytorch from [WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4) |
|[centernet](./centernet)| CenterNet DLA-34, pytorch from [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet) |
|[efficientnet](./efficientnet)| EfficientNet b0-b8 and l2, pytorch from [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) |
|[detr](./detr)| DE⫶TR, pytorch from [facebookresearch/detr](https://github.com/facebookresearch/detr) |
|[swin-transformer](./swin-transformer)| Swin Transformer - Semantic Segmentation, only support Swin-T. The Pytorch implementation is [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer.git) |
|[real-esrgan](./real-esrgan)| Real-ESRGAN. The Pytorch implementation is [real-esrgan](https://github.com/xinntao/Real-ESRGAN) |
|[superpoint](./superpoint)| SuperPoint. The Pytorch model is from [magicleap/SuperPointPretrainedNetwork](https://github.com/magicleap/SuperPointPretrainedNetwork) |
|[csrnet](./csrnet)| CSRNet. The Pytorch implementation is [leeyeehoo/CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch) |
|[EfficientAd](./efficient_ad)| EfficientAd: Accurate Visual Anomaly Detection at Millisecond-Level Latencies. From [anomalib](https://github.com/openvinotoolkit/anomalib) |

## Model Zoo

The .wts files can be downloaded from model zoo for quick evaluation. But it is recommended to convert .wts from pytorch/mxnet/tensorflow model, so that you can retrain your own model.

[GoogleDrive](https://drive.google.com/drive/folders/1Ri0IDa5OChtcA3zjqRTW57uG6TnfN4Do?usp=sharing) | [BaiduPan](https://pan.baidu.com/s/19s6hO8esU7-TtZEXN7G3OA) pwd: uvv2

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
