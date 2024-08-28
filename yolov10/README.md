## Introduce

Yolov10 model supports TensorRT-8.

## Environment

CUDA: 11.8

CUDNN: 8.9.1.23

TensorRT: TensorRT-8.2.5.1   / GPU: RTX1650

TensorRT: TensorRT-8.4.3.1   / GPU: RTX4070

```
# faq
Error Code 1: Internal Error (Unsupported SM: 0x809)
The architecture of the higher version does not support the use of the earlier version of TensorRT,
and you need to upgrade the TensorRT version
```

## Support

* [x] YOLOv10-det support FP32/FP16/INT8 and Python/C++ API

## Config

* Choose the YOLOv10 sub-model n/s/m/b/l/x from command line arguments.
* Other configs please check [src/config.h](src/config.h)

## Build and Run

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```shell
git clone https://github.com/THU-MIG/yolov10.git
cd yolov10/
wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt

git clone https://github.com/wang-xinyu/tensorrtx.git
cp [PATH-TO-TENSORRTX]/yolov10/gen_wts.py .

python gen_wts.py -w yolov10n.pt -o yolov10n.wts
# A file 'yolov10n.wts' will be generated.
```

2. build tensorrtx/yolov10 and run

#### Detection

```shell
cd [PATH-TO-TENSORRTX]/yolov10

# add test images
mkdir images
cp [PATH-TO-TENSORRTX]/yolov3-spp/samples/*.jpg ./images

# Update kNumClass in src/config.h if your model is trained on custom dataset
mkdir build
cd build
cp [PATH-TO-yolov10]/yolov10n.wts .
cmake ..
make

# Build and serialize TensorRT engine
./yolov10_det -s yolov10n.wts yolov10n.engine [n/s/m/b/l/x]

# Run inference
./yolov10_det -d yolov10n.engine ../images
# The results are displayed in the console
```

3. Optional, load and run the tensorrt model in Python
```shell
// Install python-tensorrt, pycuda, etc.
// Ensure the yolov10n.engine
python yolov10_det_trt.py ./build/yolov10n.engine ./build/libmyplugins.so
```

## INT8 Quantization
1. Prepare calibration images, you can randomly select 1000s images from your train set. For coco, you can also download my calibration images `coco_calib` from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh
2. unzip it in yolov10/build
3. set the macro `USE_INT8` in src/config.h and make again
4. serialize the model and test

## More Information
See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)
