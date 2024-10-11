## Introduction

Yolo11 model supports TensorRT-8.

Training code [ultralytics v8.3.0](https://github.com/ultralytics/ultralytics/tree/v8.3.0)

## Environment

* cuda 11.8
* cudnn 8.9.1.23
* tensorrt 8.6.1.6
* opencv 4.8.0
* ultralytics 8.3.0

## Support

* [x] YOLO11-det support FP32/FP16/INT8 and Python/C++ API
* [x] YOLO11-cls support FP32/FP16/INT8 and Python/C++ API
* [x] YOLO11-seg support FP32/FP16/INT8 and Python/C++ API
* [x] YOLO11-pose support FP32/FP16/INT8 and Python/C++ API

## Config

* Choose the YOLO11 sub-model n/s/m/l/x from command line arguments.
* Other configs please check [include/config.h](include/config.h)

## Build and Run

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```shell
# Download ultralytics
wget https://github.com/ultralytics/ultralytics/archive/refs/tags/v8.3.0.zip -O ultralytics-8.3.0.zip
# Unzip ultralytics
unzip ultralytics-8.3.0.zip
cd ultralytics-8.3.0
# Download models
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -O yolo11n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt -O yolo11n-cls.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt -O yolo11n-seg.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt -O yolo11n-pose.pt
# Generate .wts
cp [PATH-TO-TENSORRTX]/yolo11/gen_wts.py .
python gen_wts.py -w yolo11n.pt -o yolo11n.wts -t detect
python gen_wts.py -w yolo11n-cls.pt -o yolo11n-cls.wts -t cls
python gen_wts.py -w yolo11n-seg.pt -o yolo11n-seg.wts -t seg
python gen_wts.py -w yolo11n-pose.pt -o yolo11n-pose.wts -t pose
# A file 'yolo11n.wts' will be generated.
```

2. build tensorrtx/yolo11 and run
```shell
cd [PATH-TO-TENSORRTX]/yolo11
mkdir build
cd build
cmake ..
make
```

### Detection
```shell
cp [PATH-TO-ultralytics]/yolo11n.wts .
# Build and serialize TensorRT engine
./yolo11_det -s yolo11n.wts yolo11n.engine [n/s/m/l/x]
# Run inference
./yolo11_det -d yolo11n.engine ../images [c/g]
# results saved in build directory
```

### Classification
```shell
cp [PATH-TO-ultralytics]/yolo11n-cls.wts .
# Build and serialize TensorRT engine
./yolo11_cls -s yolo11n-cls.wts yolo11n-cls.engine [n/s/m/l/x]
# Download ImageNet labels
wget https://github.com/joannzhang00/ImageNet-dataset-classes-labels/blob/main/imagenet_classes.txt
# Run inference
./yolo11_cls -d yolo11n-cls.engine ../images
```

### Segmentation
```shell
cp [PATH-TO-ultralytics]/yolo11n-seg.wts .
# Build and serialize TensorRT engine
./yolo11_seg -s yolo11n-seg.wts yolo11n-seg.engine [n/s/m/l/x]
# Download the labels file
wget -O coco.txt https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt
# Run inference
./yolo11_seg -d yolo11n-seg.engine ../images c coco.txt
```

### Pose
```shell
cp [PATH-TO-ultralytics]/yolo11n-pose.wts .
# Build and serialize TensorRT engine
./yolo11_pose -s yolo11n-pose.wts yolo11n-pose.engine [n/s/m/l/x]
# Run inference
./yolo11_pose -d yolo11n-pose.engine ../images
```

3. Optional, load and run the tensorrt model in Python
```shell
// Install python-tensorrt, pycuda, etc.
// Ensure the yolo11n.engine
python yolo11_det_trt.py ./build/yolo11n.engine ./build/libmyplugins.so
# faq: in windows bug pycuda._driver.LogicError
# faq: in linux bug Segmentation fault
# Add the following code to the py file:
# import pycuda.autoinit
# import pycuda.driver as cuda
```

## INT8 Quantization
1. Prepare calibration images, you can randomly select 1000s images from your train set. For coco, you can also download my calibration images `coco_calib` from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh
2. unzip it in yolo11/build
3. set the macro `USE_INT8` in src/config.h and make again
4. serialize the model and test

## More Information
See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)
