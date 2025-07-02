## Introduction

Yolo11 model supports TensorRT-7 (Jetson Nano).

Training code [link](https://github.com/ultralytics/ultralytics/archive/refs/tags/v8.3.38.zip)

## Environment

* cuda 10.2
* cudnn 8
* tensorrt 7.1.3.0
* opencv 4.5.3

## Support

* [x] YOLO11-det support FP32/FP16 and Python/C++ API
* [x] YOLO11-cls support FP32/FP16 and Python/C++ API
* [x] YOLO11-seg support FP32/FP16 and Python/C++ API
* [x] YOLO11-pose support FP32/FP16 and Python/C++ API
* [x] YOLO11-obb support FP32/FP16 and Python/C++ API

Note:
Jetson Nano doesn't have any tensor cores, so it doesn't support INT8 quantization

## Config

* Choose the YOLO11 sub-model n/s/m/l/x from command line arguments.
* Other configs please check [src/config.h](src/config.h)

## Build and Run

1. generate .wts files.
Since ultralytics package requires python >= 3.8 and Jetson Nano is tied to python 3.6, use the provided `colab_export_wts.ipynb` Google Colab notebook for exporting the weights to the .wts format.

The notebook will download the compressed wts files as `yolo11-wts.tar.gz`.

2. After downloading, decompress and place the wts weights files into the yolo11-jetson-nano directory:
```shell

tar xvf yolo11-wts.tar.gz
cp *.wts [PATH-TO-TENSORRTX]/yolo11-jetson-nano


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
# Build and serialize TensorRT engine
./yolo11_det -s ../yolo11n.wts yolo11n.engine [n/s/m/l/x]
# Run inference
./yolo11_det -d yolo11n.engine ../images [c/g]
# results saved in build directory
```

### Classification
```shell

# Build and serialize TensorRT engine
./yolo11_cls -s ../yolo11n-cls.wts yolo11n-cls.engine [n/s/m/l/x]
# Download ImageNet labels
wget https://github.com/joannzhang00/ImageNet-dataset-classes-labels/blob/main/imagenet_classes.txt
# Run inference
./yolo11_cls -d yolo11n-cls.engine ../images
```

### Segmentation
```shell

# Build and serialize TensorRT engine
./yolo11_seg -s ../yolo11n-seg.wts yolo11n-seg.engine [n/s/m/l/x]
# Download the labels file
wget -O coco.txt https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt
# Run inference
./yolo11_seg -d yolo11n-seg.engine ../images c coco.txt
```

### Pose
```shell

# Build and serialize TensorRT engine
./yolo11_pose -s ../yolo11n-pose.wts yolo11n-pose.engine [n/s/m/l/x]
# Run inference
./yolo11_pose -d yolo11n-pose.engine ../images
```

### Obb
```shell

# Build and serialize TensorRT engine
./yolo11_obb -s ../yolo11n-obb.wts yolo11n-obb.engine [n/s/m/l/x]
# Download the image
wget -O P0015.png https://github.com/mpj1234/YOLO11-series-TensorRT8/releases/download/images/P0015.png
mv P0015.png ../images
# Run inference
./yolo11_obb -d yolo11n-obb.engine ../images
```


## INT8 Quantization
Jetson Nano doesn't support INT8 Quantization.
