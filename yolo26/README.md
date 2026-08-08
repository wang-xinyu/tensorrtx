## Introduction

Yolo26 model supports TensorRT-8.

Training code [link](https://github.com/ultralytics/ultralytics/archive/refs/tags/v8.4.0.zip)

## Environment

* cuda 12.4
* cudnn 9.1.0.70
* tensorrt 8.6.3
* opencv 4.8.0
* ultralytics 8.4.0

## Support

* [✅] Yolo26n-det, Yolo26s-det, Yolo26m-det, Yolo26l-det, Yolo26sx-det, support FP32/FP16 and C++ API
* [✅] Yolo26n-obb, Yolo26s-obb, Yolo26m-obb, Yolo26l-obb, Yolo26sx-obb, support FP32/FP16 and C++ API
* [✅] Yolo26n-cls, Yolo26s-cls, Yolo26m-cls, Yolo26l-cls, Yolo26sx-cls, support FP32/FP16 and C++ API
* [✅] Yolo26n-pose, Yolo26s-pose, Yolo26m-pose, Yolo26l-pose, Yolo26sx-pose, support FP32/FP16 and C++ API
* [✅] Yolo26n-seg, Yolo26s-seg, Yolo26m-seg, Yolo26l-seg, Yolo26sx-seg, support FP32/FP16 and C++ API

## COMING FEATURES
* [⏳] Windows OS Support
* [⏳] Support Quantization

## Config

* Choose the YOLO26 sub-model n/s/m/l/x from command line arguments.
* Other configs please check [include/config.h](include/config.h)

## Build and Run

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```shell
# Download ultralytics
wget https://github.com/ultralytics/ultralytics/archive/refs/tags/v8.4.4.zip -O ultralytics-8.4.4.zip
# Unzip ultralytics
unzip ultralytics-8.4.4.zip
cd ultralytics-8.4.4
# Download models For Detection
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt -O yolo26n.pt # to download other models, replace 'yolo26n.pt' with 'yolo26s.pt', 'yolo26m.pt', 'yolo26l.pt' or 'yolo26x.pt'
# Generate .wts
cp [PATH-TO-MAIN-FOLDER]/gen_wts.py .
python gen_wts.py -w yolo26n.pt -o yolo26n.wts -t detect
# A file 'yolo26n.wts' will be generated.

# Download models for Obb
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.pt -O yolo26n-obb.pt # to download other models, replace 'yolo26n-obb.pt' with 'yolo26s-obb.pt', 'yolo26m-obb.pt', 'yolo26l-obb.pt' or 'yolo26x-obb.pt'
# Generate .wts
cp [PATH-TO-MAIN-FOLDER]/gen_wts.py .
python gen_wts.py -w yolo26n-obb.pt -o yolo26n-obb.wts -t obb
# A file 'yolo26n-obb.wts' will be generated.

# Download models for Cls
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-cls.pt -O yolo26n-cls.pt # to download other models, replace 'yolo26n-cls.pt' with 'yolo26s-cls.pt', 'yolo26m-cls.pt', 'yolo26l-cls.pt' or 'yolo26x-cls.pt'
# Generate .wts
cp [PATH-TO-MAIN-FOLDER]/gen_wts.py .
python gen_wts.py -w yolo26n-cls.pt -o yolo26n-cls.wts -t cls
# A file 'yolo26n-cls.wts' will be generated.

# Download models for Pose
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt -O yolo26n-pose.pt # to download other models, replace 'yolo26n-pose.pt' with 'yolo26s-pose.pt', 'yolo26m-pose.pt', 'yolo26l-pose.pt' or 'yolo26x-pose.pt'
# Generate .wts
cp [PATH-TO-MAIN-FOLDER]/gen_wts.py .
python gen_wts.py -w yolo26n-pose.pt -o yolo26n-pose.wts -t pose
# A file 'yolo26n-pose.wts' will be generated.

# Download models for Seg
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt -O yolo26n-seg.pt # to download other models, replace 'yolo26n-seg.pt' with 'yolo26s-seg.pt', 'yolo26m-seg.pt', 'yolo26l-seg.pt' or 'yolo26x-seg.pt'
# Generate .wts
cp [PATH-TO-MAIN-FOLDER]/gen_wts.py .
python gen_wts.py -w yolo26n-seg.pt -o yolo26n-seg.wts -t seg
# A file 'yolo26n-seg.wts' will be generated.

```

2. build and run
```shell
cd [PATH-TO-MAIN-FOLDER]
mkdir build
cd build
cmake ..
make
```

### Detection
```shell
cp [PATH-TO-ultralytics]/yolo26n.wts .
# Build and serialize TensorRT engine
./yolo26_det -s yolo26n.wts yolo26n.engine [n/s/m/l/x]
# Run inference
./yolo26_det -d yolo26n.engine ../images
# results saved in build directory
```

### Obb
```shell
cp [PATH-TO-ultralytics]/yolo26n-obb.wts .
# Build and serialize TensorRT engine
./yolo26_obb -s yolo26n-obb.wts yolo26n-obb.engine [n/s/m/l/x]
# Run inference
./yolo26_obb -d yolo26n-obb.engine ../images
# results saved in build directory
```

### Cls
```shell
Generate classification text file in build folder or download it
# wget https://github.com/joannzhang00/ImageNet-dataset-classes-labels/blob/main/imagenet_classes.txt

cp [PATH-TO-ultralytics]/yolo26n-cls.wts .
# Build and serialize TensorRT engine
./yolo26_cls -s yolo26n-cls.wts yolo26n-cls.engine [n/s/m/l/x]
# Run inference
./yolo26_cls -d yolo26n-cls.engine ../images
# results saved in build directory
```

### Pose
```shell
cp [PATH-TO-ultralytics]/yolo26n-pose.wts .
# Build and serialize TensorRT engine
./yolo26_pose -s yolo26n-pose.wts yolo26n-pose.engine [n/s/m/l/x]
# Run inference
./yolo26_pose -d yolo26n-pose.engine ../images
# results saved in build directory
```

### Seg
```shell
Generate coco text file in build folder or download it

cp [PATH-TO-ultralytics]/yolo26n-seg.wts .
# Build and serialize TensorRT engine
./yolo26_seg -s yolo26n-seg.wts yolo26n-seg.engine [n/s/m/l/x]
# Run inference
./yolo26_seg -d yolo26n-seg.engine ../images
# results saved in build directory
```

## More Information
See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)