## Introduction

Yolov12 model supports TensorRT-8.

Detection training code [link](https://github.com/sunsmarterjie/yolov12/releases/tag/turbo)【建议】
Segment training code[link](https://github.com/sunsmarterjie/yolov12/releases/tag/seg)【建议】
Classify training code[link](https://github.com/sunsmarterjie/yolov12/releases/tag/cls)【建议】

## Environment

* cuda 11.6
* cudnn 8.9.1.23
* tensorrt 8.6.1.6
* opencv 4.8.0
* ultralytics 8.3.63

## Support

* [x] YOLOv12-det support FP32/FP16 and C++ API
* [x] YOLOv12-seg support FP32/FP16 and C++ API
* [x] YOLOv12-cls support FP32/FP16 and C++ API


## Config

* Choose the YOLOv12 sub-model n/s/m/l/x from command line arguments.
* Other configs please check [src/config.h](src/config.h)

## Build and Run (Detection)

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```shell
# 获取 ultralytics
# 推荐方式：要么从 ultralytics 的 releases 下载对应的 asset，要么使用 pip 或 git clone。示例：
# pip install ultralytics==8.3.63
# 或者：git clone https://github.com/ultralytics/ultralytics.git && cd ultralytics
# Training your ownself models
to download other models, replace 'yolov12n.pt' with 'yolov12s.pt', 'yolov12m.pt', 'yolov12l.pt' or 'yolov12x.pt'
# Generate .wts
# 在 yolov12 目录中运行 gen_wts.py，例如：
# cp [PATH-TO-TENSORRTX]/yolov12/gen_wts.py .
# python gen_wts.py -w yolov12n.pt -o yolov12n.wts -t detect
# A file 'yolov12n.wts' will be generated.
```

2. build tensorrtx/yolov12 and run
```shell
cd [PATH-TO-TENSORRTX]/yolov12
mkdir build
cd build
cmake ..
make
```



## Build and Run (Segment)

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```shell
# 获取 ultralytics（同上）
# pip install ultralytics==8.3.63
# 或者：git clone https://github.com/ultralytics/ultralytics.git && cd ultralytics
# Training your ownself models
to download other models, replace 'yolov12n-seg.pt' with 'yolov12s-seg.pt', 'yolov12m-seg.pt', 'yolov12l-seg.pt' or 'yolov12x-seg.pt'
# Generate .wts
# 在 yolov12 目录中运行 gen_wts.py，例如：
# cp [PATH-TO-TENSORRTX]/yolov12/gen_wts.py .
# python gen_wts.py -w yolov12n-seg.pt -o yolov12n-seg.wts -t detect
# A file 'yolov12n-seg.wts' will be generated.
```

2. build tensorrtx/yolov12 and run
```shell
cd [PATH-TO-TENSORRTX]/yolov12
mkdir build
cd build
cmake ..
make
```

## Build and Run (Classify)

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```shell
# 获取 ultralytics（同上）
# pip install ultralytics==8.3.63
# 或者：git clone https://github.com/ultralytics/ultralytics.git && cd ultralytics
# Training your ownself models
# to download other models, replace 'yolov12n-cls.pt' with 'yolov12s-cls.pt', 'yolov12m-cls.pt', 'yolov12l-cls.pt' or 'yolov12x-cls.pt'
# Generate .wts
cp [PATH-TO-TENSORRTX]/yolov12/gen_wts.py .
python gen_wts.py -w yolo12n-cls.pt -t cls -o yolo12n-cls.wts 
# A file 'yolo12n-cls.wts' will be generated.
```

2. build tensorrtx/yolov12 and run
```shell
cd [PATH-TO-TENSORRTX]/yolov12
mkdir build
cd build
cmake ..
make
```

### Detection
```shell
# cp [PATH-TO-ultralytics]/yolov12n.wts .
# Build and serialize TensorRT engine
./yolov12-det -s yolov12n.wts yolov12n.engine [n/s/m/l/x]
# Run inference
./yolov12-det -d yolov12n.engine ../images [c/g]
# results saved in build directory
```

### Segment
```shell
# cp [PATH-TO-ultralytics]/yolov12n-seg.wts .
# Build and serialize TensorRT engine
./yolov12-seg -s yolov12n-seg.wts yolov12n-seg.engine [n/s/m/l/x]
# Run inference
./yolov12-seg -d yolov12n-seg.engine ../images
# results saved in build directory
```

### Classify
```shell
# cp [PATH-TO-ultralytics]/yolov12n-cls.wts .
# Build and serialize TensorRT engine
./yolov12-cls -s yolov12n-cls.wts yolov12n-cls.engine [n/s/m/l/x]
# Run inference
./yolov12-cls -d yolov12n-cls.engine ../images 
# results saved in build directory
```
