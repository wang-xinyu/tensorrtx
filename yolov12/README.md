## Introduction

Yolo12 model supports TensorRT-10.

Training code [link](https://github.com/ultralytics/ultralytics/archive/refs/tags/v8.3.38.zip)

## Environment

* cuda 12.6.2
* cudnn 9.5.0
* tensorrt 10.5.0.18
* opencv 4.12.0
* ultralytics 8.3.119

## Support

* [x] YOLO12-det support FP32/FP16 and C++ API


## Config

* Choose the YOLO12 sub-model n/s/m/l/x from command line arguments.
* Other configs please check [include/config.h](include/config.h)

## Build and Run

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```shell
# Download ultralytics
wget https://github.com/ultralytics/ultralytics/archive/refs/tags/v8.3.119.zip -O ultralytics-8.3.119.zip
# Unzip ultralytics
unzip ultralytics-8.3.119.zip
cd ultralytics-8.3.119
# Download models
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt -O yolo12n.pt # to download other models, replace 'yolo12n.pt' with 'yolo12s.pt', 'yolo12m.pt', 'yolo12l.pt' or 'yolo12x.pt'
# Generate .wts
cp [PATH-TO-TENSORRTX]/yolov12/gen_wts.py .
python gen_wts.py -w yolo12n.pt -o yolo12n.wts -t detect
# A file 'yolo12n.wts' will be generated.
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
cp [PATH-TO-ultralytics]/yolo12n.wts .
# Build and serialize TensorRT engine
./yolo12_det -s yolo12n.wts yolo12n.engine [n/s/m/l/x]
# Run inference
./yolo12_det -d yolo12n.engine ../images [c/g]
# results saved in build directory
```

## More Information
See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)
