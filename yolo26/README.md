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

* [✅] Yolo26n-det support FP32/FP16 and C++ API

## COMING FEATURES
* [⏳] Yolo26s-det, Yolo26m-det, Yolo26l-det, Yolo26sx-det
* [⏳] Windows OS Support
* [⏳] Ssupport Batched Inputs
* [⏳] Support Quantization
* [⏳] Yolo26-cls models
* [⏳] Yolo26-pose models
* [⏳] Yolo26-seg models
* [⏳] Yolo26-obb models

## Config

* Choose the YOLO26 sub-model n/s/m/l/x from command line arguments. (For now only tested n model!)
* Other configs please check [src/config.h](src/config.h)

## Build and Run

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```shell
# Download ultralytics
wget https://github.com/ultralytics/ultralytics/archive/refs/tags/v8.4.0.zip -O ultralytics-8.4.0.zip
# Unzip ultralytics
unzip ultralytics-8.4.0.zip
cd ultralytics-8.4.0
# Download models
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt -O yolo26n.pt # to download other models, replace 'yolo26n.pt' with 'yolo26s.pt', 'yolo26m.pt', 'yolo26l.pt' or 'yolo26x.pt'
# Generate .wts
cp [PATH-TO-MAIN-FOLDER]/gen_wts.py .
python gen_wts.py -w yolo26n.pt -o yolo26n.wts -t detect
# A file 'yolo26n.wts' will be generated.
```

2. build tensorrt_yolo26 and run
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
./yolo26_det -d yolo26n.engine ../images [c/g]
# results saved in build directory
```

## More Information
See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)