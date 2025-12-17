## Introduction

Yolov13 model supports TensorRT-8.

Detection training code [link](https://github.com/iMoonLab/yolov13/releases/tag/yolov13) (recommended)


## Environment

* cuda 11.6
* cudnn 8.9.1.23
* tensorrt 8.6.1.6
* opencv 4.8.0
* ultralytics 8.3.63

## Support

* [x] YOLOV13-det support FP32/FP16/INT8 and C++ API


## Config

* Choose the YOLOV13 sub-model n/s/l/x from command line arguments.
* Other configs please check [src/config.h](src/config.h)

## Build and Run (Detection)

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```shell
# Acquire ultralytics
# Recommended: download the appropriate asset from the ultralytics releases, or install/clone via pip/git. Examples:
# pip install ultralytics==8.3.63
# or: git clone https://github.com/ultralytics/ultralytics.git && cd ultralytics
# To train your own models, download a model .pt and convert it to .wts
# To download other models, replace 'yolov13n.pt' with 'yolov13s.pt', 'yolov13l.pt', or 'yolov13x.pt'
# Generate .wts
# Run gen_wts.py from the yolov13 directory, for example:
# cp [PATH-TO-TENSORRTX]/yolov13/gen_wts.py .
# python gen_wts.py -w yolov13n.pt -t detect -o yolov13n.wts
# A file 'yolov13n.wts' will be generated.
```

2. build tensorrtx/yolov13 and run
```shell
cd [PATH-TO-TENSORRTX]/yolov13
mkdir build
cd build
cmake ..
make
```



### Detection
```shell
cp [PATH-TO-ultralytics]/yolov13n.wts .
# Build and serialize TensorRT engine
./yolov13-det -s yolov13n.wts yolov13n.engine [n/s/l/x]
# Run inference
./yolov13-det -d yolov13n.engine ../images [c/g]
# results saved in build directory
```

# INT8 Quantization

1. Prepare calibration images, you can randomly select 1000s images from your train set.
     For coco, you can also download my calibration images `coco_calib` from
     [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing)
     or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh
2. unzip it in yolo11/build
3. set the macro `USE_INT8` in src/config.h and make again
4. serialize the model and test

### Personal report: RTX 3050 Ti - failure
### Personal report: RTX 4060 - success




















## More Information
See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)
