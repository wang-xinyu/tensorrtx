## Introduce

Yolov5 model supports TensorRT-10.

## Environment

CUDA: 11.8
CUDNN: 8.9.1.23
TensorRT: TensorRT-10.2.0.19

## Support

* [x] YOLOv5-cls support FP32/FP16/INT8 and Python/C++ API
* [x] YOLOv5-det support FP32/FP16/INT8 and Python/C++ API
* [x] YOLOv5-seg support FP32/FP16/INT8 and Python/C++ API

## Config

* Choose the YOLOv5 sub-model n/s/m/l/x/n6/s6/m6/l6/x6 from command line arguments.
* Other configs please check [src/config.h](src/config.h)

## Build and Run

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```shell
git clone -b v7.0 https://github.com/ultralytics/yolov5.git
git clone -b trt10 https://github.com/wang-xinyu/tensorrtx.git
cd yolov5/
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-cls.pt
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-seg.pt
cp [PATH-TO-TENSORRTX]/yolov5/gen_wts.py .
python gen_wts.py -w yolov5n-cls.pt -o yolov5n-cls.wts -t cls
python gen_wts.py -w yolov5n.pt -o yolov5n.wts
python gen_wts.py -w yolov5n-seg.pt -o yolov5n.wts -t seg
# A file 'yolov5n.wts' will be generated.
```

2. build tensorrtx/yolov5/yolov5_trt10 and run

#### Classification

```shell
cd [PATH-TO-TENSORRTX]/yolov5/yolov5_trt10
# Update kNumClass in src/config.h if your model is trained on custom dataset
mkdir build
cd build
cp [PATH-TO-ultralytics-yolov5]/yolov5sn-cls.wts .
cmake ..
make

# Download ImageNet labels
wget https://github.com/joannzhang00/ImageNet-dataset-classes-labels/blob/main/imagenet_classes.txt

# Build and serialize TensorRT engine
./yolov5_cls -s yolov5n-cls.wts yolov5n-cls.engine [n/s/m/l/x]

# Run inference
./yolov5_cls -d yolov5n-cls.engine ../../images
# The results are displayed in the console
```

3. Optional, load and run the tensorrt model in Python
```shell
// Install python-tensorrt, pycuda, etc.
// Ensure the yolov5n-cls.engine
python yolov5_cls_trt.py
# faq: in windows bug pycuda._driver.LogicError
# faq: in linux bug Segmentation fault
# Add the following code to the py file:
# import pycuda.autoinit
# import pycuda.driver as cuda
```

#### Detection

```shell
cd [PATH-TO-TENSORRTX]/yolov5/yolov5_trt10
# Update kNumClass in src/config.h if your model is trained on custom dataset
mkdir build
cd build
cp [PATH-TO-ultralytics-yolov5]/yolov5n.wts .
cmake ..
make

# Build and serialize TensorRT engine
./yolov5_det -s yolov5n.wts yolov5n.engine [n/s/m/l/x]

# Run inference
./yolov5_det -d yolov5n.engine ../../images
# The results are displayed in the console
```

#### Segmentation

```shell
cd [PATH-TO-TENSORRTX]/yolov5/yolov5_trt10
# Update kNumClass in src/config.h if your model is trained on custom dataset
mkdir build
cd build
cp [PATH-TO-ultralytics-yolov5]/yolov5n-seg.wts .
cmake ..
make

# Build and serialize TensorRT engine
./yolov5_seg -s yolov5n-seg.wts yolov5n-seg.engine [n/s/m/l/x]

# Download the labels file
wget -O coco.txt https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt

# Run inference
./yolov5_seg -d yolov5n-seg.engine ../../images coco.txt
# The results are displayed in the console
```

## INT8 Quantization
1. Prepare calibration images, you can randomly select 1000s images from your train set. For coco, you can also download my calibration images `coco_calib` from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh
2. unzip it in yolov5_trt10/build
3. set the macro `USE_INT8` in src/config.h and make again
4. serialize the model and test

## More Information
See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)
