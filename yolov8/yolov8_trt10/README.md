## Introduce

Yolov8 model supports TensorRT-10.

## Environment

CUDA: 11.8
CUDNN: 8.9.1.23
TensorRT: TensorRT-10.2.0.19

## Support

* [x] YOLOv8-cls support FP32/FP16/INT8 and Python/C++ API
* [x] YOLOv8-det support FP32/FP16/INT8 and Python/C++ API
* [x] YOLOv8-seg support FP32/FP16/INT8 and Python/C++ API
* [x] YOLOv8-pose support FP32/FP16/INT8 and Python/C++ API

## Config

* Choose the YOLOv8 sub-model n/s/m/l/x/n6/s6/m6/l6/x6 from command line arguments.
* Other configs please check [src/config.h](src/config.h)

## Build and Run

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```shell
git clone https://gitclone.com/github.com/ultralytics/ultralytics.git
git clone -b trt10 https://github.com/wang-xinyu/tensorrtx.git
cd yolov8/
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-cls.pt
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt
cp [PATH-TO-TENSORRTX]/yolov8/gen_wts.py .
python gen_wts.py -w yolov8n-cls.pt -o yolov8n-cls.wts -t cls
python gen_wts.py -w yolov8n.pt -o yolov8n.wts
python gen_wts.py -w yolov8n-seg.pt -o yolov8n-seg.wts -t seg
python gen_wts.py -w yolov8n-pose.pt -o yolov8n-pose.wts -t pose
# A file 'yolov8n.wts' will be generated.
```

2. build tensorrtx/yolov8/yolov8_trt10 and run

#### Classification

```shell
cd [PATH-TO-TENSORRTX]/yolov8/yolov8_trt10
# Update kNumClass in src/config.h if your model is trained on custom dataset
mkdir build
cd build
cp [PATH-TO-ultralytics-yolov8]/yolov8sn-cls.wts .
cmake ..
make

# Download ImageNet labels
wget https://github.com/joannzhang00/ImageNet-dataset-classes-labels/blob/main/imagenet_classes.txt

# Build and serialize TensorRT engine
./yolov8_cls -s yolov8n-cls.wts yolov8n-cls.engine [n/s/m/l/x]

# Run inference
./yolov8_cls -d yolov8n-cls.engine ../images
# The results are displayed in the console
```

3. Optional, load and run the tensorrt model in Python
```shell
// Install python-tensorrt, pycuda, etc.
// Ensure the yolov8n-cls.engine
python yolov8_cls_trt.py ./build/yolov8n-cls.engine ../images
# faq: in windows bug pycuda._driver.LogicError
# faq: in linux bug Segmentation fault
# Add the following code to the py file:
# import pycuda.autoinit
# import pycuda.driver as cuda
```

#### Detection

```shell
cd [PATH-TO-TENSORRTX]/yolov8/yolov8_trt10
# Update kNumClass in src/config.h if your model is trained on custom dataset
mkdir build
cd build
cp [PATH-TO-ultralytics-yolov8]/yolov8n.wts .
cmake ..
make

# Build and serialize TensorRT engine
./yolov8_det -s yolov8n.wts yolov8n.engine [n/s/m/l/x]

# Run inference
./yolov8_det -d yolov8n.engine ../images [c/g]
# The results are displayed in the console
```

#### Segmentation

```shell
cd [PATH-TO-TENSORRTX]/yolov8/yolov8_trt10
# Update kNumClass in src/config.h if your model is trained on custom dataset
mkdir build
cd build
cp [PATH-TO-ultralytics-yolov8]/yolov8n-seg.wts .
cmake ..
make

# Build and serialize TensorRT engine
./yolov8_seg -s yolov8n-seg.wts yolov8n-seg.engine [n/s/m/l/x]

# Download the labels file
wget -O coco.txt https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt

# Run inference
./yolov8_seg -d yolov8n-seg.engine ../images [c/g] coco.txt
# The results are displayed in the console
```

#### Pose

```shell
cd [PATH-TO-TENSORRTX]/yolov8/yolov8_trt10
# Update kNumClass in src/config.h if your model is trained on custom dataset
mkdir build
cd build
cp [PATH-TO-ultralytics-yolov8]/yolov8n-pose.wts .
cmake ..
make

# Build and serialize TensorRT engine
./yolov8_seg -s yolov8n-pose.wts yolov8n-pose.engine [n/s/m/l/x]

# Run inference
./yolov8_seg -d yolov8n-seg.engine ../images c
# The results are displayed in the console
```

## INT8 Quantization
1. Prepare calibration images, you can randomly select 1000s images from your train set. For coco, you can also download my calibration images `coco_calib` from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh
2. unzip it in yolov8_trt10/build
3. set the macro `USE_INT8` in src/config.h and make again
4. serialize the model and test

## More Information
See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)
