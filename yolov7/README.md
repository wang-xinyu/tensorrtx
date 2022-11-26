# yolov7

The Pytorch implementation is [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7).

The tensorrt code is derived from [QIANXUNZDL123/tensorrtx-yolov7](https://github.com/QIANXUNZDL123/tensorrtx-yolov7)

## Different versions of yolov7

Currently, we support yolov7 v0.1

- For yolov7 v0.1, download .pt from [yolov7 release v0.1](https://github.com/WongKinYiu/yolov7/releases/tag/v0.10), then follow how-to-run in current page.


## Config

- Choose the model tiny/v7/x/d6/w6/e6/e6e from command line arguments.
- Input shape defined in yololayer.h
- Number of classes defined in yololayer.h, **DO NOT FORGET TO ADAPT THIS, If using your own model**
- INT8/FP16/FP32 can be selected by the macro in yolov7.cpp, **INT8 need more steps, pls follow `How to Run` first and then go the `INT8 Quantization` below**
- GPU id can be selected by the macro in yolov7.cpp
- NMS thresh in yolov7.cpp
- BBox confidence thresh in yolov7.cpp
- Batch size in yolov7.cpp

## How to Run, yolov7-tiny as example

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```
// download https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
cp {tensorrtx}/yolov7/gen_wts.py {WongKinYiu}/yolov7
cd {WongKinYiu}/yolov7
python gen_wts.py
// a file 'yolov7.wts' will be generated.
```

2. build tensorrtx/yolov7 and run

```
cd {tensorrtx}/yolov7/
// update CLASS_NUM in yololayer.h if your model is trained on custom dataset
mkdir build
cd build
cp {WongKinYiu}/yolov7/yolov7.wts {tensorrtx}/yolov7/build
cmake ..
make
sudo ./yolov7 -s [.wts] [.engine] [t/v7/x/w6/e6/d6/e6e gd gw]  // serialize model to plan file
sudo ./yolov7 -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.
// For example yolov7
sudo ./yolov7 -s yolov7.wts yolov7.engine v7
sudo ./yolov7 -d yolov7.engine ../images
```

3. check the images generated, as follows. _zidane.jpg and _bus.jpg

4. optional, load and run the tensorrt model in python

```
// install python-tensorrt, pycuda, etc.
// ensure the yolov7.engine and libmyplugins.so have been built
python yolov7_trt.py
```

# INT8 Quantization

1. Prepare calibration images, you can randomly select 1000s images from your train set. For coco, you can also download my calibration images `coco_calib` from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh

2. unzip it in yolov7/build

3. set the macro `USE_INT8` in yolov7.cpp and make

4. serialize the model and test

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247927-4d9fac00-751e-11ea-8b1b-704a0aeb3fcf.jpg">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247970-60b27c00-751e-11ea-88df-41473fed4823.jpg">
</p>

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)
