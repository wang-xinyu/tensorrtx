# YOLOv7

The Pytorch implementation is [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7).

The tensorrt code is derived from [QIANXUNZDL123/tensorrtx-yolov7](https://github.com/QIANXUNZDL123/tensorrtx-yolov7)

## Contributors

<a href="https://github.com/QIANXUNZDL123"><img src="https://avatars.githubusercontent.com/u/46549527?v=4?s=48" width="40px;" alt=""/></a>
<a href="https://github.com/lindsayshuo"><img src="https://avatars.githubusercontent.com/u/45239466?v=4?s=48" width="40px;" alt=""/></a>
<a href="https://github.com/wang-xinyu"><img src="https://avatars.githubusercontent.com/u/15235574?s=48&v=4" width="40px;" alt=""/></a> 

## Requirements

- TensorRT 8.0+
- OpenCV 3.4.0+

## Different versions of yolov7

Currently, we support yolov7 v0.1

- For yolov7 v0.1, download .pt from [yolov7 release v0.1](https://github.com/WongKinYiu/yolov7/releases/tag/v0.10), then follow how-to-run in current page.

## Config

- Choose the model tiny/v7/x/d6/w6/e6/e6e from command line arguments.
- Check more configs in [include/config.h](./include/config.h)

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
// update kNumClass in config.h if your model is trained on custom dataset
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

3. set the macro `USE_INT8` in config.h and make

4. serialize the model and test

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247927-4d9fac00-751e-11ea-8b1b-704a0aeb3fcf.jpg" height="360px;">
</p>

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

