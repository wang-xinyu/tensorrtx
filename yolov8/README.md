# yolov8

The Pytorch implementation is [ultralytics/yolov8](https://github.com/ultralytics/ultralytics/tree/main/ultralytics).

The tensorrt code is derived from [xiaocao-tian/yolov8_tensorrt](https://github.com/xiaocao-tian/yolov8_tensorrt)

## Contributors

<a href="https://github.com/xiaocao-tian"><img src="https://avatars.githubusercontent.com/u/46549527?v=4?s=48" width="40px;" alt=""/></a>
<a href="https://github.com/lindsayshuo"><img src="https://avatars.githubusercontent.com/u/45239466?v=4?s=48" width="40px;" alt=""/></a>


## Requirements

- TensorRT 8.0+
- OpenCV 3.4.0+

## Different versions of yolov8

Currently, we support yolov8 v0.1

- For yolov8 v0.1, download .pt from [yolov8 release v0.1](https://github.com/WongKinYiu/yolov8/releases/tag/v0.1), then follow how-to-run in current page.

## Config

- Choose the model tiny/v7/x/d6/w6/e6/e6e from command line arguments.
- Check more configs in [include/config.h](./include/config.h)

## How to Run, yolov8-tiny as example

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```
// download https://github.com/WongKinYiu/yolov8/releases/download/v0.1/yolov8-tiny.pt
cp {tensorrtx}/yolov8/gen_wts.py {WongKinYiu}/yolov8
cd { ultralytics}/ ultralytics
python gen_wts.py
// a file 'yolov8.wts' will be generated.
```

2. build tensorrtx/yolov8 and run

```
cd {tensorrtx}/yolov8/
// update kNumClass in config.h if your model is trained on custom dataset
mkdir build
cd build
cp {WongKinYiu}/yolov8/yolov8.wts {tensorrtx}/yolov8/build
cmake ..
make
sudo ./yolov8 -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to plan file
sudo ./yolov8 -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.
// For example yolov8
sudo ./yolov8 -s yolov8.wts yolov8.engine v7
sudo ./yolov8 -d yolov8.engine ../images
```

3. check the images generated, as follows. _zidane.jpg and _bus.jpg

4. optional, load and run the tensorrt model in python

```
// install python-tensorrt, pycuda, etc.
// ensure the yolov8n.engine and libmyplugins.so have been built
python yolov8_trt.py
```

# INT8 Quantization

1. Prepare calibration images, you can randomly select 1000s images from your train set. For coco, you can also download my calibration images `coco_calib` from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh

2. unzip it in yolov8/build

3. set the macro `USE_INT8` in config.h and make

4. serialize the model and test

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247927-4d9fac00-751e-11ea-8b1b-704a0aeb3fcf.jpg" height="360px;">
</p>

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

