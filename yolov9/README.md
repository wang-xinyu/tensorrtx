# yolov9

The Pytorch implementation is [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9).

## Contributors


## Requirements

- TensorRT 8.0+
- OpenCV 3.4.0+

## How to Run, yolov9 as example

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```
// download https://github.com/WongKinYiu/yolov9
cp {tensorrtx}/yolov9/gen_wts.py {yolov9}/yolov9
cd {yolov9}/yolov9
python gen_wts.py
// a file 'yolov9.wts' will be generated.
```
2. build tensorrtx/yolov9 and run


```
cd {tensorrtx}/yolov9/
// update kNumClass in config.h if your model is trained on custom dataset
mkdir build
cd build
cp {ultralytics}/ultralytics/yolov9.wts {tensorrtx}/yolov9/build
cmake ..
make
sudo ./yolov9 -s [.wts] [.engine] [c/e]  // serialize model to plan file
sudo ./yolov9 -d [.engine] [image folder] // deserialize and run inference, the images in [image folder] will be processed.
// For example yolov9
sudo ./yolov9 -s yolov9-c.wts yolov9-c.engine c
sudo ./yolov9 -d yolov9-c.engine ../images
```

3. check the images generated, as follows. _zidane.jpg and _bus.jpg

4. optional, load and run the tensorrt model in python

```
// install python-tensorrt, pycuda, etc.
// ensure the yolov9.engine and libmyplugins.so have been built
python yolov9_trt.py
```

# INT8 Quantization

1. Prepare calibration images, you can randomly select 1000s images from your train set. For coco, you can also download my calibration images `coco_calib` from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh

2. unzip it in yolov8/build

3. set the macro `USE_INT8` in config.h and change the path of calibration images in config.h, such as 'gCalibTablePath="./coco_calib/";'

4. serialize the model and test

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247927-4d9fac00-751e-11ea-8b1b-704a0aeb3fcf.jpg" height="360px;">
</p>

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)


