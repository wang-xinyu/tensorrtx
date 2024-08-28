# YOLOv9

The Pytorch implementation is [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9).

## Contributors

<a href="https://github.com/WuxinrongY"><img src="https://avatars.githubusercontent.com/u/53141838?v=4?s=48" width="40px;" alt=""/></a>

## Progress
- [x] YOLOv9-t
- [x] YOLOv9-t-convert(gelan)
- [x] YOLOv9-s
- [x] YOLOv9-s-convert(gelan)
- [x] YOLOv9-m
- [x] YOLOv9-m-convert(gelan)
- [x] YOLOv9-c
- [x] YOLOv9-c-convert(gelan)
- [x] YOLOv9-e
- [x] YOLOv9-e-convert(gelan)

## Requirements

- TensorRT 8.0+
- OpenCV 3.4.0+

## Speed Test

The speed test is done on a desktop with R7-5700G CPU and RTX 4060Ti GPU. The input size is 640x640. The FP32, FP16 and INT8 models are tested. The time only includes the inference time, not includes the pre-processing and post-processing. The time is the average of 1000 times inference.

| frame  | Model | FP32 | FP16 | INT8 |
| --- | --- | --- | --- | --- |
| tensorrt | YOLOv5-n | -ms | 0.58ms | -ms |
| tensorrt | YOLOv5-s | -ms | 0.90ms | -ms |
| tensorrt | YOLOv5-m | -ms | 1.9ms | -ms |
| tensorrt | YOLOv5-l | -ms | 2.8ms | -ms |
| tensorrt | YOLOv5-x | -ms | 5.1ms | -ms |
| tensorrt | YOLOv9-t-convert | -ms | 1.37ms | -ms |
| tensorrt | YOLOv9-s | -ms | 1.78ms | -ms |
| tensorrt | YOLOv9-s-convert | -ms | 1.78ms | -ms |
| tensorrt | YOLOv9-m | -ms | 3.1ms | -ms |
| tensorrt | YOLOv9-m-convert | -ms | 2.8ms | -ms |
| tensorrt | YOLOv9-c | 13.5ms | 4.6ms | 3.0ms |
| tensorrt | YOLOv9-e | 8.3ms | 3.2ms | 2.15ms |

**GELAN will be updated later.**

YOLOv9-e is faster than YOLOv9-c in tensorrt, because the YOLOv9-e requires fewer layers of inference.

```
YOLOv9-c:
[[31, 34, 37, 16, 19, 22], 1, DualDDetect, [nc]] # [A3, A4, A5, P3, P4, P5]

YOLOv9-e:
[[35, 32, 29, 42, 45, 48], 1, DualDDetect, [nc]]

```

In DualDDetect, the A3, A4, A5, P3, P4, P5 are the output of the backbone. The first 3 layers are used for the inference of the final result.

The YOLOv9-c requires 37 layers of inference, but YOLOv9-e requires 35 layers of inference.

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

2. unzip it in yolov9/build

3. set the macro `USE_INT8` in config.h and change the path of calibration images in config.h, such as 'gCalibTablePath="./coco_calib/";'

4. serialize the model and test

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247927-4d9fac00-751e-11ea-8b1b-704a0aeb3fcf.jpg" height="360px;">
</p>

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)
