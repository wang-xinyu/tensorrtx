# yolov5

The Pytorch implementation is [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

Currently, we support yolov5 v1.0(yolov5s only), v2.0, v3.0, v3.1 and v4.0.

- For yolov5 v4.0, please visit [yolov5 release v4.0](https://github.com/ultralytics/yolov5/releases/tag/v4.0), and use the latest commit of this repo.
- For yolov5 v3.0 and v3.1, please visit [yolov5 release v3.0](https://github.com/ultralytics/yolov5/releases/tag/v3.0) and [yolov5 release v3.1](https://github.com/ultralytics/yolov5/releases/tag/v3.1), and checkout commit ['6d0f5cb'](https://github.com/wang-xinyu/tensorrtx/commit/6d0f5cbf4745bc00b69aad54a905383fb906f103) of this repo.
- For yolov5 v2.0, please visit [yolov5 release v2.0](https://github.com/ultralytics/yolov5/releases/tag/v2.0), and checkout commit ['5cfa444'](https://github.com/wang-xinyu/tensorrtx/commit/5cfa4445170eabaa54acd5ad7f469ef65a8763f1) of this repo.
- For yolov5 v1.0, please visit [yolov5 release v1.0](https://github.com/ultralytics/yolov5/releases/tag/v1.0), and checkout commit ['f09aa3b'](https://github.com/wang-xinyu/tensorrtx/commit/f09aa3bbebf4d4d37b6d3b32a1d39e1f2678a07b) of this repo.

## Config

- Choose the model s/m/l/x by `NET` macro in yolov5.cpp
- Input shape defined in yololayer.h
- Number of classes defined in yololayer.h, **DO NOT FORGET TO ADAPT THIS, If using your own model**
- FP16/FP32 can be selected by the macro in yolov5.cpp
- GPU id can be selected by the macro in yolov5.cpp
- NMS thresh in yolov5.cpp
- BBox confidence thresh in yolov5.cpp
- Batch size in yolov5.cpp

## How to Run, yolov5s as example

1. generate yolov5s.wts from pytorch with yolov5s.pt, or download .wts from model zoo

```
git clone https://github.com/wang-xinyu/tensorrtx.git
git clone https://github.com/ultralytics/yolov5.git
// download its weights 'yolov5s.pt'
// copy tensorrtx/yolov5/gen_wts.py into ultralytics/yolov5
// ensure the file name is yolov5s.pt and yolov5s.wts in gen_wts.py
// go to ultralytics/yolov5
python gen_wts.py
// a file 'yolov5s.wts' will be generated.
```

2. build tensorrtx/yolov5 and run

```
// put yolov5s.wts into tensorrtx/yolov5
// go to tensorrtx/yolov5
// ensure the macro NET in yolov5.cpp is s
// update CLASS_NUM in yololayer.h if your model is trained on custom dataset
mkdir build
cd build
cmake ..
make
sudo ./yolov5 -s 0.33 0.50   // serialize yolov5s model to plan file i.e. 'yolov5.engine'
sudo ./yolov5 -s 0.67 0.75   // serialize yolov5m model to plan file i.e. 'yolov5.engine'
sudo ./yolov5 -s 1.0 1.0   // serialize yolov5l model to plan file i.e. 'yolov5.engine'
sudo ./yolov5 -s 1.33 1.25   // serialize yolov5x model to plan file i.e. 'yolov5.engine'
sudo ./yolov5 -s 0.17 0.25   // serialize your own yolov5 model with depth_multiple and width_multiple which you set in '.yaml' file when training the model. i.e. depth = 0.17, width = 0.25
sudo ./yolov5 -d  ../samples // deserialize plan file and run inference, the images in samples will be processed.
```

3. check the images generated, as follows. _zidane.jpg and _bus.jpg

4. optional, load and run the tensorrt model in python

```
// install python-tensorrt, pycuda, etc.
// ensure the yolov5s.engine and libmyplugins.so have been built
python yolov5_trt.py
```

# INT8 Quantization

1. Prepare calibration images, you can randomly select 1000s images from your train set. For coco, you can also download my calibration images `coco_calib` from [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh

2. unzip it in yolov5/build

3. set the macro `USE_INT8` in yolov5.cpp and make

4. serialize the model and test

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247927-4d9fac00-751e-11ea-8b1b-704a0aeb3fcf.jpg">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247970-60b27c00-751e-11ea-88df-41473fed4823.jpg">
</p>

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

