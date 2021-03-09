# yolov3

The Pytorch implementation is [ultralytics/yolov3](https://github.com/ultralytics/yolov3). It provides two trained weights of yolov3, `yolov3.weights` and `yolov3.pt`

This branch is using tensorrt7 API, there is also a yolov3 implementation using tensorrt4 API, go to [branch trt4/yolov3](https://github.com/wang-xinyu/tensorrtx/tree/trt4/yolov3), which is using [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3).

## Config

- Input shape defined in yololayer.h
- Number of classes defined in yololayer.h
- INT8/FP16/FP32 can be selected by the macro in yolov3.cpp
- GPU id can be selected by the macro in yolov3.cpp
- NMS thresh in yolov3.cpp
- BBox confidence thresh in yolov3.cpp

## How to run

1. generate yolov3.wts from pytorch implementation with yolov3.cfg and yolov3.weights, or download .wts from model zoo

```
git clone https://github.com/wang-xinyu/tensorrtx.git
git clone -b archive https://github.com/ultralytics/yolov3.git
// download its weights 'yolov3.pt' or 'yolov3.weights'
cp {tensorrtx}/yolov3/gen_wts.py {ultralytics/yolov3/}
cd {ultralytics/yolov3/}
python gen_wts.py yolov3.weights
// a file 'yolov3.wts' will be generated.
// the master branch of yolov3 should work, if not, you can checkout cf7a4d31d37788023a9186a1a143a2dab0275ead
```

2. put yolov3.wts into tensorrtx/yolov3, build and run

```
mv yolov3.wts {tensorrtx}/yolov3/
cd {tensorrtx}/yolov3
mkdir build
cd build
cmake ..
make
sudo ./yolov3 -s                          // serialize model to plan file i.e. 'yolov3.engine'
sudo ./yolov3 -d ../../yolov3-spp/samples // deserialize plan file and run inference, the images in samples will be processed.
```

3. check the images generated, as follows. _zidane.jpg and _bus.jpg

# INT8 Quantization

1. Prepare calibration images, you can randomly select 1000s images from your train set. For coco, you can also download my calibration images `coco_calib` from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh

2. unzip it in yolov3/build

3. set the macro `USE_INT8` in yolov3.cpp and make

4. serialize the model and test

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247927-4d9fac00-751e-11ea-8b1b-704a0aeb3fcf.jpg">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247970-60b27c00-751e-11ea-88df-41473fed4823.jpg">
</p>

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

