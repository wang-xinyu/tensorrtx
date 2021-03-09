# yolov3-tiny

The Pytorch implementation is [ultralytics/yolov3](https://github.com/ultralytics/yolov3).

## Excute:

```
1. generate yolov3-tiny.wts from pytorch implementation with yolov3-tiny.cfg and yolov3-tiny.weights, or download .wts from model zoo

git clone -b archive https://github.com/ultralytics/yolov3.git
// download its weights 'yolov3-tiny.pt' or 'yolov3-tiny.weights'
// put tensorrtx/yolov3-tiny/gen_wts.py into ultralytics/yolov3 and run
python gen_wts.py yolov3-tiny.weights
// a file 'yolov3-tiny.wts' will be generated.

2. put yolov3-tiny.wts into tensorrtx/yolov3-tiny, build and run

// go to tensorrtx/yolov3-tiny
mkdir build
cd build
cmake ..
make
sudo ./yolov3-tiny -s             // serialize model to plan file i.e. 'yolov3-tiny.engine'
sudo ./yolov3-tiny -d  ../../yolov3-spp/samples // deserialize plan file and run inference, the images in samples will be processed.

3. check the images generated, as follows. _zidane.jpg and _bus.jpg
```

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247927-4d9fac00-751e-11ea-8b1b-704a0aeb3fcf.jpg">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247970-60b27c00-751e-11ea-88df-41473fed4823.jpg">
</p>

## Config

- Input shape defined in yololayer.h
- Number of classes defined in yololayer.h
- FP16/FP32 can be selected by the macro in yolov3-tiny.cpp
- GPU id can be selected by the macro in yolov3-tiny.cpp
- NMS thresh in yolov3-tiny.cpp
- BBox confidence thresh in yolov3-tiny.cpp

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

