# yolov3-spp

Currently this is supporting dynamic input shape, if you want to use non-dynamic version, please checkout commit [659fd2b](https://github.com/wang-xinyu/tensorrtx/commit/659fd2b23482197b19dccf746a5a3dbff1611381).

The Pytorch implementation is [ultralytics/yolov3](https://github.com/ultralytics/yolov3). It provides two trained weights of yolov3-spp, `yolov3-spp.pt` and `yolov3-spp-ultralytics.pt`(originally named `ultralytics68.pt`).

## Config

- Number of classes defined in yololayer.h
- FP16/FP32 can be selected by the macro in yolov3-spp.cpp
- GPU id can be selected by the macro in yolov3-spp.cpp
- NMS thresh in yolov3-spp.cpp
- BBox confidence thresh in yolov3-spp.cpp
- MIN and MAX input size defined in yolov3-spp.cpp
- Optimization width and height for IOptimizationProfile defined in yolov3-spp.cpp

## How to Run

1. generate yolov3-spp_ultralytics68.wts from pytorch implementation with yolov3-spp.cfg and yolov3-spp-ultralytics.pt, or download .wts from model zoo

```
git clone https://github.com/wang-xinyu/tensorrtx.git
git clone https://github.com/ultralytics/yolov3.git
// download its weights 'yolov3-spp-ultralytics.pt'
// copy gen_wts.py from tensorrtx/yolov3-spp/ to ultralytics/yolov3/
// go to ultralytics/yolov3/
python gen_wts.py yolov3-spp-ultralytics.pt
// a file 'yolov3-spp_ultralytics68.wts' will be generated.
// the master branch of yolov3 should work, if not, you can checkout 4ac60018f6e6c1e24b496485f126a660d9c793d8
```

2. build tensorrtx/yolov3-spp and run

```
// put yolov3-spp_ultralytics68.wts into tensorrtx/yolov3-spp/
// go to tensorrtx/yolov3-spp/
mkdir build
cd build
cmake ..
make
sudo ./yolov3-spp -s             // serialize model to plan file i.e. 'yolov3-spp.engine'
sudo ./yolov3-spp -d  ../samples // deserialize plan file and run inference, the images in samples will be processed.
```

3. check the images generated, as follows. _zidane.jpg and _bus.jpg

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247927-4d9fac00-751e-11ea-8b1b-704a0aeb3fcf.jpg">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247970-60b27c00-751e-11ea-88df-41473fed4823.jpg">
</p>

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

