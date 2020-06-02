# yolov3-spp

yolov4 is [here](../yolov4).

The Pytorch implementation is [ultralytics/yolov3](https://github.com/ultralytics/yolov3). It provides two trained weights of yolov3-spp, `yolov3-spp.pt` and `yolov3-spp-ultralytics.pt`(originally named `ultralytics68.pt`).

Following tricks are used in this yolov3-spp:

- Yololayer plugin is different from the plugin used in [this repo's yolov3](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov3). In this version, three yololayer are implemented in one plugin to improve speed, codes derived from [lewes6369/TensorRT-Yolov3](https://github.com/lewes6369/TensorRT-Yolov3)
- Batchnorm layer, implemented by scale layer.

## Excute:

```
1. generate yolov3-spp_ultralytics68.wts from pytorch implementation with yolov3-spp.cfg and yolov3-spp-ultralytics.pt

git clone https://github.com/wang-xinyu/tensorrtx.git
git clone https://github.com/ultralytics/yolov3.git
// download its weights 'yolov3-spp-ultralytics.pt'
cd yolov3
cp ../tensorrtx/yolov3-spp/gen_wts.py .
python gen_wts.py yolov3-spp-ultralytics.pt
// a file 'yolov3-spp_ultralytics68.wts' will be generated.
// the master branch of yolov3 should work, if not, you can checkout 4ac60018f6e6c1e24b496485f126a660d9c793d8

2. put yolov3-spp_ultralytics68.wts into yolov3-spp, build and run

mv yolov3-spp_ultralytics68.wts ../tensorrtx/yolov3-spp/
cd ../tensorrtx/yolov3-spp
mkdir build
cd build
cmake ..
make
sudo ./yolov3-spp -s             // serialize model to plan file i.e. 'yolov3-spp.engine'
sudo ./yolov3-spp -d  ../samples // deserialize plan file and run inference, the images in samples will be processed.

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
- FP16/FP32 can be selected by the macro in yolov3-spp.cpp
- GPU id can be selected by the macro in yolov3-spp.cpp
- NMS thresh in yolov3-spp.cpp
- BBox confidence thresh in yolov3-spp.cpp

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

