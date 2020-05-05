# yolov4

The Pytorch implementation is from [ultralytics/yolov3](https://github.com/ultralytics/yolov3). It can load yolov4.cfg and yolov4.weights(from AlexeyAB/darknet).

Following tricks are used in this yolov4:

- Three yololayer are implemented in one plugin to improve speed, codes derived from [lewes6369/TensorRT-Yolov3](https://github.com/lewes6369/TensorRT-Yolov3)
- Mish activation, implemented in a plugin.
- Batchnorm layer, implemented by scale layer.

## Excute:

```
1. generate yolov4.wts from pytorch implementation with yolov4.cfg and yolov4.weights

git clone https://github.com/wang-xinyu/tensorrtx.git
git clone https://github.com/ultralytics/yolov3.git
// download yolov4.weights from https://github.com/AlexeyAB/darknet#pre-trained-models
cd yolov3
cp ../tensorrtx/yolov4/gen_wts.py .
python gen_wts.py yolov4.weights
// a file 'yolov4.wts' will be generated.
// the master branch of yolov3 should work, if not, you can checkout be87b41aa2fe59be8e62f4b488052b24ad0bd450

2. put yolov4.wts into ./yolov4, build and run

mv yolov4.wts ../tensorrtx/yolov4/
cd ../tensorrtx/yolov4
mkdir build
cd build
cmake ..
make
sudo ./yolov4 -s             // serialize model to plan file i.e. 'yolov4.engine'
sudo ./yolov4 -d  ../../yolov3-spp/samples // deserialize plan file and run inference, the images in samples will be processed.

3. check the images generated, as follows. _zidane.jpg and _bus.jpg
```

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/80863728-cbd3a780-8cb0-11ea-8640-7983bb41c354.jpg">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/80863730-cfffc500-8cb0-11ea-810e-94d693e71d80.jpg">
</p>

## Config

- Input shape `INPUT_H`, `INPUT_W` defined in yololayer.h
- Number of classes `CLASS_NUM` defined in yololayer.h
- FP16/FP32 can be selected by the macro `USE_FP16` in yolov4.cpp
- GPU id can be selected by the macro `DEVICE` in yolov4.cpp
- NMS thresh `NMS_THRESH` in yolov4.cpp
- bbox confidence threshold `BBOX_CONF_THRESH` in yolov4.cpp
- `BATCH_SIZE` in yolov4.cpp

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)
