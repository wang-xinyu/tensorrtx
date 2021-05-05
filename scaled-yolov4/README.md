# scaled-yolov4

The Pytorch implementation is from [WongKinYiu/ScaledYOLOv4 yolov4-csp branch](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp). It can load yolov4-csp.cfg and yolov4-csp.weights(from AlexeyAB/darknet).

Note: There is a slight difference in yolov4-csp.cfg for darknet and pytorch. Use the one given in the above repo.

## Config

- Input shape `INPUT_H`, `INPUT_W` defined in yololayer.h
- Number of classes `CLASS_NUM` defined in yololayer.h
- FP16/FP32 can be selected by the macro `USE_FP16` in yolov4_csp.cpp
- GPU id can be selected by the macro `DEVICE` in yolov4_csp.cpp
- NMS thresh `NMS_THRESH` in yolov4_csp.cpp
- bbox confidence threshold `BBOX_CONF_THRESH` in yolov4_csp.cpp
- `BATCH_SIZE` in yolov4_csp.cpp

## How to run

1. generate yolov4_csp.wts from pytorch implementation with yolov4-csp.cfg and yolov4-csp.weights.

```
git clone https://github.com/wang-xinyu/tensorrtx.git
git clone -b yolov4-csp https://github.com/WongKinYiu/ScaledYOLOv4.git
// download yolov4-csp.weights from https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp#yolov4-csp
cp {tensorrtx}/scaled-yolov4/gen_wts.py {ScaledYOLOv4/}
cd {ScaledYOLOv4/}
python gen_wts.py yolov4-csp.weights
// a file 'yolov4_csp.wts' will be generated.
```

2. put yolov4_csp.wts into {tensorrtx}/scaled-yolov4, build and run

```
mv yolov4_csp.wts {tensorrtx}/scaled-yolov4/
cd {tensorrtx}/scaled-yolov4
mkdir build
cd build
cmake ..
make
sudo ./yolov4csp -s                          // serialize model to plan file i.e. 'yolov4csp.engine'
sudo ./yolov4csp -d ../../yolov3-spp/samples // deserialize plan file and run inference, the images in samples will be processed.
```

3. check the images generated, as follows. _zidane.jpg and _bus.jpg
<p align="center">
<img src= https://user-images.githubusercontent.com/39617050/117172509-824cf980-ade9-11eb-8e4c-27dbe658e355.jpg>
</p>

<p align="center">
<img src= https://user-images.githubusercontent.com/39617050/117172880-dbb52880-ade9-11eb-839a-0814fd46198e.jpg>
</p>


## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)
