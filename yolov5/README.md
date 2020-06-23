# yolov5

The Pytorch implementation is [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

I was using [commit 6fd3939] of [ultralytics/yolov5](https://github.com/ultralytics/yolov5). And I made a copy of [yolov5s.pt(google drive)](https://drive.google.com/file/d/1w38DgmrP3iwiJi_AOdabuuE_zanMhO5_/view?usp=sharing). Just in case the yolov5 model updated.

## How to Run

```
1. generate yolov5s.wts from pytorch implementation with yolov5s.pt

git clone https://github.com/wang-xinyu/tensorrtx.git
git clone https://github.com/ultralytics/yolov5.git
// download its weights 'yolov5s.pt'
cd yolov5
cp ../tensorrtx/yolov5s/gen_wts.py .
python gen_wts.py
// a file 'yolov5s.wts' will be generated.

2. put yolov5s.wts into yolov5, build and run

mv yolov5s.wts ../tensorrtx/yolov5/
cd ../tensorrtx/yolov5
mkdir build
cd build
cmake ..
make
sudo ./yolov5s -s             // serialize model to plan file i.e. 'yolov5s.engine'
sudo ./yolov5s -d  ../samples // deserialize plan file and run inference, the images in samples will be processed.

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
- FP16/FP32 can be selected by the macro in yolov5s.cpp
- GPU id can be selected by the macro in yolov5s.cpp
- NMS thresh in yolov5s.cpp
- BBox confidence thresh in yolov5s.cpp
- Batch size in yolov5s.cpp

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

