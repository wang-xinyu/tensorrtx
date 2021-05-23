# CenterNet

This is the trt implementation of detection model [ctdet_coco_dla_2x](https://drive.google.com/open?id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT) from [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet) official work. 

## How to Run

1. Follow [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) tutorial to build TensorRT7

2. Copy folder `dcnv2Plugin` to `TensorRT/plugin` and edit `InferPlugin.cpp` and `CMakeLists.txt`

3. Rebuild to install custom plugin

4. Use `tensorrt-7.2.3.4-cp36-none-linux_x86_64.whl` in TensorRT OSS to update your python-tensorrt

5. Run `python centernet.py -m ${PTH_PATH} -s` to create trt engine 

## Sample

```
// Download ctdet_coco_dla_2x.pth and transfer it into trt engine first
// Download the test img from https://raw.githubusercontent.com/tensorflow/models/master/research/deeplab/g3doc/img/image2.jpg or choose your own one
cd sample
python test.py ${ENGINE_PATH} ${IMG_PATH}
```
![trt_out](https://user-images.githubusercontent.com/47047345/119128637-7a878900-ba68-11eb-91ff-5dcc10f01b77.jpg)

## TODO

Integrate the post process with trt engine to make it more easier to use.