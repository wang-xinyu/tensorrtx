# Rcnn

The Pytorch implementation is [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2).

## Models

- [x] Faster R-CNN(R50-C4)

- [ ] Mask R-CNN(R50-C4)

## Test Environment

- GTX2080Ti / Ubuntu16.04 / cuda10.2 / cudnn8.0.4 / TensorRT7.0.0 / OpenCV4.2
- GTX2080Ti / win10 / cuda10.2 / cudnn8.0.4 / TensorRT7.2.1 / OpenCV4.2 / VS2017 (need to replace function corresponding to the dirent.h and add "--extended-lambda" in CUDA C/C++ -> Command Line -> Other options)

## How to Run

1. generate .wts from pytorch with .pkl or .pth

```
// git clone -b v0.4 https://github.com/facebookresearch/detectron2.git
// go to facebookresearch/detectron2
python setup.py build develop // more install information see https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
// download https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl
// copy tensorrtx/rcnn/(gen_wts.py,demo.jpg) into facebookresearch/detectron2
// ensure cfg.MODEL.WEIGHTS in gen_wts.py is correct
// go to facebookresearch/detectron2
python gen_wts.py
// a file 'faster.wts' will be generated.
```

2. build tensorrtx/rcnn and run

```
// put faster.wts into tensorrtx/rcnn
// go to tensorrtx/rcnn
// update parameters in rcnn.cpp if your model is trained on custom dataset.The parameters are corresponding to config in detectron2.
mkdir build
cd build
cmake ..
make
sudo ./rcnn -s [.wts] // serialize model to plan file
sudo ./rcnn -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.
// For example
sudo ./rcnn -s faster.wts faster.engine
sudo ./rcnn -d faster.engine ../samples
```

3. check the images generated, as follows. _zidane.jpg and _bus.jpg

## NOTE

- if you meet the error below, just try to make again. The flag has been added in CMakeLists.txt

  ```
  error: __host__ or __device__ annotation on lambda requires --extended-lambda nvcc flag
  ```

- the image preprocess was moved into tensorrt, see DataPreprocess(rcnn.cpp line 61), so the input data is {H, W, C}

- the predicted boxes is corresponding to new image size, so the final boxes need to multiply with the ratio, see calculateRatio(rcnn.cpp line 113)

- tensorrt use fixed input size, if the size of your data is different from the engine, you need to adjust your data and the result.

## Quantization

1. quantizationType:fp32,fp16,int8. see BuildRcnnModel(rcnn.cpp line 276) for detail.

2. the using of int8 is same with [tensorrtx/yolov5](../yolov5/README.md), but it has no improvement comparing to fp16.


