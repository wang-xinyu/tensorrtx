# DETR

The Pytorch implementation is [facebookresearch/detr](https://github.com/facebookresearch/detr).

For details see [End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers).

## Test Environment

- GTX2080Ti / Ubuntu16.04 / cuda10.2 / cudnn8.0.4 / TensorRT7.2.1 / OpenCV4.2
- GTX2080Ti / win10 / cuda10.2 / cudnn8.0.4 / TensorRT7.2.1 / OpenCV4.2 / VS2017

## How to Run

1. generate .wts from pytorch with .pth

```
// git clone https://github.com/facebookresearch/detr.git
// go to facebookresearch/detr
// download https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
// download https://raw.githubusercontent.com/freedenS/TestImage/main/demo.jpg
// copy tensorrtx/detr/gen_wts.py and demo.jpg into facebookresearch/detr
python gen_wts.py
// a file 'detr.wts' will be generated.
```

2. build tensorrtx/detr and run

```
// put detr.wts into tensorrtx/detr
// go to tensorrtx/detr
// update parameters in detr.cpp if your model is trained on custom dataset.The parameters are corresponding to config in detr.
mkdir build
cd build
cmake ..
make
sudo ./detr -s [.wts] // serialize model to plan file
sudo ./detr -d [.engine] [image folder] // deserialize and run inference, the images in [image folder] will be processed
// For example
sudo ./detr -s ../detr.wts detr.engine
sudo ./detr -d detr.engine ../samples
```

3. check the images generated, as follows. _demo.jpg and so on.

## NOTE

- tensorrt use fixed input size, if the size of your data is different from the engine, you need to adjust your data and the result.
- image preprocessing with c++ is a little different with python(opencv vs PIL)


## Latency

average cost of doInference(in detr.cpp) from second time with batch=1 under the ubuntu environment above

|      | fp32    | fp16    | int8 |
| ---- | ------- | ------- | ---- |
| R50  | 19.57ms | 9.424ms | TODO |

