# Rcnn

The Pytorch implementation is [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2).

## Models

- [x] Faster R-CNN(C4)

- [x] Mask R-CNN(C4)

## Test Environment

- GTX2080Ti / Ubuntu16.04 / cuda10.2 / cudnn8.0.4 / TensorRT7.2.1 / OpenCV4.2
- GTX2080Ti / win10 / cuda10.2 / cudnn8.0.4 / TensorRT7.2.1 / OpenCV4.2 / VS2017 (need to replace function corresponding to the dirent.h and add "--extended-lambda" in CUDA C/C++ -> Command Line -> Other options)

TensorRT7.2 is recomended because Resize layer in 7.0 with kLINEAR mode is a little different with opencv. You can also implement data preprocess out of tensorrt if you want to use TensorRT7.0 or more previous version. 

**The result under fp32 is same to pytorch about 4 decimal places**!

## How to Run

1. generate .wts from pytorch with .pkl or .pth

```
// git clone -b v0.4 https://github.com/facebookresearch/detectron2.git
// go to facebookresearch/detectron2
python setup.py build develop // more install information see https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
// download https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl
// download https://raw.githubusercontent.com/freedenS/TestImage/main/demo.jpg
// copy tensorrtx/rcnn/gen_wts.py and demo.jpg into facebookresearch/detectron2
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
sudo ./rcnn -s [.wts] [m] // serialize model to plan file, add m for maskrcnn
sudo ./rcnn -d [.engine] [image folder] [m] // deserialize and run inference, the images in [image folder] will be processed. add m for maskrcnn
// For example
sudo ./rcnn -s faster.wts faster.engine
sudo ./rcnn -d faster.engine ../samples
// sudo ./rcnn -s mask.wts mask.engine m
// sudo ./rcnn -d mask.engine ../samples m
```

3. check the images generated, as follows. _demo.jpg and so on.

## Backbone

#### R18, R34, R152

```
// python
1.download pretrained model
  R18: https://download.pytorch.org/models/resnet18-f37072fd.pth
  R34: https://download.pytorch.org/models/resnet34-b627a593.pth
  R50: https://download.pytorch.org/models/resnet50-0676ba61.pth
  R101: https://download.pytorch.org/models/resnet101-63fe2227.pth
  R152: https://download.pytorch.org/models/resnet152-394f9c45.pth
2.convert pth to pkl by facebookresearch/detectron2/tools/convert-torchvision-to-d2.py
3.set merge_from_file in gen_wts.py
  ./configs/COCO-Detections/faster_rcnn_R_50_C4_1x.yaml for fasterRcnn
  ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml for maskRcnn
4.set cfg.MODEL.RESNETS.DEPTH = 18(34,50,101,152),
      cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False,
      cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64, // for R18, R34; 256 for others
      cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530],
      cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375],
      cfg.INPUT.FORMAT = "RGB"
  and then train your own model
5.generate your wts file.
// c++
6.set BACKBONE_RESNETTYPE = R18(R34,R50,R101,R152) in rcnn.cpp line 14
7.modify PIXEL_MEAN and PIXEL_STD in rcnn.cpp
8.set STRIDE_IN_1X1=false in backbone.hpp line 9
9.set other parameters if it's not same with default
10.build your engine, refer to how to run
11.convert your image to RGB before inference
```

#### R50, R101

```
1.download pretrained model
  R50: https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl for fasterRcnn
       https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl for maskRcnn
  R101: https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl for fasterRcnn
        https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl for maskRcnn
2.set merge_from_file in gen_wts.py
  R50-faster: ./configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml
  R101-faster: ./configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml
  R50-mask: ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml
  R101-mask: ./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml
3.set BACKBONE_RESNETTYPE = R50(R101) rcnn.cpp line 13
4.set STRIDE_IN_1X1=true in backbone.hpp
5.follow how to run
```

## NOTE

- if you meet the error below, just try to make again. The flag has been added in CMakeLists.txt

  ```
  error: __host__ or __device__ annotation on lambda requires --extended-lambda nvcc flag
  ```

- the image preprocess was moved into tensorrt, see DataPreprocess in rcnn.cpp, so the input data is {H, W, C}

- the predicted boxes is corresponding to new image size, so the final boxes need to multiply with the ratio, see calculateRatio in rcnn.cpp

- tensorrt use fixed input size, if the size of your data is different from the engine, you need to adjust your data and the result.

- if you want to use maskrcnn with cuda10.2, please be sure that you have upgraded cuda to the latest patch. see https://github.com/NVIDIA/TensorRT/issues/1151 for detail.

- you can build fasterRcnn with maskRcnn weights file.

- do initializing for _pre_nms_topk in RpnNmsPlugin and _count in BatchedNmsPlugin inside class to prevent error assert, because the configurePlugin function is implemented after clone() and before serialize(). one can also set it through constructor.

## Quantization

1. quantizationType:fp32,fp16,int8. see BuildRcnnModel(rcnn.cpp line 276) for detail.

2. the usage of int8 is same with [tensorrtx/yolov5](../yolov5/README.md), but it has no improvement comparing to fp16.

## Plugins

decode and nms plugins are modified from [retinanet-examples](https://github.com/NVIDIA/retinanet-examples/tree/master/csrc/plugins)

- RpnDecodePlugin: calculate coordinates of  proposals which is the first n

```
parameters:
  top_n: num of proposals to select
  anchors: coordinates of all anchors
  stride: stride of current feature map
  image_height: iamge height after DataPreprocess for clipping the box beyond the boundary
  image_width: iamge width after DataPreprocess for clipping the box beyond the boundary

Inputs:
  scores{C,H,W} C is number of anchors, H and W are the size of feature map
  boxes{C,H,W} C is 4*number of anchors, H and W are the size of feature map
Outputs:
  scores{C,1} C is equal to top_n
  boxes{C,4} C is equal to top_n
```

- RpnNmsPlugin: apply nms to proposals

```
parameters:
  nms_thresh: thresh of nms
  post_nms_topk: number of proposals to select
  
Inputs:
  scores{C,1} C is equal to top_n
  boxes{C,4} C is equal to top_n
Outputs:
  boxes{C,4} C is equal to post_nms_topk
```

- RoiAlignPlugin: implement of RoiAlign(align=True). see https://github.com/facebookresearch/detectron2/blob/f50ec07cf220982e2c4861c5a9a17c4864ab5bfd/detectron2/layers/roi_align.py#L7 for detail

```
parameters:
  pooler_resolution: output size
  spatial_scale: scale the input boxes by this number
  sampling_ratio: number of inputs samples to take for each output
  num_proposals: number of proposals
  
Inputs:
  boxes{N,4} N is number of boxes
  features{C,H,W} C is channels of feature map, H and W are sizes of feature map
Outputs:
  features{N,C,H,W} N is number of boxes, C is channels of feature map, H and W are equal to pooler_resolution
```

- PredictorDecodePlugin: calculate coordinates of predicted boxes by applying delta to proposals

```
parameters:
  num_boxes: num of proposals
  image_height: iamge height after DataPreprocess for clipping the box beyond the boundary
  image_width: iamge width after DataPreprocess for clipping the box beyond the boundary
  bbox_reg_weights: the weights for dx,dy,dw,dh. see https://github.com/facebookresearch/detectron2/blob/master/detectron2/config/defaults.py#L292 for detail

Inputs:
  scores{N,C,1,1} N is euqal to num_boxes, C is the num of classes
  boxes{N,C,1,1} N is euqal to num_boxes, C is the num of classes
  proposals{N,4} N is equal to num_boxes
Outputs:
  scores{N,1} N is equal to num_boxes
  boxes{N,4} N is equal to num_boxes
  classes{N,1} N is equal to num_boxes
```

- BatchedNmsPlugin: apply nms to predicted boxes with different classes. same with https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/nms.py#L19

```
parameters:
  nms_thresh: thresh of nms
  detections_per_im: number of detections to return per image

Inputs:
  scores{N,1} N is the number of the boxes
  boxes{N,4} N is the number of the boxes
  classes{N,1} N is the number of the boxes
Outputs:
  scores{N,1} N is equal to detections_per_im
  boxes{N,4} N is equal to detections_per_im
  classes{N,1} N is equal to detections_per_im
```

- MaskRcnnInferencePlugin:  extract the masks for the predicted classes and do sigmoid. same with https://github.com/facebookresearch/detectron2/blob/9c7f8a142216ebc52d3617c11f8fafd75b74e637/detectron2/modeling/roi_heads/mask_head.py#L114

```
parameters:
  detections_per_im: number of detections to return per image
  output_size: same with output size of RoiAlign

Inputs:
  indices{N,1} N is the number of the predicted boxes
  masks{N,C,H,W} N is the number of the predicted boxes
Outputs:
  selected_masks{N,1,H,W} N is the number of the predicted boxes, H and W is equal to output_size
```

