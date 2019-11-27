# yolo v3

Thanks to Ayoosh Kathuria, for his remarkable tutorials of yolov3. The github link is [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3).

I forked his github repo, and implement inference_on_one_pic and export weights for tensorrt. You can refer to 

[pytorchx/pytorch-yolo-v3](https://github.com/wang-xinyu/pytorchx/tree/master/pytorch-yolo-v3)

Following tricks are used in this yolov3,

- I wrote a leaky relu plugin(leaky.cu  leaky.cuh  leakyplugin.cpp  leakyplugin.h) in the beginning, but I found there is PRelu in `NvInferPlugin.h`.
- yolo layer is implemented as a plugin. I learn a lot from [lewes6369/TensorRT-Yolov3](https://github.com/lewes6369/TensorRT-Yolov3).
- upsample layer is replaced by a deconvolution layer.
- Batchnorm layer, implemented by scale layer.

For FP16 mode, just need add one line `builder->setFp16Mode(true);`. On my TX1, it's 115ms in fp16, while 145ms in fp32.

```
// 1. generate yolov3.wts from [pytorchx/pytorch-yolo-v3](https://github.com/wang-xinyu/pytorchx/tree/master/pytorch-yolo-v3)

// 2. put yolov3.wts into tensorrtx/yolov3

// 3. build and run

cd tensorrtx/yolov3

mkdir build

cd build

cmake ..

make

sudo ./yolov3 -s   // serialize model to plan file i.e. 'yolov3.engine'
sudo ./yolov3 -d   // deserialize plan file and run inference

// 4. see if the output is same as pytorchx/pytorch-yolo-v3
```


