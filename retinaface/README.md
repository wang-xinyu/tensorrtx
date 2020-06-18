# RetinaFace

 The pytorch implementation is [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface), I forked it into 
[wang-xinyu/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) and add genwts.py

This branch is using TensorRT 7 API, branch [trt4->retinaface](https://github.com/wang-xinyu/tensorrtx/tree/trt4/retinaface) is using TensorRT 4.

## Run

```
1. generate retinaface.wts from pytorch implementation https://github.com/wang-xinyu/Pytorch_Retinaface

git clone https://github.com/wang-xinyu/Pytorch_Retinaface.git
// download its weights 'Resnet50_Final.pth', put it in Pytorch_Retinaface/weights
cd Pytorch_Retinaface
python detect.py --save_model
python genwts.py
// a file 'retinaface.wts' will be generated.

2. put retinaface.wts into tensorrtx/retinaface, build and run

git clone https://github.com/wang-xinyu/tensorrtx.git
cd tensorrtx/retinaface
// put retinaface.wts here
mkdir build
cd build
cmake ..
make
sudo ./retina_r50 -s  // build and serialize model to file i.e. 'retina_r50.engine'
wget https://github.com/TencentYoutuResearch/FaceDetection-DSFD/raw/master/data/worlds-largest-selfie.jpg
sudo ./retina_r50 -d  // deserialize model file and run inference.

3. check the images generated, as follows. result.jpg
```

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78901890-9077fb80-7aab-11ea-94f1-237f51fcc347.jpg">
</p>

## Config

- Input shape `INPUT_H`, `INPUT_W` defined in `decode.h`
- FP16/FP32 can be selected by the macro `USE_FP16` in `retina_r50.cpp`
- GPU id can be selected by the macro `DEVICE` in `retina_r50.cpp`
- Batchsize can be selected by the macro `BATCHSIZE` in `retina_r50.cpp`

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)
