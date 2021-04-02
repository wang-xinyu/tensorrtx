# RetinaFace

 The pytorch implementation is [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface), I forked it into 
[wang-xinyu/Pytorch_Retinaface](https://github.com/wang-xinyu/Pytorch_Retinaface) and add genwts.py

This branch is using TensorRT 7 API, branch [trt4->retinaface](https://github.com/wang-xinyu/tensorrtx/tree/trt4/retinaface) is using TensorRT 4.

## Config

- Input shape `INPUT_H`, `INPUT_W` defined in `decode.h`
- INT8/FP16/FP32 can be selected by the macro `USE_FP16` or `USE_INT8` or `USE_FP32` in `retina_r50.cpp`
- GPU id can be selected by the macro `DEVICE` in `retina_r50.cpp`
- Batchsize can be selected by the macro `BATCHSIZE` in `retina_r50.cpp`

## Run

The following described how to run `retina_r50`. While `retina_mnet` is nearly the same, just generate `retinaface.wts` with `mobilenet0.25_Final.pth` and run `retina_mnet`.

1. generate retinaface.wts from pytorch implementation https://github.com/wang-xinyu/Pytorch_Retinaface

```
git clone https://github.com/wang-xinyu/Pytorch_Retinaface.git
// download its weights 'Resnet50_Final.pth', put it in Pytorch_Retinaface/weights
cd Pytorch_Retinaface
python detect.py --save_model
python genwts.py
// a file 'retinaface.wts' will be generated.
```

2. put retinaface.wts into tensorrtx/retinaface, build and run

```
git clone https://github.com/wang-xinyu/tensorrtx.git
cd tensorrtx/retinaface
// put retinaface.wts here
mkdir build
cd build
cmake ..
make
sudo ./retina_r50 -s  // build and serialize model to file i.e. 'retina_r50.engine'
wget https://github.com/Tencent/FaceDetection-DSFD/raw/master/data/worlds-largest-selfie.jpg
sudo ./retina_r50 -d  // deserialize model file and run inference.
```

3. check the images generated, as follows. 0_result.jpg

4. we also provide a python wrapper

```
// install python-tensorrt, pycuda, etc.
// ensure the retina_r50.engine and libdecodeplugin.so have been built
python retinaface_trt.py
```

# INT8 Quantization

1. Prepare calibration images, you can randomly select 1000s images from your train set. For widerface, you can also download my calibration images `widerface_calib` from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh

2. unzip it in retinaface/build

3. set the macro `USE_INT8` in retina_r50.cpp and make

4. serialize the model and test

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78901890-9077fb80-7aab-11ea-94f1-237f51fcc347.jpg">
</p>

## More Information

Check the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

