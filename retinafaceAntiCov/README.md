# RetinaFaceAntiCov

 The mxnet implementation is [deepinsight/insightface/RetinaFaceAntiCov](https://github.com/deepinsight/insightface/tree/master/RetinaFaceAntiCov).

## Run

```
1. generate retinafaceAntiCov.wts from mxnet implementation.

git clone https://github.com/deepinsight/insightface.git
cd insightface/RetinaFaceAntiCov
// download its weights 'cov2.zip', put it into insightface/RetinaFaceAntiCov, and unzip it
// put tensorrtx/retinafaceAntiCov/gen_wts.py into insightface/RetinaFaceAntiCov
python gen_wts.py
// a file 'retinafaceAntiCov.wts' will be generated.

2. put retinafaceAntiCov.wts into tensorrtx/retinafaceAntiCov, build and run

git clone https://github.com/wang-xinyu/tensorrtx.git
cd tensorrtx/retinafaceAntiCov
// put retinafaceAntiCov.wts here
mkdir build
cd build
cmake ..
make
sudo ./retinafaceAntiCov -s  // build and serialize model to file i.e. 'retinafaceAntiCov.engine'
wget http://www.kaixian.tv/gd/d/file/201611/07/23efff3a26e2385620e719378c654fb1.jpg -O test.jpg
sudo ./retinafaceAntiCov -d  // deserialize model file and run inference.

3. check the image generated, as follows 'out.jpg'
```

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/84776553-069c5f80-b013-11ea-893c-70a138b843d6.jpg">
</p>

## Config

- Input shape `INPUT_H`, `INPUT_W` defined in `decode.h`
- FP16/FP32 can be selected by the macro `USE_FP16` in `retinafaceAntiCov.cpp`
- GPU id can be selected by the macro `DEVICE` in `retinafaceAntiCov.cpp`

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)
