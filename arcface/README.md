# arcface
### TensortRT 8

The mxnet implementation is from [deepinsight/insightface.](https://github.com/deepinsight/insightface)

**Updated Pretrained Weights:** ArcFace-R100 [Insight Face Google Drive](https://drive.google.com/file/d/1Hc5zUfBATaXUgcU2haUNa7dcaZSw95h2/view)

---

**Previous Pre-trained models:** The pretrained models are from [LResNet50E-IR,ArcFace@ms1m-refine-v1](https://github.com/deepinsight/insightface/wiki/Model-Zoo#32-lresnet50e-irarcfacems1m-refine-v1), [LResNet100E-IR,ArcFace@ms1m-refine-v2](https://github.com/deepinsight/insightface/wiki/Model-Zoo#31-lresnet100e-irarcfacems1m-refine-v2) and [MobileFaceNet,ArcFace@ms1m-refine-v1](https://github.com/deepinsight/insightface/wiki/Model-Zoo#34-mobilefacenetarcfacems1m-refine-v1)

---

The two input images used in this project are joey0.ppm and joey1.ppm, download them from [Google Drive.](https://drive.google.com/drive/folders/1ctqpkRCRKyBZRCNwo9Uq4eUoMRLtFq1e). The input image is 112x112, and generated from `get_input()` in `insightface/deploy/face_model.py`, which is cropped and aligned face image.

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/83122953-f45f8d80-a106-11ea-84b0-4f6ff91b5924.jpg">
</p>

## Config

- FP16/FP32 can be selected by the macro `USE_FP16` in arcface-r50/r100/mobilefacenet.cpp
- GPU id can be selected by the macro `DEVICE` in arcface-r50/r100/mobilefacenet.cpp

## Run

1.Generate .wts file from mxnet implementation of pretrained model. The following example described how to generate arcface-r100.wts from mxnet implementation of LResNet100E-IR,ArcFace@ms1m-refine-v1.
```
git clone https://github.com/deepinsight/insightface
cd insightface
git checkout 3866cd77a6896c934b51ed39e9651b791d78bb57
cd deploy
// copy tensorrtx/arcface/gen_wts.py to here(insightface/deploy)
// download model-r100-ii.zip and unzip here(insightface/deploy)
python gen_wts.py
// a file 'arcface-r100.wts' will be generated.
// the master branch of insightface should work, if not, you can checkout 94ad870abb3203d6f31b049b70dd080dc8f33fca
// arcface-r50.wts/arcface-mobilefacenet.wts can be generated in similar way from mxnet implementation of LResNet50E-IR,ArcFace@ms1m-refine-v1/MobileFaceNet,ArcFace@ms1m-refine-v1 pretrained model.

```
2.Put .wts file into tensorrtx/arcface, build and run

```
cd tensorrtx/arcface
// download joey0.ppm and joey1.ppm, and put here(tensorrtx/arcface)
mkdir build
cd build
cmake ..
make
sudo ./arcface-r100 -s    // serialize model to plan file i.e. 'arcface-r100.engine'
sudo ./arcface-r100 -d    // deserialize plan file and run inference

or

sudo ./arcface-r50 -s   // serialize model to plan file i.e. 'arcface-r50.engine'
sudo ./arcface-r50 -d   // deserialize plan file and run inference


or

sudo ./arcface-mobilefacenet -s   // serialize model to plan file i.e. 'arcface-mobilefacenet.engine'
sudo ./arcface-mobilefacenet -d   // deserialize plan file and run inference
```

3.Check the output log, latency and similarity score.

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)
