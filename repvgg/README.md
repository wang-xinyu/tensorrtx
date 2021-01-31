# RepVGG

RepVGG models from
"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>

For the Pytorch implementation, you can refer to [DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)

# How to run

1. generate wts file.

```
git clone https://github.com/DingXiaoH/RepVGG.git
cd ReoVGG
```

You may convert a trained model into the inference-time structure with

```
python convert.py [weights file of the training-time model to load] [path to save] -a [model name]
```

For example,

```
python convert.py RepVGG-B2-train.pth RepVGG-B2-deploy.pth -a RepVGG-B2
```

Then copy `gen_wts.py` to `RepVGG` and generate .wts file, for example

```
python gen_wts.py -w RepVGG-B2-deploy.pth -s RepVGG-B2.wts
```

2. build and run
```
cd tensorrtx/repvgg

mkdir build

cd build

cmake ..

make

sudo ./repvgg -s RepVGG-B2  // serialize model to plan file i.e. 'RepVGG-B2.engine'
sudo ./repvgg -d RepVGG-B2  // deserialize plan file and run inference
```

