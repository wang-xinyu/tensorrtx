# Googlenet

## Introduction

GoogLeNet (Inception v1) model architecture from [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842). For model details, refer to code from [torchvision](https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py#L29), for generating `.wts` file, refer to [pytorchx/googlenet](https://github.com/wang-xinyu/pytorchx/tree/master/googlenet)

## Usage

1. use `gen_wts.py` to generate wts file.

```bash
python3 gen_wts.py
```

2. build C++ code

```bash
pushd tensorrtx/googlenet
cmake -S . -B build -G Ninja --fresh
cmake --build build
```

3. serialize wts model to engine file.

```bash
./build/googlenet -s
```

4. run inference

```bash
./build/googlenet -i
```

output looks like:

```bash
...
====
Execution time: 637us
-1.823, -0.9841, 0.6483, 0.7607, -0.4659, -1.407, -2.807, -1.175, -0.4034, -1.881, -1.267, -1.654, 0.7542, -1.777, -0.7118, -2.134, -1.542, 0.1852, -3.036, -0.5396, -0.1669,
====
prediction result:
Top: 0 idx: 285, logits: 9.9, label: Egyptian cat
Top: 1 idx: 281, logits: 8.304, label: tabby, tabby cat
Top: 2 idx: 282, logits: 6.859, label: tiger cat
```
