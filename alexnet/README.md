# alexnet

## Introduction

AlexNet model architecture comes from this paper: [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997). To generate `.wts` file, you can refer to [pytorchx/alexnet](https://github.com/wang-xinyu/pytorchx/tree/master/alexnet). To check the pytorch implementation of AlexNet, refer to [HERE](https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py#L17)

AlexNet consists of 3 major parts: features, adaptive average pooling, and classifier:

- features: just several stacked `CRP`(conv-relu-pool) and `CR` layers
- adaptive average pooling: pytorch can decide its inner parameters, but we need to calculate it ourselves in TensorRT API
- classifier: just several `fc-relu` layers. All layers can be implemented by tensorrt api, including `addConvolution`, `addActivation`, `addPooling`, `addMatrixMultiply`, `addElementWise` etc.

## Use AlexNet from PyTorch

We can use torchvision to load the pretrained alexnet model:

```python
alexnet = torchvision.models.alexnet(pretrained=True)
```

The model structure is:

```bash
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```

## Usage

1. use `gen_wts.py` to generate wts file.

```bash
python3 gen_wts.py
```

2. build C++ code

```bash
pushd tensorrtx/alexnet
cmake -S . -B build -G Ninja --fresh
cmake --build build
```

3. serialize wts model to engine file.

```bash
./build/alexnet -s
```

4. run inference

```bash
./build/alexnet -d
```

output looks like:

```txt
...
====
Execution time: 1ms
0.1234, -0.5678, ...
====
prediction result:
Top: 0 idx: 285, logits: 9.9, label: Egyptian cat
Top: 1 idx: 281, logits: 8.304, label: tabby, tabby cat
Top: 2 idx: 282, logits: 6.859, label: tiger cat
```

## FAQ

### How to align the output with Pytorch?

If your output is different from pytorch, you have to check which TensorRT API or your code cause this. A simple solution would be check the `.engine` output part by part, e.g., you can set the early layer of alexnet as output:

```c++
fc3_1->getOutput(0)->setName(OUTPUT_NAME);
network->markOutput(*pool3->getOutput(0)); // original is: "*fc3_1->getOutput(0)"
```

For this line of code, i use the output from "feature" part of alexnet, ignoring the rest of the model, then, don't forget to change the `OUTPUT_SIZE` macro on top of the file, lastly, build the `.engine` file to apply the changes.

You can sum up all output from C++ code, and compare it with Pytorch output, for Pytorch, you can do this by: `torch.sum(x)` at debug phase. The ideal value deviation between 2 values would be $[10^{-1}, 10^{-2}]$, for this example, since the output elements for "feature" is $256 * 6 * 6$ (bacth = 1), the final error would roughly be $10^{-4}$.

Note: This is a quick check, for more accurate check, you have to save the output tensor into a file to compare them value by value, but this situation is rare.
