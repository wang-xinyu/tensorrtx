# squeezenet v1.1

SqueezeNet 1.1 model from the official SqueezeNet repo
<https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>

SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
than SqueezeNet 1.0, without sacrificing accuracy.

For the Pytorch implementation, you can refer to [pytorchx/squeezenet](https://github.com/wang-xinyu/pytorchx/tree/master/squeezenet)

## Usage

1. use `gen_wts.py` to generate wts file

```bash
python3 gen_wts.py
```

2. build C++ code

```bash
pushd tensorrtx/squeezenet
cmake -S . -B build -G Ninja --fresh
cmake --build build
```

3. serialize wts model to engine file

```bash
./build/squeezenet -s
```

4. run inference

```bash
./build/squeezenet -d
```

output looks like:

```bash
...
====
Execution time: 183us
3.481, 3.901, 4.438, 4.346, 3.3, 6.519, 6.03, 10.89, 10.45, 10.39, 8.874, 5.889, 9.529, 3.703, 5.865, 6.982, 8.894, 7.76, 4.599, 7.89, 4.795,
====
prediction result:
Top: 0 idx: 281, logits: 25.18, label: tabby, tabby cat
Top: 1 idx: 282, logits: 23.2, label: tiger cat
Top: 2 idx: 309, logits: 22.72, label: bee
```
