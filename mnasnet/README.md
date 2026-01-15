# mnasnet

MNASNet with depth multiplier of 0.5 from
"MnasNet: Platform-Aware Neural Architecture Search for Mobile" <https://arxiv.org/pdf/1807.11626.pdf>

For the Pytorch implementation, you can refer to [pytorchx/mnasnet](https://github.com/wang-xinyu/pytorchx/tree/master/mnasnet)

Following tricks are used in this mnasnet, nothing special, group conv and batchnorm are used.

- Batchnorm layer, implemented by scale layer.

## Usage

1. use `gen_wts.py` to generate wts file

```bash
python gen_wts.py
```

2. build C++ code

```bash
pushd tensorrtx/mnasnet
cmake -S . -B build -G Ninja --fresh
cmake --build build
```

3. serialize wts model to engine file

```bash
./build/mnasnet -s
```

4. run inference

```bash
./build/mnasnet -d
```

The output looks like:

```bash
...
====
Execution time: 0ms
-2.024, -1.266, -1.602, -1.465, -0.7756, -0.2096, 0.05945, 1.342, -0.2382, 1.279, 1.251, 0.2579, 1.836, -0.5296, 0.3196, 0.9055, -0.4915, 0.1604, -0.6305, -0.1019, -0.8816,
====
prediction result:
Top: 0 idx: 285, logits: 4.869, label: Egyptian cat
Top: 1 idx: 281, logits: 4.837, label: tabby, tabby cat
Top: 2 idx: 282, logits: 4.019, label: tiger cat
```
