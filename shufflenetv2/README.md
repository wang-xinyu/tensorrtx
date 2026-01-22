# shufflenet v2

ShuffleNetV2 with 0.5x output channels, as described in: [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

Following tricks are used in this demo:

- `torch.chunk` is used in shufflenet v2. We implemented the `chunk(2, dim=C)` by tensorrt plugin. Which is the simplest plugin in this tensorrtx project. You can learn the basic procedures of build tensorrt plugin.
- shuffle layer is used, the `channel_shuffle()` in `pytorchx/shufflenet` can be implemented by two shuffle layers in tensorrt.
- Batchnorm layer, implemented by scale layer.

## Usage

1. use `gen_wts.py` to generate wts file.

```bash
python3 gen_wts.py
```

2. build C++ code

```bash
pushd tensorrtx/shufflenetv2
cmake -S . -B build -G Ninja --fresh
cmake --build build
```

3. serialize wts model to engine file.

```bash
./build/shufflenetv2 -s
```

4. run inference

```bash
./build/shufflenetv2 -i
```

The inference output looks like:

```bash
...
328us
-5.481, -0.1151, 4.004, -1.47, 1.007, -5.943, -2.311, 1.708, 1.569, 0.3112, 1.589, 0.1816, -2.253, -3.261, -3.269, -0.9116, -2.132, -1.159, -2.108, -0.3869, -4.653,
====
...
prediction result:
Top: 0 idx: 285, logits: 10.44, label: Egyptian cat
Top: 1 idx: 309, logits: 10.19, label: bee
Top: 2 idx: 94, logits: 9.399, label: hummingbird
```
