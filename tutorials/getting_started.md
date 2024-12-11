# Getting Started with TensorRTx

## 1. Setup the development environment

(**RECOMMENDED**) If you prefer to run everything in a docker container, check [HERE](../docker/README.md)

If you prefer to install every dependencies locally, check [HERE](./install.md)

## 2. Run TensorRTx demo

It is recommended to go through the [lenet5](https://github.com/wang-xinyu/tensorrtx/tree/master/lenet) or [mlp](https://github.com/wang-xinyu/tensorrtx/tree/master/mlp) first. But if you are proficient in TensorRT, please check the readme file of the model you want directly.

We use "lenet5" to explain how we build DL network with TensorRT API.

### 2.1. Export lenet5 weights in pytorch

1. Clone the [wang-xinyu/pytorchx](https://github.com/wang-xinyu/pytorchx) in your machine, then enter lenet folder:

    ```bash
    pip install torch
    git clone https://github.com/wang-xinyu/pytorchx
    cd pytorchx/lenet
    ```

2. Run lenet5.py to generate lenet5.pth which is the pytorch serialized model. The lenet5 arch is defined in lenet5.py.

    ```bash
    python lenet5.py
    ```

3. Run inference.py to generate lenet5.wts, which is weights file for tensorrt.

    ```bash
    python inference.py
    ```

The terminal output would be like:
```txt
the output of lenet5 is [[0.0950, 0.0998, 0.1101, 0.0975, 0.0966, 0.1097, 0.0948, 0.1056, 0.0992, 0.0917]], shape is [1, 10].

cuda device count:  2
input:  torch.Size([1, 1, 32, 32])
conv1 torch.Size([1, 6, 28, 28])
pool1:  torch.Size([1, 6, 14, 14])
conv2 torch.Size([1, 16, 10, 10])
pool2 torch.Size([1, 16, 5, 5])
view:  torch.Size([1, 400])
fc1:  torch.Size([1, 120])
lenet out: tensor([[0.0950, 0.0998, 0.1101, 0.0975, 0.0966, 0.1097, 0.0948, 0.1056, 0.0992,
         0.0917]], device='cuda:0', grad_fn=<SoftmaxBackward>)
```

### 2.2. Run lenet5 in TensorRT

Clone the wang-xinyu/tensorrtx in your machine. Enter lenet folder, copy lenet5.wts generated above, and cmake&make c++ code.

And of course you should install cuda/cudnn/tensorrt first. You might need to adapt the tensorrt path in CMakeLists.txt if you install tensorrt from tar package.

```bash
git clone https://github.com/wang-xinyu/tensorrtx
cd tensorrtx/lenet
cp [PATH-OF-pytorchx]/pytorchx/lenet/lenet5.wts .
cmake -S . -B build
cd build
make
```

If the `make` succeed, the executable `lenet` will generated.

Run lenet to build tensorrt engine and serialize it to file `lenet5.engine`.

```bash
./lenet -s
```

Deserialize the engine and run inference.

```bash
./lenet -d
```

You should see the output like this,

```txt
Output:

0.0949623, 0.0998472, 0.110072, 0.0975036, 0.0965564, 0.109736, 0.0947979, 0.105618, 0.099228, 0.0916792,
```

## 3. Compare the two output

As the input to pytorch and tensorrt are same, i.e. a [1,1,32,32] all ones tensor.

So the output should be same, otherwise there must be something wrong.

```txt
The pytorch output is
0.0950, 0.0998, 0.1101, 0.0975, 0.0966, 0.1097, 0.0948, 0.1056, 0.0992, 0.0917

The tensorrt output is
0.0949623, 0.0998472, 0.110072, 0.0975036, 0.0965564, 0.109736, 0.0947979, 0.105618, 0.099228, 0.0916792
```

Same! exciting, isn't it?

## 4. The `.wts` content format

The `.wts` is plain text file, e.g. `lenet5.wts`, part of the contents are:

```txt
10
conv1.weight 150 be40ee1b bd20bab8 bdc4bc53 ...
conv1.bias 6 bd327058 ...
conv2.weight 2400 3c6f2220 3c693090 ...
conv2.bias 16 bd183967 bcb1ac8a ...
fc1.weight 48000 3c162c20 bd25196a ...
fc1.bias 120 3d3c3d49 bc64b948 ...
fc2.weight 10080 bce095a4 3d33b9dc ...
fc2.bias 84 bc71eaa0 3d9b276c ...
fc3.weight 840 3c252870 3d855351 ...
fc3.bias 10 bdbe4bb8 3b119ee0 ...
...
```

The first line is a number, indicate how many lines it has, excluding itself.

And then each line is

`[weight name] [value count = N] [value1] [value2], ..., [valueN]`

The value is in HEX format.

## 5. Frequently Asked Questions (FAQ)

check [HERE](./faq.md) for the answers of questions you may encounter.
