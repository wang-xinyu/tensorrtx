# MLP

MLP is the most basic net in this tensorrtx project for starters. You can learn the basic procedures of building TensorRT app from the provided APIs. The process of building a TensorRT engine explained in the chart below.

![TensorRT Image](https://user-images.githubusercontent.com/33795294/148565279-795b12da-5243-4e7e-881b-263eb7658683.jpg)

This demo creates a single-layer MLP with `TensorRT >= 7.x` version support.

## Helper Files

`logging.h` : A logger file for using NVIDIA TensorRT API (mostly same for all models)

`mlp.wts` : Converted weight file, can be generated from [pytorchx/mlp](https://github.com/wang-xinyu/pytorchx/tree/master/mlp), for mlp, it looks like:
```txt
2
linear.weight 1 3fff7e32
linear.bias 1 3c138a5a
```
(you can create `mlp.wts` and copy this content into it directly)

## TensorRT C++ API

see [HERE](../README.md#how-to-run)

## TensorRT Python API

1. Generate mlp.wts (from `pytorchx` or create on your own)

2. Put mlp.wts into tensorrtx/mlp (if using the generated weights)

3. Run
    ```bash
    cd tensorrtx/mlp
    python mlp.py -s   # serialize model to plan file, i.e. 'mlp.engine'
    python mlp.py -d   # deserialize plan file and run inference
    ```
