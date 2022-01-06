# MLP

MLP is the most basic net in this tensorrtx project for starters. You can learn the basic procedures of building
TensorRT app from the provided APIs. The process of building a TensorRT engine explained in the chart below.

![TensorRT Image](./imgs/model_creation.jpg?raw=true "")

## TensorRT C++ API

```

```

## TensorRT Python API

```
# 1. Generate mlp.wts from https://github.com/wang-xinyu/pytorchx/tree/master/mlp

# 2. Put mlp.wts into tensorrtx/mlp

# 3. Install Python dependencies (tensorrt/pycuda/numpy)

# 4. Run 
    
    cd tensorrtx/mlp
    
    python mlp.py -s   # serialize model to plan file, i.e. 'mlp.engine'
    
    python mlp.py -d   # deserialize plan file and run inference

# 5. See if the output is same as pytorchx/mlp
```

