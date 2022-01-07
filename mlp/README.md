# MLP

MLP is the most basic net in this tensorrtx project for starters. You can learn the basic procedures of building
TensorRT app from the provided APIs. The process of building a TensorRT engine explained in the chart below.

![TensorRT Image](./imgs/model_creation.jpg?raw=true "")

## Helper Files

`logging.h` : A logger file for using NVIDIA TRT API
`mlp.wts` : Converted weight file (simple file, you can open and check it)

## TensorRT C++ API

```
// 1. generate mlp.wts from https://github.com/wang-xinyu/pytorchx/tree/master/mlp -- or use the given .wts file

// 2. put mlp.wts into tensorrtx/mlp (if using the generated weights)

// 3. build and run

cd tensorrtx/mlp

mkdir build

cd build

cmake ..

make

sudo ./mlp -s   // serialize model to plan file i.e. 'mlp.engine'

sudo ./mlp -d   // deserialize plan file and run inference

// 4. see if the output is same as pytorchx/mlp
```

## TensorRT Python API

```
# 1. Generate mlp.wts from https://github.com/wang-xinyu/pytorchx/tree/master/mlp -- or use the given .wts file

# 2. Put mlp.wts into tensorrtx/mlp (if using the generated weights)

# 3. Install Python dependencies (tensorrt/pycuda/numpy)

# 4. Run 
    
    cd tensorrtx/mlp
    
    python mlp.py -s   # serialize model to plan file, i.e. 'mlp.engine'
    
    python mlp.py -d   # deserialize plan file and run inference

# 5. See if the output is same as pytorchx/mlp
```

## Note
It also supports the latest CUDA-11.4 and TensorRT-8.2.x