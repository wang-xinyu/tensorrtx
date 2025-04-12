# lenet5

lenet5 is the simplest net in this tensorrtx project. You can learn the basic procedures of building tensorrt app from API. Including `define network`, `build engine`, `set output`, `do inference`, `serialize model to file`, `deserialize model from file`, etc.

## TensorRT C++ API

```
// 1. generate lenet5.wts from https://github.com/wang-xinyu/pytorchx/tree/master/lenet

// 2. put lenet5.wts into tensorrtx/lenet

// 3. build and run

cd tensorrtx/lenet

mkdir build

cd build

cmake ..

make

sudo ./lenet -s   // serialize model to plan file i.e. 'lenet5.engine'

sudo ./lenet -d   // deserialize plan file and run inference

// 4. see if the output is same as pytorchx/lenet
```

## TensorRT Python API

```
# 1. generate lenet5.wts from https://github.com/wang-xinyu/pytorchx/tree/master/lenet

# 2. put lenet5.wts into tensorrtx/lenet

# 3. install Python dependencies (tensorrt/pycuda/numpy)

cd tensorrtx/lenet

python lenet.py -s   # serialize model to plan file, i.e. 'lenet5.engine'

python lenet.py -d   # deserialize plan file and run inference

# 4. see if the output is same as pytorchx/lenet
```


## Tripy (New TensorRT Python Programming Model)

1. Generate `lenet5.wts` from https://github.com/wang-xinyu/pytorchx/tree/master/lenet

2. Copy `lenet5.wts` into [tensorrtx/lenet](./)

3. Install Tripy:

    ```bash
    python3 -m pip install nvtripy -f https://nvidia.github.io/TensorRT-Incubator/packages.html
    ```

4. Change directories:

    ```bash
    cd tensorrtx/lenet
    ```

5. Compile and save the model:

    ```bash
    python3 lenet_tripy.py -s
    ```

6. Load and run the model:

    ```bash
    python3 lenet_tripy.py -d
    ```
