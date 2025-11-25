## Introduction

regnet supports for TensorRT-10.


## Environment

* cuda 12.6.85
* cudnn 9.6.0.74
* tensorrt 10.7.0.23
* opencv 4.10.0


## Build and Run(Python API and C++ API)

```
// 1. generate regnet_x_400mf.pth file and architecture file regnet_x_400mf_summary.txt by running save_pth_model.py
python save_pth_model.py

// 2. generate regnet_x_400mf.wts file by running save_wts_models.py
python save_wts_models.py

// 3. serialize model using python api.
python regnet.py -s

// 4. deserialize and run inference using python api.
python regnet.py -d

// 5. build and run using c++ api.
mkdir build

cd build

cmake ..

make

./regnet -s // serialize model.

./regnet -d // deserialize and run inference.

// 6. compare the result with pytorch model result.

```
