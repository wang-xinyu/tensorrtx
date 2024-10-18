# GhostNet

GhostNetv1 architecture is from the paper "GhostNet: More Features from Cheap Operations" [(https://arxiv.org/abs/1911.11907)](https://arxiv.org/abs/1911.11907).
GhostNetv2 architecture is from the paper "GhostNetV2: Enhance Cheap Operation with Long-Range Attention" [(https://arxiv.org/abs/2211.12905)](https://arxiv.org/abs/2211.12905).

For the PyTorch implementations, you can refer to [huawei-noah/ghostnet](https://github.com/huawei-noah/ghostnet).

Both versions use the following techniques in their TensorRT implementations:

- **BatchNorm** layer is implemented by TensorRT's **Scale** layer.
- **Ghost Modules** are used to generate more features from cheap operations, as described in the paper.
- Replacing `IPoolingLayer` with `IReduceLayer` in TensorRT for Global Average Pooling. The `IReduceLayer` allows you to perform reduction operations (such as sum, average, max) over specified dimensions without being constrained by the kernel size limitations of pooling layers.

## Project Structure

```plaintext
ghostnet
│
├── ghostnetv1
│   ├── CMakeLists.txt
│   ├── gen_wts.py
│   ├── ghostnetv1.cpp
│   └── logging.h
│
├── ghostnetv2
│   ├── CMakeLists.txt
│   ├── gen_wts.py
│   ├── ghostnetv2.cpp
│   └── logging.h
│
└── README.md
```

## Steps to use GhostNet in TensorRT

### 1. Generate `.wts` files for both GhostNetv1 and GhostNetv2

```bash
# For ghostnetv1
python ghostnetv1/gen_wts.py

# For ghostnetv2
python ghostnetv2/gen_wts.py
```

### 2. Build the project

```bash
cd tensorrtx/ghostnet
mkdir build
cd build
cmake ..
make
```

### 3. Serialize the models to engine files

Use the following commands to serialize the PyTorch models into TensorRT engine files (`ghostnetv1.engine` and `ghostnetv2.engine`):

```bash
# For ghostnetv1
sudo ./ghostnetv1 -s

# For ghostnetv2
sudo ./ghostnetv2 -s
```

### 4. Run inference using the engine files

Once the engine files are generated, you can run inference with the following commands:

```bash
# For ghostnetv1
sudo ./ghostnetv1 -d

# For ghostnetv2
sudo ./ghostnetv2 -d
```

### 5. Verify output

Compare the output with the PyTorch implementation from [huawei-noah/ghostnet](https://github.com/huawei-noah/ghostnet) to ensure that the TensorRT results are consistent with the PyTorch model.
