# GhostNet

GhostNet architecture from "GhostNet: More Features from Cheap Operations" [(https://arxiv.org/abs/1911.11907)](https://arxiv.org/abs/1911.11907).

For the PyTorch implementation, you can refer to [huawei-noah/ghostnet](https://github.com/huawei-noah/ghostnet).

Following tricks are used in this GhostNet implementation:

- **BatchNorm** layer is implemented by TensorRT's **Scale** layer.
- **Ghost Modules** are used to generate more features from cheap operations, as described in the paper.
- Replacing `IPoolingLayer` with `IReduceLayer` in TensorRT for Global Average Pooling.The `IReduceLayer` allows you to perform reduction operations (such as sum, average, max) over specified dimensions without being constrained by the kernel size limitations of pooling layers.

## Steps to use GhostNet in TensorRT

### 1. Generate `ghostnet.wts` 

```shell
python gen_wts.py
```



### 2. build

```bash
cd tensorrtx/ghostnet
mkdir build
cd build
cmake ..
make
```



### 3. Serialize the model to an engine file.

Use the following command to serialize the PyTorch model into a TensorRT engine file (`ghostnet.engine`):

```bash
sudo ./ghostnet -s   # Serialize model to plan file i.e. 'ghostnet.engine'
```

### 4. Run inference using the engine file.

Once the engine file is generated, you can run inference with the following command:

```bash
sudo ./ghostnet -d   # Deserialize plan file and run inference
```

### 5. Verify output.

Compare the output with the PyTorch implementation from [huawei-noah/ghostnet](https://github.com/huawei-noah/ghostnet) to ensure that the TensorRT results are consistent with the PyTorch model.

