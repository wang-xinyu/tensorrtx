# lenet5

lenet5 is one of the simplest net in this repo. You can learn the basic procedures of building CNN from TensorRT API. This demo includes 2 major steps:

1. Build engine
   - define network
   - set input/output
   - serialize model to `.engine` file
2. Do inference
   - load and deserialize model from `.engine` file
   - run inference

## Usage

1. download pt model from `https://github.com/SunnyHaze/LeNet5-MNIST-Pytorch/blob/main/model.pt`

2. run `gen_wts.py` to generate `.wts` file

```bash
python3 gen_wts.py
```

output looks like:

```bash
lenet out shape: torch.Size([1, 10])
lenet out: [tensor([0.0725, 0.0730, 0.1056, 0.1201, 0.1059, 0.0741, 0.1328, 0.0953, 0.1230,
        0.0975])]
inference result: 6
```

3. build C++ code

```bash
cd tensorrtx/lenet
cmake -S . -B build
cmake --build build
```

4. serialize wts model to engine file

```bash
./build/lenet -s
```

5. run inference

```bash
./build/lenet -d
```

output looks like:

```bash
...
Execution time: 32us
0.09727, 0.09732, 0.1005, 0.102, 0.1006, 0.09743, 0.1033, 0.09951, 0.1023, 0.09973,
====
Execution time: 33us
0.09727, 0.09732, 0.1005, 0.102, 0.1006, 0.09743, 0.1033, 0.09951, 0.1023, 0.09973,
====
prediction result:
Top: 0 idx: 6, logits: 0.1033, label: 6
Top: 1 idx: 8, logits: 0.1023, label: 8
Top: 2 idx: 3, logits: 0.102, label: 3
```

## Tripy (New TensorRT Python Programming Model)

1. Generate `lenet5.wts`

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
