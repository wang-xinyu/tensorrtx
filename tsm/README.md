# Temporal Shift Module

TSM-R50 from "TSM: Temporal Shift Module for Efficient Video Understanding" <https://arxiv.org/abs/1811.08383>

## How to run

### Tutorials

+ Step 1: Train/Download TSM-R50 checkpoints from [offical Github repo](https://github.com/mit-han-lab/temporal-shift-module) or [MMAction2](https://github.com/open-mmlab/mmaction2)
  + Mutable configs: `num_segments`, `shift_div`, `num_classes`.
  + Immutable config: `backbone`(ResNet50), `shift_place`(blockres), `temporal_pool`(False).

+ Step 2: Convert PyTorch checkpoints to TensorRT weights.

```shell
python gen_wts.py /path/to/pytorch.pth --out-filename /path/to/tensorrt.wts
```

+ Step 3: Modify mutable configs in `tsm_r50.py`.

```python
BATCH_SIZE = 2
NUM_SEGMENTS = 8
INPUT_H = 384
INPUT_W = 224
OUTPUT_SIZE = 400
SHIFT_DIV = 8
```

+ Step 4: Inference with `tsm_r50.py`.

### Script

```shell
# Step 1: Get checkpoints from mmaction2
# https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm
wget https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth

# Step 2: Convert checkpoints
python gen_wts.py tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth --out-filename ./tsm_r50_kinetics400_mmaction2.wts

# Step 3: Skip this step since we use default settings.

# Step 4: Inference
# 1) Save local engine file to `./tsm_r50_kinetics400_mmaction2.trt`.
python tsm_r50.py ./tsm_r50_kinetics400_mmaction2.wts \
    --engine-path ./tsm_r50_kinetics400_mmaction2.trt

# 2) Predict the recognition result using a single video `demo.mp4`.
#    Should print `Result class id 6`, aka `arm wrestling`
python tsm_r50.py ./tsm_r50_kinetics400_mmaction2.wts \
    --engine-path ./tsm_r50_kinetics400_mmaction2.trt \
    --input-video ./demo.mp4

# 3) Optional: Compare inference result with MMAction2 TSM-R50 model
#    Have to install MMAction2 First, https://github.com/open-mmlab/mmaction2/blob/master/docs/install.md
python tsm_r50.py ./tsm_r50_kinetics400_mmaction2.wts \
    --test-mmaction2 \
    --mmaction2-config mmaction2_tsm_r50_config.py \
    --mmaction2-checkpoint tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth
```

## TODO

+ [x] Python Shift module.
+ [x] Generate wts of official tsm and mmaction2 tsm.
+ [x] Python API Definition
+ [x] Test with mmaction2 demo
+ [x] Tutorial
+ [ ] C++ API Definition
