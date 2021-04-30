# Step 1: Get checkpoints from mmaction2
# https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm
wget https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth

# Step 2: Convert pytorch checkpoints to TensorRT weights
python gen_wts.py tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth --out-filename ./tsm_r50_kinetics400_mmaction2.wts

# Step 3: Test Python API.
# 3.1 Skip this step since we use default settings.
# 3.2 Inference
# 3.2.1 Save local engine file to `./tsm_r50_kinetics400_mmaction2.trt`.
python tsm_r50.py \
    --tensorrt-weights ./tsm_r50_kinetics400_mmaction2.wts \
    --save-engine-path ./tsm_r50_kinetics400_mmaction2.trt

# 3.2.2 Predict the recognition result using a single video `demo.mp4`.
#       Should print `Result class id 6`, aka `arm wrestling`
# Download demo video
wget https://raw.githubusercontent.com/open-mmlab/mmaction2/master/demo/demo.mp4
# # use *.wts as input
# python tsm_r50.py --tensorrt-weights ./tsm_r50_kinetics400_mmaction2.wts \
#     --input-video ./demo.mp4
# use engine file as input
python tsm_r50.py --load-engine-path ./tsm_r50_kinetics400_mmaction2.trt \
    --input-video ./demo.mp4

# 3.2.3 Optional: Compare inference result with MMAction2 TSM-R50 model
#       Have to install MMAction2 First, please refer to https://github.com/open-mmlab/mmaction2/blob/master/docs/install.md
# pip3 install pytest-runner
# pip3 install mmcv
# pip3 install mmaction2
# # use *.wts as input
# python tsm_r50.py \
#     --tensorrt-weights ./tsm_r50_kinetics400_mmaction2.wts \
#     --test-mmaction2 \
#     --mmaction2-config mmaction2_tsm_r50_config.py \
#     --mmaction2-checkpoint tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth
# # use TensorRT engine as input
# python tsm_r50.py \
#     --load-engine-path ./tsm_r50_kinetics400_mmaction2.trt \
#     --test-mmaction2 \
#     --mmaction2-config mmaction2_tsm_r50_config.py \
#     --mmaction2-checkpoint tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth

# Step 4: Test Python API.
# 4.1 Skip this step since we use default settings.
# 4.2 Build CPP
mkdir build && cd build && cmake .. && make
# 4.3 Generate Engine file
./tsm_r50 -s
# 4.4 Get Predictions
./tsm_r50 -d
# 4.5 Compare C++ Results with Python Results
cd ..
python tsm_r50.py --test-cpp --tensorrt-weights ./tsm_r50_kinetics400_mmaction2.wts
