# PP-OCRv5 / PP-Structure TensorRT

Language: [English](#english) | [中文](#zh-cn)

<a id="english"></a>
## English

TensorRT C++/CUDA implementation for PP-OCRv5, PP-Structure document models, UVDoc, and PP-FormulaNet. It builds TensorRT engines from WTS weights and hand-written TensorRT network layers, then runs OCR, document analysis, table, document correction, and formula recognition inference.


### Model Source

Models come from official Baidu/PaddlePaddle PaddleOCR/PaddleX releases, not from a local cache path.

```text
https://paddlepaddle.github.io/PaddleX/3.3/en/support_list/models_list.html
https://github.com/PaddlePaddle/PaddleX/tree/develop/paddlex/configs/modules
https://github.com/PaddlePaddle/PaddleOCR
```

Official inference model base URL:

```text
https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/
```

Download all supported models:

```bash
mkdir -p official_models
cd official_models

wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_table_cls_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBlockLayout_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wired_table_cell_det_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wireless_table_cell_det_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_plus_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wired_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-L_infer.tar
```

Extract model packages:

```bash
for f in *_infer.tar; do
  d="${f%_infer.tar}"
  mkdir -p "${d}"
  tar -xf "${f}" -C "${d}"
done
```

The official packages include `inference.yml` in the extracted model directories. Use the recognition model YAML for OCR dictionaries and the FormulaNet YAML for tokenizer decoding:

```text
official_models/PP-OCRv5_mobile_rec/inference.yml
official_models/PP-OCRv5_server_rec/inference.yml
official_models/PP-FormulaNet_plus-L/inference.yml
```

For custom OCR recognition models, keep the exported `inference.yml` with `character_dict:`, or pass a plain `rec_dict.txt` file instead. For custom FormulaNet, keep the tokenizer YAML that contains `vocab:` and special token ids.

### WTS Export Environment

`gen_wts.py` does not depend on `paddleocr` or `paddlex`. It reads official Paddle inference directories directly and only needs:

```text
paddlepaddle or paddlepaddle-gpu
numpy
```

CPU Paddle is enough because WTS export uses `paddle.CPUPlace()`:

```bash
cd /home/algo/ppocrv5
python3 -m venv .venv-wts
source .venv-wts/bin/activate
python -m pip install -U pip
python -m pip install paddlepaddle==3.2.0 numpy==2.2.6
```

Check the export environment:

```bash
cd /home/algo/ppocrv5
source .venv-wts/bin/activate
python - <<'PY'
import paddle
import numpy
from paddle.static.pir_io import get_pir_parameters

print("paddle", paddle.__version__)
print("numpy", numpy.__version__)
print("pir_io ok", get_pir_parameters is not None)
PY
```

### Requirements

Required components are TensorRT, CUDA, OpenCV, CMake, and PaddlePaddle for WTS export. The active TensorRT/CUDA/OpenCV build paths and architecture are read from `CMakeLists.txt`; runtime constants such as TensorRT workspace cap, builder optimization level, OCR thresholds, OCR tensor names, and FormulaNet limits are in `include/config.h`.

Builds have been verified with TensorRT 10.13.3.9 and TensorRT 8.6.1.6. To use another TensorRT package, pass `-DTENSORRT_ROOT=/path/to/TensorRT-x.y.z` when configuring CMake.

If TensorRT runtime libraries are not in the shell environment, add the TensorRT library directory configured by the local `CMakeLists.txt` to `LD_LIBRARY_PATH`.

### Memory Limits

WTS export runs on CPU. GPU memory matters when converting `.wts` to `.engine`.

The C++ builders cap TensorRT workspace to `2048 MiB` by default and use builder optimization level `0`:

```text
TensorRT builder workspace: 2048 MiB, optimization level: 0
```

To lower workspace further:

```bash
export PPOCRV5_TRT_WORKSPACE_MB=1536
export PPOCRV5_BUILDER_OPT_LEVEL=0
```

Build engines one at a time under the 2 GB limit. TensorRT optimization profiles are fixed to batch 1.

### Optional INT8 Calibration

Default builds are FP32. To enable TensorRT INT8 calibration for one engine build, set a calibration image directory before running the normal `-s` command:

```bash
export PPOCRV5_INT8_CALIB_DIR=${PPOCRV5_SAMPLE_DIR}/ppocrv5_mobile_det
export PPOCRV5_INT8_CALIB_TABLE=ppocrv5_mobile_det_int8.cache
export PPOCRV5_INT8_READ_CACHE=0
./ppocrv5_ocr -s all_models/ppocrv5_mobile_det.wts all_models_engines/ppocrv5_mobile_det_int8.engine m
unset PPOCRV5_INT8_CALIB_DIR PPOCRV5_INT8_CALIB_TABLE PPOCRV5_INT8_READ_CACHE
```

OCR detection uses PP-OCR detection normalization, OCR recognition uses PP-OCR recognition normalization, and generic image models use fixed-size ImageNet-style normalization.

### Build

```bash
cd /home/algo/ppocrv5
mkdir -p build
cd build
cmake ..
cmake --build . -j8
```

Main binaries:

| Binary | Build mode | Run mode | Purpose |
| --- | --- | --- | --- |
| `ppocrv5_det` | `-s det.wts det.engine` | `-d det.engine image_or_dir` | Single OCR text detector engine build and detector-only visualization. |
| `ppocrv5_rec` | `-s rec.wts rec.engine` | `-d rec.engine image_or_dir rec_dict.txt|rec_inference.yml` | Single OCR recognizer engine build and line-image recognition. |
| `ppocrv5_ocr` | `-s model.wts model.engine m|s` | `-d det.engine rec.engine image_or_dir [dict|rec_inference.yml]` | OCR detector/recognizer engine build and full OCR pair inference. |
| `ppocrv5_model` | `-s model.wts model.engine` | `-d [model_name] model.engine image_or_dir` | Generic PP-Structure/UVDoc/table model engine build and tensor-summary validation. |
| `ppocrv5_formula` | `-s formula.wts formula.engine` | `-d formula.engine formula.decoder.engine image_or_dir inference.yml` | FormulaNet encoder/decoder engine build and LaTeX inference. |
| `ppocrv5_dump` | None | `-d model.engine shape out_prefix [input.bin]` | Tensor dump tool for parity regression. It does not build engines; use the model-specific `-s` binary first. |
| `ppocr_system` | None | Function modes such as `-ocr`, `-layout`, `-table`, `-formula`, `-all` | System-level inference entry that loads only the engines required by the selected function. |

Dictionary and tokenizer inputs:

| File | Used by | Required content |
| --- | --- | --- |
| `rec_dict.txt` | OCR recognition | Plain text dictionary, one character per line. |
| `rec_inference.yml` | OCR recognition | Official recognition config containing `character_dict:`. Only this character list is read. |
| `formula_inference.yml` | FormulaNet | Official tokenizer config containing `vocab:` and special token entries with `content` and `id`. They are used only to decode token IDs into LaTeX text. |

### Export WTS

Export all supported official models into `.wts` files.

```bash
cd /home/algo/ppocrv5
source .venv-wts/bin/activate
python gen_wts.py --official-model-dir /path/to/official_models --out-dir build/all_models
```

This exports the standard WTS names used by the engine build examples:

```text
ppocrv5_mobile_det.wts
ppocrv5_mobile_rec.wts
ppocrv5_server_det.wts
ppocrv5_server_rec.wts
pp_lcnet_x1_0_doc_ori.wts
pp_lcnet_x1_0_table_cls.wts
pp_lcnet_x1_0_textline_ori.wts
pp_docblocklayout.wts
pp_doclayout_plus_l.wts
rt_detr_l_wired_table_cell_det.wts
rt_detr_l_wireless_table_cell_det.wts
slanet_plus.wts
slanext_wired.wts
uvdoc.wts
pp_formulanet_plus_l.wts
```

Export one custom Paddle inference model. `--tag` becomes the WTS prefix and C++ registry model name.

```bash
python gen_wts.py \
  --model /path/to/Paddle/inference_model_dir \
  --tag my_model_name \
  --out-dir build/all_models
```

Single-model export is also supported. The final argument is the model type; short aliases such as `m_det`, `m_rec`, `s_det`, `s_rec`, `doc_ori`, `table_cls`, `textline_ori`, `docblock`, `layout`, `wired_table`, `wireless_table`, `slanet`, `slanext`, `uvdoc`, and `formula` are normalized to the standard model names.

```bash
python gen_wts.py -w /path/to/PP-OCRv5_mobile_det -o build/all_models/ppocrv5_mobile_det.wts m_det
python gen_wts.py -w /path/to/PP-OCRv5_mobile_rec -o build/all_models/ppocrv5_mobile_rec.wts m_rec
python gen_wts.py -w /path/to/PP-OCRv5_server_det -o build/all_models/ppocrv5_server_det.wts s_det
python gen_wts.py -w /path/to/PP-OCRv5_server_rec -o build/all_models/ppocrv5_server_rec.wts s_rec
python gen_wts.py -w /path/to/PP-LCNet_x1_0_doc_ori -o build/all_models/pp_lcnet_x1_0_doc_ori.wts doc_ori
python gen_wts.py -w /path/to/PP-LCNet_x1_0_table_cls -o build/all_models/pp_lcnet_x1_0_table_cls.wts table_cls
python gen_wts.py -w /path/to/PP-LCNet_x1_0_textline_ori -o build/all_models/pp_lcnet_x1_0_textline_ori.wts textline_ori
python gen_wts.py -w /path/to/PP-DocBlockLayout -o build/all_models/pp_docblocklayout.wts docblock
python gen_wts.py -w /path/to/PP-DocLayout_plus-L -o build/all_models/pp_doclayout_plus_l.wts layout
python gen_wts.py -w /path/to/RT-DETR-L_wired_table_cell_det -o build/all_models/rt_detr_l_wired_table_cell_det.wts wired_table
python gen_wts.py -w /path/to/RT-DETR-L_wireless_table_cell_det -o build/all_models/rt_detr_l_wireless_table_cell_det.wts wireless_table
python gen_wts.py -w /path/to/SLANet_plus -o build/all_models/slanet_plus.wts slanet
python gen_wts.py -w /path/to/SLANeXt_wired -o build/all_models/slanext_wired.wts slanext
python gen_wts.py -w /path/to/UVDoc -o build/all_models/uvdoc.wts uvdoc
python gen_wts.py -w /path/to/PP-FormulaNet_plus-L -o build/all_models/pp_formulanet_plus_l.wts formula
```

### Build Engines

Set low-memory TensorRT build options before building engines.

```bash
cd /home/algo/ppocrv5/build
mkdir -p all_models_engines
export PPOCRV5_TRT_WORKSPACE_MB=2048
export PPOCRV5_BUILDER_OPT_LEVEL=0
```

Validation samples are hosted in a separate repository to keep this module small:

```bash
git clone https://github.com/lindsayshuo/infer_pic.git /path/to/infer_pic
export PPOCRV5_SAMPLE_DIR=/path/to/infer_pic/ppocr/samples/validation
```

Each standard engine has a dedicated validation directory named after the engine basename, for example `${PPOCRV5_SAMPLE_DIR}/ppocrv5_mobile_det`. The root-level images in the sample repository are kept for whole-system validation and backward-compatible examples. The full mapping is also recorded in `samples/validation/README.md` and in the sample repository.

| Engine basename | Validation directory |
| --- | --- |
| `ppocrv5_mobile_det` | `samples/validation/ppocrv5_mobile_det` |
| `ppocrv5_mobile_rec` | `samples/validation/ppocrv5_mobile_rec` |
| `ppocrv5_server_det` | `samples/validation/ppocrv5_server_det` |
| `ppocrv5_server_rec` | `samples/validation/ppocrv5_server_rec` |
| `pp_lcnet_x1_0_doc_ori` | `samples/validation/pp_lcnet_x1_0_doc_ori` |
| `pp_lcnet_x1_0_table_cls` | `samples/validation/pp_lcnet_x1_0_table_cls` |
| `pp_lcnet_x1_0_textline_ori` | `samples/validation/pp_lcnet_x1_0_textline_ori` |
| `pp_docblocklayout` | `samples/validation/pp_docblocklayout` |
| `pp_doclayout_plus_l` | `samples/validation/pp_doclayout_plus_l` |
| `rt_detr_l_wired_table_cell_det` | `samples/validation/rt_detr_l_wired_table_cell_det` |
| `rt_detr_l_wireless_table_cell_det` | `samples/validation/rt_detr_l_wireless_table_cell_det` |
| `slanet_plus` | `samples/validation/slanet_plus` |
| `slanext_wired` | `samples/validation/slanext_wired` |
| `uvdoc` | `samples/validation/uvdoc` |
| `pp_formulanet_plus_l` | `samples/validation/pp_formulanet_plus_l` |
| `pp_formulanet_plus_l.decoder` | `samples/validation/pp_formulanet_plus_l.decoder` |

#### OCR Det/Rec

`ppocrv5_ocr -s` builds one WTS into one engine each time. The final letter selects the OCR network: `m` for mobile, `s` for server.

```bash
# ppocrv5_mobile_det: mobile OCR text detection engine used by `-ocr m`
./ppocrv5_ocr -s all_models/ppocrv5_mobile_det.wts all_models_engines/ppocrv5_mobile_det.engine m

# ppocrv5_mobile_rec: mobile OCR text recognition engine used by `-ocr m`
./ppocrv5_ocr -s all_models/ppocrv5_mobile_rec.wts all_models_engines/ppocrv5_mobile_rec.engine m

# ppocrv5_server_det: server OCR text detection engine used by `-ocr s`
./ppocrv5_ocr -s all_models/ppocrv5_server_det.wts all_models_engines/ppocrv5_server_det.engine s

# ppocrv5_server_rec: server OCR text recognition engine used by `-ocr s`
./ppocrv5_ocr -s all_models/ppocrv5_server_rec.wts all_models_engines/ppocrv5_server_rec.engine s
```

Validate the OCR engine pairs:

```bash
# mobile OCR detection + recognition validation
./ppocrv5_ocr -d all_models_engines/ppocrv5_mobile_det.engine all_models_engines/ppocrv5_mobile_rec.engine ${PPOCRV5_SAMPLE_DIR}/ppocrv5_mobile_det ../official_models/PP-OCRv5_mobile_rec/inference.yml

# server OCR detection + recognition validation
./ppocrv5_ocr -d all_models_engines/ppocrv5_server_det.engine all_models_engines/ppocrv5_server_rec.engine ${PPOCRV5_SAMPLE_DIR}/ppocrv5_server_det ../official_models/PP-OCRv5_server_rec/inference.yml
```

#### Generic Models

Build non-OCR engines from WTS plus C++ network-layer functions.

```bash
# pp_lcnet_x1_0_doc_ori: document orientation classification
./ppocrv5_model -s all_models/pp_lcnet_x1_0_doc_ori.wts all_models_engines/pp_lcnet_x1_0_doc_ori.engine

# pp_lcnet_x1_0_table_cls: table image classification
./ppocrv5_model -s all_models/pp_lcnet_x1_0_table_cls.wts all_models_engines/pp_lcnet_x1_0_table_cls.engine

# pp_lcnet_x1_0_textline_ori: textline orientation classification
./ppocrv5_model -s all_models/pp_lcnet_x1_0_textline_ori.wts all_models_engines/pp_lcnet_x1_0_textline_ori.engine

# pp_docblocklayout: coarse document block layout detection
./ppocrv5_model -s all_models/pp_docblocklayout.wts all_models_engines/pp_docblocklayout.engine

# pp_doclayout_plus_l: document layout detection
./ppocrv5_model -s all_models/pp_doclayout_plus_l.wts all_models_engines/pp_doclayout_plus_l.engine

# rt_detr_l_wired_table_cell_det: wired table cell detection
./ppocrv5_model -s all_models/rt_detr_l_wired_table_cell_det.wts all_models_engines/rt_detr_l_wired_table_cell_det.engine

# rt_detr_l_wireless_table_cell_det: wireless table cell detection
./ppocrv5_model -s all_models/rt_detr_l_wireless_table_cell_det.wts all_models_engines/rt_detr_l_wireless_table_cell_det.engine

# slanet_plus: table structure recognition
./ppocrv5_model -s all_models/slanet_plus.wts all_models_engines/slanet_plus.engine

# slanext_wired: wired table structure recognition
./ppocrv5_model -s all_models/slanext_wired.wts all_models_engines/slanext_wired.engine

# uvdoc: document unwarping/correction
./ppocrv5_model -s all_models/uvdoc.wts all_models_engines/uvdoc.engine
```

Validate each generic engine:

```bash
# pp_lcnet_x1_0_doc_ori: document orientation classification
./ppocrv5_model -d pp_lcnet_x1_0_doc_ori all_models_engines/pp_lcnet_x1_0_doc_ori.engine ${PPOCRV5_SAMPLE_DIR}/pp_lcnet_x1_0_doc_ori

# pp_lcnet_x1_0_table_cls: table image classification
./ppocrv5_model -d pp_lcnet_x1_0_table_cls all_models_engines/pp_lcnet_x1_0_table_cls.engine ${PPOCRV5_SAMPLE_DIR}/pp_lcnet_x1_0_table_cls

# pp_lcnet_x1_0_textline_ori: textline orientation classification
./ppocrv5_model -d pp_lcnet_x1_0_textline_ori all_models_engines/pp_lcnet_x1_0_textline_ori.engine ${PPOCRV5_SAMPLE_DIR}/pp_lcnet_x1_0_textline_ori

# pp_docblocklayout: coarse document block layout detection
./ppocrv5_model -d pp_docblocklayout all_models_engines/pp_docblocklayout.engine ${PPOCRV5_SAMPLE_DIR}/pp_docblocklayout

# pp_doclayout_plus_l: document layout detection
./ppocrv5_model -d pp_doclayout_plus_l all_models_engines/pp_doclayout_plus_l.engine ${PPOCRV5_SAMPLE_DIR}/pp_doclayout_plus_l

# rt_detr_l_wired_table_cell_det: wired table cell detection
./ppocrv5_model -d rt_detr_l_wired_table_cell_det all_models_engines/rt_detr_l_wired_table_cell_det.engine ${PPOCRV5_SAMPLE_DIR}/rt_detr_l_wired_table_cell_det

# rt_detr_l_wireless_table_cell_det: wireless table cell detection
./ppocrv5_model -d rt_detr_l_wireless_table_cell_det all_models_engines/rt_detr_l_wireless_table_cell_det.engine ${PPOCRV5_SAMPLE_DIR}/rt_detr_l_wireless_table_cell_det

# slanet_plus: table structure recognition
./ppocrv5_model -d slanet_plus all_models_engines/slanet_plus.engine ${PPOCRV5_SAMPLE_DIR}/slanet_plus

# slanext_wired: wired table structure recognition
./ppocrv5_model -d slanext_wired all_models_engines/slanext_wired.engine ${PPOCRV5_SAMPLE_DIR}/slanext_wired

# uvdoc: document unwarping/correction
./ppocrv5_model -d uvdoc all_models_engines/uvdoc.engine ${PPOCRV5_SAMPLE_DIR}/uvdoc
```

#### FormulaNet

Build FormulaNet recognition engines. One WTS produces two engines: encoder and decoder.

```bash
./ppocrv5_formula -s all_models/pp_formulanet_plus_l.wts all_models_engines/pp_formulanet_plus_l.engine

# outputs:
# all_models_engines/pp_formulanet_plus_l.engine
# all_models_engines/pp_formulanet_plus_l.decoder.engine
```

Validate FormulaNet and print LaTeX:

```bash
./ppocrv5_formula -d \
  all_models_engines/pp_formulanet_plus_l.engine \
  all_models_engines/pp_formulanet_plus_l.decoder.engine \
  ${PPOCRV5_SAMPLE_DIR}/pp_formulanet_plus_l \
  ../official_models/PP-FormulaNet_plus-L/inference.yml
```

### Verified Engines

Supported engine names:

```text
ppocrv5_mobile_det.engine
ppocrv5_mobile_rec.engine
ppocrv5_server_det.engine
ppocrv5_server_rec.engine
pp_lcnet_x1_0_doc_ori.engine
pp_lcnet_x1_0_table_cls.engine
pp_lcnet_x1_0_textline_ori.engine
pp_docblocklayout.engine
pp_doclayout_plus_l.engine
rt_detr_l_wired_table_cell_det.engine
rt_detr_l_wireless_table_cell_det.engine
slanet_plus.engine
slanext_wired.engine
uvdoc.engine
pp_formulanet_plus_l.engine
pp_formulanet_plus_l.decoder.engine
```

#### 2 GB Build Validation

The full engine set was rebuilt serially with:

```bash
export PPOCRV5_TRT_WORKSPACE_MB=2048
export PPOCRV5_BUILDER_OPT_LEVEL=0
```

Peak GPU memory samples:

```text
model                              peak_mib  delta_mib
ppocrv5_mobile_det                 1260      462
ppocrv5_mobile_rec                 1584      786
ppocrv5_server_det                 1656      858
ppocrv5_server_rec                 1670      872
pp_lcnet_x1_0_doc_ori              1584      786
pp_lcnet_x1_0_table_cls            1584      786
pp_lcnet_x1_0_textline_ori         1584      786
pp_docblocklayout                  1592      794
pp_doclayout_plus_l                1626      828
rt_detr_l_wired_table_cell_det     1612      814
rt_detr_l_wireless_table_cell_det  1702      904
slanet_plus                        1590      792
slanext_wired                      1940      1142
uvdoc                              1200      402
pp_formulanet_plus_l               1642      844
```

The `pp_formulanet_plus_l` row builds both encoder and decoder engines.

### Accuracy Regression

Tensor parity was checked against Paddle inference outputs from the official models during development. `ppocrv5_dump` remains available for deterministic TensorRT tensor dumps from built engines; engine build and inference remain C++/CUDA.

`ppocrv5_dump` intentionally has no `-s` mode. Build the engine with `ppocrv5_det`, `ppocrv5_rec`, `ppocrv5_ocr`, `ppocrv5_model`, or `ppocrv5_formula` first, then use `ppocrv5_dump -d` to export input/output tensors for regression checks.

Generate TensorRT dumps from `build`:

```bash
mkdir -p regression_latest

./ppocrv5_dump -d all_models_engines/ppocrv5_mobile_det.engine 1x3x960x960 regression_latest/ppocrv5_mobile_det
./ppocrv5_dump -d all_models_engines/ppocrv5_mobile_rec.engine 1x3x48x320 regression_latest/ppocrv5_mobile_rec
./ppocrv5_dump -d all_models_engines/ppocrv5_server_det.engine 1x3x960x960 regression_latest/ppocrv5_server_det
./ppocrv5_dump -d all_models_engines/ppocrv5_server_rec.engine 1x3x48x320 regression_latest/ppocrv5_server_rec
./ppocrv5_dump -d all_models_engines/pp_lcnet_x1_0_doc_ori.engine 1x3x224x224 regression_latest/pp_lcnet_x1_0_doc_ori
./ppocrv5_dump -d all_models_engines/pp_lcnet_x1_0_table_cls.engine 1x3x224x224 regression_latest/pp_lcnet_x1_0_table_cls
./ppocrv5_dump -d all_models_engines/pp_lcnet_x1_0_textline_ori.engine 1x3x80x160 regression_latest/pp_lcnet_x1_0_textline_ori
./ppocrv5_dump -d all_models_engines/uvdoc.engine image=1x3x800x800 regression_latest/uvdoc
./ppocrv5_dump -d all_models_engines/pp_docblocklayout.engine 'image=1x3x640x640;im_shape=1x2;scale_factor=1x2' regression_latest/pp_docblocklayout
./ppocrv5_dump -d all_models_engines/pp_doclayout_plus_l.engine 'image=1x3x800x800;im_shape=1x2;scale_factor=1x2' regression_latest/pp_doclayout_plus_l
./ppocrv5_dump -d all_models_engines/rt_detr_l_wired_table_cell_det.engine 'image=1x3x640x640;im_shape=1x2;scale_factor=1x2' regression_latest/rt_detr_l_wired_table_cell_det
./ppocrv5_dump -d all_models_engines/rt_detr_l_wireless_table_cell_det.engine 'image=1x3x640x640;im_shape=1x2;scale_factor=1x2' regression_latest/rt_detr_l_wireless_table_cell_det
./ppocrv5_dump -d all_models_engines/slanet_plus.engine 1x3x800x800 regression_latest/slanet_plus
./ppocrv5_dump -d all_models_engines/slanext_wired.engine 1x3x512x512 regression_latest/slanext_wired
```

Current regression result: all 14 tensor-parity models are `PASS`. The largest observed matched detection deviation is on `rt_detr_l_wired_table_cell_det` with `matched_max_abs=0.180969`, `matched_mean_abs=0.000945295`, and `bad_coord=0`. OCR, LCNet, UVDoc, SLANet, and SLANeXt outputs are within the configured cosine and absolute-error thresholds.

FormulaNet is validated end to end with decoded LaTeX:

```bash
./ppocrv5_formula -d \
  all_models_engines/pp_formulanet_plus_l.engine \
  all_models_engines/pp_formulanet_plus_l.decoder.engine \
  ${PPOCRV5_SAMPLE_DIR}/pp_formulanet_plus_l \
  ../official_models/PP-FormulaNet_plus-L/inference.yml
```

Expected validation output includes:

```text
tokens=60 latex=E=m c^{\wedge}2+a^{\wedge}2+b^{\wedge}2=c^{\wedge}2
```

Clean system validation logs can be kept under `build/regression_latest/`, for example `inference_smoke.txt`, `system_smoke.txt`, and `system_all_clean_smoke.txt`.

### Inference

`ppocr_system` is function-oriented. It does not keep every engine resident at startup. Each mode loads only the required engine or engine pair, runs the input image(s), and releases the session. Use `-all` mainly for validation or demos.

`engine_dir` is the directory path passed on the command line. In these examples, engines are written to:

```text
build/all_models_engines
```

When running commands from the `build` directory, use the relative path `all_models_engines`. Directory modes accept any engine directory with standard names. Explicit modes accept exact engine file paths.

#### System Modes

All modes accept one image path or an image directory.

| Mode | Engines loaded | Purpose | Output |
| --- | --- | --- | --- |
| `-ocr m engine_dir image_or_dir [dict|rec_inference.yml]` | `ppocrv5_mobile_det.engine`, `ppocrv5_mobile_rec.engine` | Mobile OCR detection + recognition for lower latency and smaller engines. | Boxes, text, confidence, visualization. |
| `-ocr s engine_dir image_or_dir [dict|rec_inference.yml]` | `ppocrv5_server_det.engine`, `ppocrv5_server_rec.engine` | Server OCR detection + recognition for higher accuracy. | Boxes, text, confidence, visualization. |
| `-ocr m|s det_model det.engine rec_model rec.engine image_or_dir [dict|rec_inference.yml]` | Explicit det/rec paths | OCR with custom paths or compatible self-trained models. | Same OCR output with model tags. |
| `-model model_name model.engine image_or_dir` | One explicit engine | Single-model debugging or custom model inference. | Tensor summaries. |
| `-classify engine_dir image_or_dir` | `pp_lcnet_x1_0_doc_ori.engine`, `pp_lcnet_x1_0_table_cls.engine`, `pp_lcnet_x1_0_textline_ori.engine` | Document orientation, table image classification, and textline orientation. | Classification summaries. |
| `-layout engine_dir image_or_dir` | `pp_docblocklayout.engine`, `pp_doclayout_plus_l.engine` | Coarse block layout and document layout detection. | Detection summaries. |
| `-table engine_dir image_or_dir` | `rt_detr_l_wired_table_cell_det.engine`, `rt_detr_l_wireless_table_cell_det.engine`, `slanet_plus.engine`, `slanext_wired.engine` | Wired/wireless table cell detection and table structure recognition. | Detection/structure summaries. |
| `-uvdoc engine_dir image_or_dir` | `uvdoc` | Document unwarping/correction. | Image-like tensor summaries. |
| `-formula engine_dir image_or_dir [inference.yml]` | FormulaNet encoder/decoder standard names | Formula recognition from a standard engine directory. | Decoded LaTeX. |
| `-formula model_name encoder.engine decoder.engine image_or_dir [inference.yml]` | Explicit FormulaNet paths | Formula recognition from custom paths. | Decoded LaTeX. |
| `-all m|s engine_dir image_or_dir [dict|rec_inference.yml] [formula_inference.yml]` | OCR, classify, layout, table, uvdoc, formula | Full validation/demo path. | Combined outputs. |

#### OCR Pair

Run the lower-level OCR binary directly with a det engine and rec engine.

```bash
./ppocrv5_ocr -d all_models_engines/ppocrv5_mobile_det.engine all_models_engines/ppocrv5_mobile_rec.engine ${PPOCRV5_SAMPLE_DIR}/ppocrv5_mobile_det ../official_models/PP-OCRv5_mobile_rec/inference.yml
```

Run OCR from a standard engine directory. `m` loads mobile engines; `s` loads server engines.

```bash
# mobile OCR
./ppocr_system -ocr m all_models_engines ${PPOCRV5_SAMPLE_DIR}/ppocrv5_mobile_det ../official_models/PP-OCRv5_mobile_rec/inference.yml

# server OCR
./ppocr_system -ocr s all_models_engines ${PPOCRV5_SAMPLE_DIR}/ppocrv5_server_det ../official_models/PP-OCRv5_server_rec/inference.yml
```

Run OCR with explicit model names and engine paths.

```bash
./ppocr_system -ocr s \
  ppocrv5_server_det /data/engines/ppocrv5_server_det.engine \
  ppocrv5_server_rec /data/engines/ppocrv5_server_rec.engine \
  ${PPOCRV5_SAMPLE_DIR}/ppocrv5_server_det ../official_models/PP-OCRv5_server_rec/inference.yml
```

#### Functional Groups

The examples below use `all_models_engines`, which means `build/all_models_engines` from the repo root, or `all_models_engines` when the current directory is `build`. Replace it with another engine directory if your engines are stored elsewhere.

```bash
# pp_lcnet_x1_0_doc_ori(document orientation) + pp_lcnet_x1_0_table_cls(table image classification) + pp_lcnet_x1_0_textline_ori(textline orientation)
./ppocr_system -classify all_models_engines ${PPOCRV5_SAMPLE_DIR}/pp_lcnet_x1_0_doc_ori

# pp_docblocklayout(coarse document block layout) + pp_doclayout_plus_l(document layout)
./ppocr_system -layout all_models_engines ${PPOCRV5_SAMPLE_DIR}/pp_doclayout_plus_l

# rt_detr_l_wired_table_cell_det(wired table cells) + rt_detr_l_wireless_table_cell_det(wireless table cells) + slanet_plus(table structure) + slanext_wired(wired table structure)
./ppocr_system -table all_models_engines ${PPOCRV5_SAMPLE_DIR}/rt_detr_l_wired_table_cell_det

# uvdoc(document unwarping/correction)
./ppocr_system -uvdoc all_models_engines ${PPOCRV5_SAMPLE_DIR}/uvdoc
```

#### Per-Model Explicit Path Examples

Replace `/data/engines` and `/data/models` with your own paths.

Mobile OCR detection + recognition.

```bash
./ppocr_system -ocr m \
  ppocrv5_mobile_det /data/engines/ppocrv5_mobile_det.engine \
  ppocrv5_mobile_rec /data/engines/ppocrv5_mobile_rec.engine \
  ${PPOCRV5_SAMPLE_DIR}/ppocrv5_mobile_det /data/models/PP-OCRv5_mobile_rec/inference.yml
```

Server OCR detection + recognition.

```bash
./ppocr_system -ocr s \
  ppocrv5_server_det /data/engines/ppocrv5_server_det.engine \
  ppocrv5_server_rec /data/engines/ppocrv5_server_rec.engine \
  ${PPOCRV5_SAMPLE_DIR}/ppocrv5_server_det /data/models/PP-OCRv5_server_rec/inference.yml
```

Document orientation classification.

```bash
./ppocr_system -model pp_lcnet_x1_0_doc_ori /data/engines/pp_lcnet_x1_0_doc_ori.engine ${PPOCRV5_SAMPLE_DIR}/pp_lcnet_x1_0_doc_ori
```

Table classification.

```bash
./ppocr_system -model pp_lcnet_x1_0_table_cls /data/engines/pp_lcnet_x1_0_table_cls.engine ${PPOCRV5_SAMPLE_DIR}/pp_lcnet_x1_0_table_cls
```

Textline orientation classification.

```bash
./ppocr_system -model pp_lcnet_x1_0_textline_ori /data/engines/pp_lcnet_x1_0_textline_ori.engine ${PPOCRV5_SAMPLE_DIR}/pp_lcnet_x1_0_textline_ori
```

PP-DocBlockLayout coarse layout detection.

```bash
./ppocr_system -model pp_docblocklayout /data/engines/pp_docblocklayout.engine ${PPOCRV5_SAMPLE_DIR}/pp_docblocklayout
```

PP-DocLayout_plus-L layout detection.

```bash
./ppocr_system -model pp_doclayout_plus_l /data/engines/pp_doclayout_plus_l.engine ${PPOCRV5_SAMPLE_DIR}/pp_doclayout_plus_l
```

Wired table cell detection.

```bash
./ppocr_system -model rt_detr_l_wired_table_cell_det /data/engines/rt_detr_l_wired_table_cell_det.engine ${PPOCRV5_SAMPLE_DIR}/rt_detr_l_wired_table_cell_det
```

Wireless table cell detection.

```bash
./ppocr_system -model rt_detr_l_wireless_table_cell_det /data/engines/rt_detr_l_wireless_table_cell_det.engine ${PPOCRV5_SAMPLE_DIR}/rt_detr_l_wireless_table_cell_det
```

SLANet_plus table structure recognition.

```bash
./ppocr_system -model slanet_plus /data/engines/slanet_plus.engine ${PPOCRV5_SAMPLE_DIR}/slanet_plus
```

SLANeXt_wired table structure recognition.

```bash
./ppocr_system -model slanext_wired /data/engines/slanext_wired.engine ${PPOCRV5_SAMPLE_DIR}/slanext_wired
```

UVDoc document correction.

```bash
./ppocr_system -model uvdoc /data/engines/uvdoc.engine ${PPOCRV5_SAMPLE_DIR}/uvdoc
```

FormulaNet recognition with explicit encoder and decoder engines.

```bash
./ppocr_system -formula pp_formulanet_plus_l \
  /data/engines/pp_formulanet_plus_l.engine \
  /data/engines/pp_formulanet_plus_l.decoder.engine \
  ${PPOCRV5_SAMPLE_DIR}/pp_formulanet_plus_l \
  /data/models/PP-FormulaNet_plus-L/inference.yml
```

#### Formula LaTeX

Run FormulaNet from a standard engine directory and print decoded LaTeX.

```bash
./ppocr_system -formula \
  all_models_engines \
  ${PPOCRV5_SAMPLE_DIR}/pp_formulanet_plus_l \
  ../official_models/PP-FormulaNet_plus-L/inference.yml
```

Run FormulaNet from explicit encoder/decoder paths.

```bash
./ppocr_system -formula pp_formulanet_plus_l \
  /data/engines/pp_formulanet_plus_l.engine \
  /data/engines/pp_formulanet_plus_l.decoder.engine \
  ${PPOCRV5_SAMPLE_DIR}/pp_formulanet_plus_l \
  ../official_models/PP-FormulaNet_plus-L/inference.yml
```

#### All System

End-to-end validation/demo for OCR, classify, layout, table, UVDoc, and FormulaNet. Generated visualization files matching `_ppocrv5_` or `_ppocr_system_` are ignored when scanning an input directory.

```bash
./ppocr_system -all m \
  all_models_engines \
  ${PPOCRV5_SAMPLE_DIR} \
  ../official_models/PP-OCRv5_mobile_rec/inference.yml \
  ../official_models/PP-FormulaNet_plus-L/inference.yml
```

Use `s` instead of `m` for server OCR. `-all` is a standard-directory mode; use explicit OCR, `-model`, and explicit FormulaNet commands when engines live at unrelated paths.

### Custom Models

For self-trained or server-side Paddle inference models:

1. Export WTS with `gen_wts.py --model /path/to/model --tag your_model_name`.
2. If the architecture matches an existing C++ network-layer function, build with `ppocrv5_model` or `ppocrv5_ocr`.
3. Run with explicit `ppocr_system` paths.

Run one custom non-OCR engine after registering a matching C++ network-layer function.

```bash
./ppocr_system -model your_model_name /path/to/your_model.engine image.jpg
```

Build and run a custom OCR pair that keeps the official mobile network. Change `m` to `s` for the official server network.

```bash
# custom_det: custom mobile-network OCR text detector
./ppocrv5_ocr -s custom_det.wts custom_det.engine m

# custom_rec: custom mobile-network OCR text recognizer
./ppocrv5_ocr -s custom_rec.wts custom_rec.engine m

# custom_det + custom_rec OCR pair
./ppocr_system -ocr m custom_det custom_det.engine custom_rec custom_rec.engine image.jpg /path/to/custom_rec/inference.yml
```

If the architecture changes, add or update the matching C++ network-layer function under `src/model.cpp` and reusable blocks under `src/block.cpp` before building the engine.

[Back to top](#pp-ocrv5--pp-structure-tensorrt)

<a id="zh-cn"></a>
## 中文

PP-OCRv5、PP-Structure 文档模型、UVDoc 和 PP-FormulaNet 的 TensorRT C++/CUDA 实现。支持从 WTS 权重和手写 TensorRT 网络层构建 engine，并运行 OCR、文档分析、表格、文档矫正和公式识别推理。


### 模型来源

模型来自 Baidu/PaddlePaddle 官方 PaddleOCR/PaddleX 发布版本。官方 inference 模型下载基地址：

```text
https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/
```

支持的标准 engine 名称：

```text
ppocrv5_mobile_det.engine
ppocrv5_mobile_rec.engine
ppocrv5_server_det.engine
ppocrv5_server_rec.engine
pp_lcnet_x1_0_doc_ori.engine
pp_lcnet_x1_0_table_cls.engine
pp_lcnet_x1_0_textline_ori.engine
pp_docblocklayout.engine
pp_doclayout_plus_l.engine
rt_detr_l_wired_table_cell_det.engine
rt_detr_l_wireless_table_cell_det.engine
slanet_plus.engine
slanext_wired.engine
uvdoc.engine
pp_formulanet_plus_l.engine
pp_formulanet_plus_l.decoder.engine
```

官方 `_infer.tar` 包解压后的模型目录里会带 `inference.yml`。OCR 识别使用 rec 模型目录里的 YAML，FormulaNet 使用公式模型目录里的 YAML：

```text
official_models/PP-OCRv5_mobile_rec/inference.yml
official_models/PP-OCRv5_server_rec/inference.yml
official_models/PP-FormulaNet_plus-L/inference.yml
```

自训练 OCR 识别模型可以保留导出目录里的 `inference.yml`，里面需要有 `character_dict:`；也可以直接传每行一个字符的 `rec_dict.txt`。自训练 FormulaNet 需要保留包含 `vocab:` 和 special token id 的 tokenizer YAML。

### 环境

`gen_wts.py` 不依赖 `paddleocr` 或 `paddlex`，只需要 PaddlePaddle 和 NumPy。WTS 导出使用 CPU 即可。

```bash
python3 -m venv .venv-wts
source .venv-wts/bin/activate
python -m pip install -U pip
python -m pip install paddlepaddle==3.2.0 numpy==2.2.6
```

TensorRT、CUDA、OpenCV、CMake 路径从本机 `CMakeLists.txt` 读取。默认 TensorRT workspace 上限是 `2048 MiB`，builder optimization level 是 `0`，适合按模型串行构建 engine。

已验证 TensorRT 10.13.3.9 和 TensorRT 8.6.1.6 可以编译通过。使用其他 TensorRT 目录时，在 CMake 配置阶段传入 `-DTENSORRT_ROOT=/path/to/TensorRT-x.y.z`。

### 编译和 WTS

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j8
```

主要二进制用途：

| 二进制 | 构建模式 | 运行模式 | 用途 |
| --- | --- | --- | --- |
| `ppocrv5_det` | `-s det.wts det.engine` | `-d det.engine image_or_dir` | 单独构建 OCR 检测 engine，并输出检测框可视化。 |
| `ppocrv5_rec` | `-s rec.wts rec.engine` | `-d rec.engine image_or_dir rec_dict.txt|rec_inference.yml` | 单独构建 OCR 识别 engine，并对文本行图片做识别。 |
| `ppocrv5_ocr` | `-s model.wts model.engine m|s` | `-d det.engine rec.engine image_or_dir [dict|rec_inference.yml]` | 构建 mobile/server OCR det 或 rec engine，并跑完整 OCR。 |
| `ppocrv5_model` | `-s model.wts model.engine` | `-d [model_name] model.engine image_or_dir` | 构建文档结构、表格、UVDoc 等通用模型，并输出 tensor 摘要。 |
| `ppocrv5_formula` | `-s formula.wts formula.engine` | `-d formula.engine formula.decoder.engine image_or_dir inference.yml` | 构建 FormulaNet encoder/decoder engine，并输出 LaTeX。 |
| `ppocrv5_dump` | 无 | `-d model.engine shape out_prefix [input.bin]` | tensor dump 回归工具，只读取已有 engine，不负责构建 engine。 |
| `ppocr_system` | 无 | `-ocr`、`-layout`、`-table`、`-formula`、`-all` 等功能模式 | 按功能串起多个 engine，运行时只加载当前功能需要的模型。 |

字典和 tokenizer 输入：

| 文件 | 使用场景 | 需要的内容 |
| --- | --- | --- |
| `rec_dict.txt` | OCR 识别 | 普通文本字典，每行一个字符。 |
| `rec_inference.yml` | OCR 识别 | 官方识别配置里的 `character_dict:` 字符列表，程序只读取这个列表。 |
| `formula_inference.yml` | FormulaNet | 官方 tokenizer 配置里的 `vocab:`，以及带 `content` 和 `id` 的 special token 条目，只用于把 token id 解码成 LaTeX 文本。 |

导出全部官方 WTS：

```bash
source .venv-wts/bin/activate
python gen_wts.py --official-model-dir /path/to/official_models --out-dir build/all_models
```

导出单个自训练或服务端模型：

```bash
python gen_wts.py --model /path/to/inference_model_dir --tag my_model_name --out-dir build/all_models
```

也可以使用单模型短参数接口，最后一个参数表示模型类型：

```bash
python gen_wts.py -w /path/to/PP-OCRv5_mobile_det -o build/all_models/ppocrv5_mobile_det.wts m_det
python gen_wts.py -w /path/to/PP-OCRv5_mobile_rec -o build/all_models/ppocrv5_mobile_rec.wts m_rec
python gen_wts.py -w /path/to/PP-OCRv5_server_det -o build/all_models/ppocrv5_server_det.wts s_det
python gen_wts.py -w /path/to/PP-OCRv5_server_rec -o build/all_models/ppocrv5_server_rec.wts s_rec
python gen_wts.py -w /path/to/PP-LCNet_x1_0_doc_ori -o build/all_models/pp_lcnet_x1_0_doc_ori.wts doc_ori
python gen_wts.py -w /path/to/PP-LCNet_x1_0_table_cls -o build/all_models/pp_lcnet_x1_0_table_cls.wts table_cls
python gen_wts.py -w /path/to/PP-LCNet_x1_0_textline_ori -o build/all_models/pp_lcnet_x1_0_textline_ori.wts textline_ori
python gen_wts.py -w /path/to/PP-DocBlockLayout -o build/all_models/pp_docblocklayout.wts docblock
python gen_wts.py -w /path/to/PP-DocLayout_plus-L -o build/all_models/pp_doclayout_plus_l.wts layout
python gen_wts.py -w /path/to/RT-DETR-L_wired_table_cell_det -o build/all_models/rt_detr_l_wired_table_cell_det.wts wired_table
python gen_wts.py -w /path/to/RT-DETR-L_wireless_table_cell_det -o build/all_models/rt_detr_l_wireless_table_cell_det.wts wireless_table
python gen_wts.py -w /path/to/SLANet_plus -o build/all_models/slanet_plus.wts slanet
python gen_wts.py -w /path/to/SLANeXt_wired -o build/all_models/slanext_wired.wts slanext
python gen_wts.py -w /path/to/UVDoc -o build/all_models/uvdoc.wts uvdoc
python gen_wts.py -w /path/to/PP-FormulaNet_plus-L -o build/all_models/pp_formulanet_plus_l.wts formula
```

### 构建 engine

OCR 的 `-s` 一次只构建一个 engine，最后的 `m` 或 `s` 选择 mobile/server 网络。

```bash
./ppocrv5_ocr -s all_models/ppocrv5_mobile_det.wts all_models_engines/ppocrv5_mobile_det.engine m
./ppocrv5_ocr -s all_models/ppocrv5_mobile_rec.wts all_models_engines/ppocrv5_mobile_rec.engine m
./ppocrv5_ocr -s all_models/ppocrv5_server_det.wts all_models_engines/ppocrv5_server_det.engine s
./ppocrv5_ocr -s all_models/ppocrv5_server_rec.wts all_models_engines/ppocrv5_server_rec.engine s
```

通用模型：

```bash
./ppocrv5_model -s all_models/pp_lcnet_x1_0_doc_ori.wts all_models_engines/pp_lcnet_x1_0_doc_ori.engine
./ppocrv5_model -s all_models/pp_doclayout_plus_l.wts all_models_engines/pp_doclayout_plus_l.engine
./ppocrv5_model -s all_models/rt_detr_l_wired_table_cell_det.wts all_models_engines/rt_detr_l_wired_table_cell_det.engine
./ppocrv5_model -s all_models/slanet_plus.wts all_models_engines/slanet_plus.engine
./ppocrv5_model -s all_models/uvdoc.wts all_models_engines/uvdoc.engine
```

FormulaNet 一个 WTS 会生成 encoder 和 decoder 两个 engine：

```bash
./ppocrv5_formula -s all_models/pp_formulanet_plus_l.wts all_models_engines/pp_formulanet_plus_l.engine
```

### 推理模式

`ppocr_system` 按功能加载 engine，不会启动时一次加载全部模型。`engine_dir` 是命令行传入的 engine 目录；在 `build` 目录下通常写 `all_models_engines`。验证图片放在独立样本仓库，先设置 `PPOCRV5_SAMPLE_DIR=/path/to/infer_pic/ppocr/samples/validation`。

```bash
./ppocr_system -ocr m all_models_engines ${PPOCRV5_SAMPLE_DIR}/ppocrv5_mobile_det ../official_models/PP-OCRv5_mobile_rec/inference.yml
./ppocr_system -ocr s all_models_engines ${PPOCRV5_SAMPLE_DIR}/ppocrv5_server_det ../official_models/PP-OCRv5_server_rec/inference.yml
./ppocr_system -classify all_models_engines ${PPOCRV5_SAMPLE_DIR}/pp_lcnet_x1_0_doc_ori
./ppocr_system -layout all_models_engines ${PPOCRV5_SAMPLE_DIR}/pp_doclayout_plus_l
./ppocr_system -table all_models_engines ${PPOCRV5_SAMPLE_DIR}/rt_detr_l_wired_table_cell_det
./ppocr_system -uvdoc all_models_engines ${PPOCRV5_SAMPLE_DIR}/uvdoc
./ppocr_system -formula all_models_engines ${PPOCRV5_SAMPLE_DIR}/pp_formulanet_plus_l ../official_models/PP-FormulaNet_plus-L/inference.yml
```

也可以传显式 engine 路径，适合自定义目录或自训练模型：

```bash
./ppocr_system -model pp_doclayout_plus_l /data/engines/pp_doclayout_plus_l.engine ${PPOCRV5_SAMPLE_DIR}/pp_doclayout_plus_l
./ppocr_system -formula pp_formulanet_plus_l /data/engines/pp_formulanet_plus_l.engine /data/engines/pp_formulanet_plus_l.decoder.engine ${PPOCRV5_SAMPLE_DIR}/pp_formulanet_plus_l /data/models/PP-FormulaNet_plus-L/inference.yml
```

### 精度回归

开发阶段已用官方 Paddle inference 模型做 tensor parity 对齐检查；`ppocrv5_dump` 保留为 C++ 侧 TensorRT tensor 导出工具。当前回归结果：14 个 tensor-parity 模型全部 `PASS`。

`ppocrv5_dump` 没有 `-s` 模式是刻意设计的：它不是 engine 构建入口，只负责对已经构建好的 engine 执行 `-d`，导出输入和输出 tensor 供精度回归使用。

FormulaNet 端到端验证输出：

```text
tokens=60 latex=E=m c^{\wedge}2+a^{\wedge}2+b^{\wedge}2=c^{\wedge}2
```

[Back to top](#pp-ocrv5--pp-structure-tensorrt)
