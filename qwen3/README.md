# qwen3-trtx

Deploy the Qwen3-0.6B model with TensorRT / TensorRTX style inference.

* ModelScope [Qwen3-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B)
* Hugging Face [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

## Features

- Build a TensorRT engine from exported Qwen3 `.wts` weights.
- Run interactive dialog inference from a serialized `.trt` engine.
- Support a custom `apply_rotary_pos_emb` TensorRT plugin.
- Support UTF-8 Chinese console input and UTF-8 dialog log output on Windows.
- Filter `<think>...</think>` content from model output.

## Project Initialization

After cloning the repository for the first time, initialize and sync the submodule:

```bash
git clone https://github.com/wang-xinyu/tensorrtx.git
cd tensorrtx/qwen3
# from https://github.com/wangzhaode/tokenizer.cpp
wget https://github.com/mpj1234/qwen3-trtx/releases/download/v1.0/tokenizer.cpp-linux-x86_64.tar.gz
tar -xzf tokenizer.cpp-linux-x86_64.tar.gz
```

## Build

Use CMake to configure and build the project, for example:

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

The executable supports two startup modes:

```bash
qwen3_trtx.exe -s [qwen3_wts_dir] [.trt]
qwen3_trtx.exe -d [.trt] [tokenizer_dir]
```

### 0. Export Qwen3 wts file

Download the Transformers library and export the. wts file of Qwen3.

Modify the `model_name` variable in the [qwen3_save_wts.py](qwen3_save_wts.py) script to be the qwen3 model name or path.

Modify the `wts_file` variable in the [qwen3_save_wts.py](qwen3_save_wts.py) script to be the qwen3 model name or path.

### 1. Serialize engine

Build a TensorRT engine from the Qwen3 `.wts` directory:

```bash
./qwen3_trtx -s ../wts/ ../models/qwen3.bf16.trt
```

Arguments:

- `qwen3_wts_dir`: directory containing exported Qwen3 `.wts` files
- `.trt`: output TensorRT engine path

### 2. Dialog inference

Load a TensorRT engine and enter interactive dialog mode:

```bash
./qwen3_trtx -d ../models/qwen3.bf16.trt ../Qwen3-0.6B/
```

Arguments:

- `.trt`: TensorRT engine path
- `tokenizer_dir`: tokenizer / model directory used by `tokenizer.cpp`; this argument is required

After startup:

- enter dialog text in the console
- type `exit` or `quit` to leave

## Dialog Behavior

- The system prompt is fixed to Chinese assistant mode.
- Assistant output is cleaned before display and logging.
- Dialog history is appended to `models/dialog_utf8.txt` in UTF-8 format.

## Notes

- `models/` is ignored by git by default, so generated engines and logs are not committed.
- On Windows, console input is handled in a UTF-8 compatible way for Chinese dialog input.
