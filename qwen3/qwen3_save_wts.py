# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2026/7/9 23:02
  @version V1.0
"""
import os
import random
import struct

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM  # noqa: F401
from transformers import __version__

assert __version__ == "4.57.6", "Please upgrade your transformers to version 4.57.6"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


# 调用
set_seed(42)

model_name = r"F:\LLM\modelscope\hub\models\Qwen\Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)  # type: Qwen3ForCausalLM
qwen3_model = model.model


def save_wts(m, path):
    with open(path, 'w') as f:
        f.write('{}\n'.format(len(m.state_dict().keys())))
        for k, v in m.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')
    print(f"saved {path}")


wts_file = "./wts/qwen3-0.6_embedding.wts"
embedding_model = qwen3_model.embed_tokens
save_wts(embedding_model, wts_file)
for i in range(len(qwen3_model.layers)):
    wts_file = f"./wts/qwen3-0.6_layer_{i}.wts"
    layer_model = qwen3_model.layers[i]
    save_wts(layer_model, wts_file)
wts_file = "./wts/qwen3-0.6_norm.wts"
save_wts(qwen3_model.norm, wts_file)
wts_file = "./wts/qwen3-0.6_lm_head.wts"
save_wts(model.lm_head, wts_file)
