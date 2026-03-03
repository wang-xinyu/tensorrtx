# Vision Transformers (ViT)

## 1. Overview

This is a handwritten TensorRT implementation of the Vision Transformers[arxiv.org.2010.11929](https://arxiv.org/abs/2010.11929) paper.

**Note**:

- Swi-GeLU activation layer is supported since TensorRT **10.0**+ SDK, we can use a approximation way as TensorRT does, check below for details.

## 2. Details

### 2.1 Features

- Support TensorRT SDK 8.5.1+ ~ 10.15.1+
- Support Windows11 OS
- Support native or self-implemented Swi-GeLU
- Support native or self-implemented multihead self-attention
- Support a dummy profiler by default
- Support a dummy output allocator by default
- Use optimization profile by default

### 2.2 Current limitations

- cannot use `IAttenion` with TensorRT SDK 10.14 ~ 10.15 because of the bugs in TensorRT
- TensorRT < 8 is not supported because some ops are not inplemented in cuDNN
- SM < 86, TensorRT < 10, CUDA < 12 cases are _NOT_ fully tested yet

### 2.3 Usage

1. use `gen_wts.py` to generate `.wts` file.

```bash
python gen_wts.py
```

2. build C++ code

```bash
pushd tensorrtx/vit
cmake -S . -B build -G Ninja --fresh
cmake --build build
```

3. serialize `.wts` model to engine file.

```bash
./build/vit -s
```

4. run inference

```bash
./build/vit -d
```

On **RTX 4080, TensorRT 10.15.1 SDK**, the output looks like:

```bash
...
====
1880us
-1.125, 0.4623, -0.1215, -0.007384, -0.004307, -0.7021, -0.748, 0.2031, -0.4862, -0.008939, -1.151, -0.408, -0.3259, 0.2202, 0.04537, -2.008, -0.2832, 0.04394, 0.5326, 0.1724, 0.5655,
====
prediction result:
Top: 0 idx: 285, logits: 8.262, label: Egyptian cat
Top: 1 idx: 281, logits: 7.872, label: tabby, tabby cat
Top: 2 idx: 282, logits: 6.477, label: tiger cat
========== VisionTransformerProfiler ==========
                                                                                                          TensorRT layer name    Runtime, %  Invocations Runtime, ms
                                                                  Reformatting CopyNode for Input Tensor 0 to patch embedding          3.2%           20         0.95
                                                                                                              patch embedding          1.5%           20         0.45
Reformatting CopyNode for Input Tensor 0 to {ForeignNode[(Unnamed Layer* 3) [Constant]...(Unnamed Layer* 518) [ElementWise]]}          0.2%           20         0.06
                                                                                                        __myl_ReshTran_myl3_0          0.8%           20         0.24
                                                                __myl_ConcAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_myl3_1          0.3%           20         0.08
                vit.encoder.layer.0.attentionvalue+vit.encoder.layer.0.attentionkey+vit.encoder.layer.0.attentionquery_myl3_2          1.4%           20         0.40
                                                                                                    __myl_TranReshMove_myl3_3          0.2%           20         0.06
                                                                                                    __myl_TranReshMove_myl3_4          0.2%           20         0.07
                                                                                                    __myl_TranReshMove_myl3_5          0.2%           20         0.06
                                                                                                          _gemm_mha_v2_myl3_6          0.5%           20         0.14
                                                                                                    __myl_MoveReshTran_myl3_7          0.2%           20         0.06
...
========== VisionTransformerProfiler total runtime = 29.67 ms ==========
```

as is shown above, we successfully triggered the internal MHA fused kernel fusion pass inside TensorRT (i.e., **"Myelin"** or **"myl"** in short), especially the MHA fused kernel: `_gemm_mha_v2_myl3_6`.

## 3. transformer details

`ViTLayer()` builds one ViT encoder block (Transformer encoder layer) using TensorRT primitives. The implementation corresponds to a **Pre-LayerNorm** Transformer layer (typical for ViT), including:

- LayerNorm before attention
- Multi-Head Self-Attention (MHSA): QKV projections → scaled dot-product attention → output projection
- Residual connection
- LayerNorm after attention
- Feed-Forward Network (FFN / MLP): dense → GeLU → dense
- Residual connection

The function returns the final residual output tensor.

### 3.1 Notation and Tensor Shapes

Let the input tensor (TensorRT `input`) be:

$$
\mathbf{X} \in \mathbb{R}^{N \times L \times D}
$$

Where:

- (N): batch size (represented by `N` in your code)
- (L): sequence length (number of tokens; dynamic in code via `-1`)
- (D): hidden size, fixed at 768 in this implementation

The attention head configuration:

$$
H = \tt{param.head\_num}, \qquad d = \frac{D}{H}
$$

### 3.2 Weight shapes (conceptual)

For a standard Transformer block:

- Q/K/V projection weights:
  $$
  \mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times D}
  $$
- Q/K/V biases (**NOTE**:Not used by native nvidia interface):
  $$
  \mathbf{b}_Q, \mathbf{b}_K, \mathbf{b}_V \in \mathbb{R}^{D}
  $$
- Output projection:
  $$
  \mathbf{W}_O \in \mathbb{R}^{D \times D}, \quad \mathbf{b}_O \in \mathbb{R}^{D}
  $$
- FFN (MLP) with expansion ratio 4:
  $$
  \mathbf{W}_1 \in \mathbb{R}^{D \times 4D}, \ \mathbf{b}_1 \in \mathbb{R}^{4D}
  $$
  $$
  \mathbf{W}_2 \in \mathbb{R}^{4D \times D}, \ \mathbf{b}_2 \in \mathbb{R}^{D}
  $$
  Here ($4 D = 3072$).

### 3.3 High-Level Block Structure

_Pre-LN Transformer Encoder Layer_ implements the following canonical computation:

$$
\begin{aligned}
\mathbf{X}' &= \mathrm{LN}_1(\mathbf{X}) \\
\mathbf{A} &= \mathrm{MHSA}(\mathbf{X}') \\
\mathbf{Y} &= \mathbf{X} + \mathbf{A} \\
\mathbf{Y}' &= \mathrm{LN}_2(\mathbf{Y}) \\
\mathbf{F} &= \mathrm{FFN}(\mathbf{Y}') \\
\mathbf{Z} &= \mathbf{Y} + \mathbf{F}
\end{aligned}
$$

The function returns ($\mathbf{Z}$).

### 3.4 LayerNorm Definition

LayerNorm is applied over the **last dimension** (D) (hidden size), independently for each ($(n, \ell)$) position.

For a token vector ($\mathbf{x} \in \mathbb{R}^{D}$):

$$
\mathrm{LN}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta
$$

Where:

$$
\mu = \frac{1}{D}\sum_{i=1}^{D} x_i,
\qquad
\sigma^2 = \frac{1}{D}\sum_{i=1}^{D}(x_i - \mu)^2
$$

- ($\gamma$) corresponds to `.weight`
- ($\beta$) corresponds to `.bias`
- ($\varepsilon = \tt{param.lnorm\_eps}$)

### 3.5 QKV Projections (Code Section 2.1)

#### 3.5.1 Linear projections

Let:

$$
\mathbf{X}' = \mathrm{LN}_1(\mathbf{X})
$$

Compute:

$$
\begin{aligned}
\mathbf{Q} &= \mathbf{X}' \mathbf{W}_Q^\top + \mathbf{b}_Q \
\mathbf{K} &= \mathbf{X}' \mathbf{W}_K^\top + \mathbf{b}_K \
\mathbf{V} &= \mathbf{X}' \mathbf{W}_V^\top + \mathbf{b}_V
\end{aligned}
\qquad
\mathbf{Q},\mathbf{K},\mathbf{V} \in \mathbb{R}^{N \times L \times D}
$$

#### 3.5.2 Multi-Head Reshape + Transpose (Shuffle Layers)

Multi-head attention splits the hidden dimension (D) into (H) heads of size (d).

#### 3.5.3 Reshape and transpose

Starting from:

$$
\mathbf{Q} \in \mathbb{R}^{N \times L \times D}
$$

Reshape:

$$
\mathbf{Q}_r \in \mathbb{R}^{N \times L \times H \times d}
$$

Transpose (swap axes to put heads first):

$$
\mathbf{Q}_h \in \mathbb{R}^{N \times H \times L \times d}
$$

Same for ($\mathbf{K}$) and ($\mathbf{V}$).

Code:

```cpp
q_s->setReshapeDimensions(Dims4{N, -1, H, d});
q_s->setSecondTranspose({0, 2, 1, 3}); // (N,H,L,d)
```

#### 3.5.4 SDPA (Scaled Dot-Product Attention)

For each batch (n) and head (h), define:

$$
\mathbf{Q}^{(n,h)} \in \mathbb{R}^{L \times d}, \quad
\mathbf{K}^{(n,h)} \in \mathbb{R}^{L \times d}, \quad
\mathbf{V}^{(n,h)} \in \mathbb{R}^{L \times d}
$$

#### 3.5.5 Attention logits ($QK^\top$)

$$
\mathbf{S}^{(n,h)} = \mathbf{Q}^{(n,h)} \left(\mathbf{K}^{(n,h)}\right)^\top
\in \mathbb{R}^{L \times L}
$$

In tensor form:

$$
\mathbf{S} \in \mathbb{R}^{N \times H \times L \times L}
$$

Code:

```cpp
qk = MatMul(q_s, NONE, k_s, TRANSPOSE); // (N,H,L,d) x (N,H,d,L) -> (N,H,L,L)
```

#### 3.5.6 Scaling

Scaled dot-product uses:

$$
\alpha = \frac{1}{\sqrt{d}}
$$

$$
\tilde{\mathbf{S}} = \alpha \mathbf{S}
$$

Code:

```cpp
scale_val = 1/sqrt(d);
attn_qk = qk * scale; // ElementWise PROD
```

#### 3.5.7 Softmax normalization

Softmax is applied on the **last dimension** (keys index), for each query position, So:

$$
\mathbf{P} \in \mathbb{R}^{N \times H \times L \times L}
$$

Code:

```cpp
qk_softmax = SoftMax(attn_qk);
qk_softmax->setAxes(1U << (nbDims-1)); // last axis
```

#### 3.5.8 Weighted sum of values

Each head output:

$$
\mathbf{O}^{(n,h)} = \mathbf{P}^{(n,h)} \mathbf{V}^{(n,h)}
\in \mathbb{R}^{L \times d}
$$

Thus:

$$
\mathbf{O} \in \mathbb{R}^{N \times H \times L \times d}
$$

Code:

```cpp
attn_qkv = MatMul(qk_softmax, NONE, v_s, NONE); // (N,H,L,L)x(N,H,L,d)->(N,H,L,d)
```

### 3.6 Merge Heads + Output Projection

#### 3.6.1 Merge heads

Transpose back:

$$
\mathbf{O} \in \mathbb{R}^{N \times H \times L \times d}
\ \xrightarrow{\text{transpose}}
\mathbb{R}^{N \times L \times H \times d}
$$

Then reshape:

$$
\mathbb{R}^{N \times L \times (H\cdot d)} = \mathbb{R}^{N \times L \times D}
$$

Code:

```cpp
attn_out->setFirstTranspose({0, 2, 1, 3}); // (N,L,H,d)
attn_out->setReshapeDimensions(Dims3{N, -1, 768}); // (N,L,D)
```

#### 3.6.2 Output projection

$$
\mathbf{A} = \mathbf{O}_{\text{merged}} \mathbf{W}_O^\top + \mathbf{b}_O
\quad\in\mathbb{R}^{N \times L \times D}
$$

Code:

```cpp
attn_fcw = MatMul(attn_out, out_proj_w^T);
attn_fcb = attn_fcw + out_proj_b;
```

### 3.7 Residual Connection After Attention

$$
\mathbf{Y} = \mathbf{X} + \mathbf{A}
\quad\in\mathbb{R}^{N \times L \times D}
$$

Code:

```cpp
attn_residual = input + attn_fcb;
```

This identity path is crucial for gradient flow and stability; at inference time it preserves a “direct” signal path even if attention becomes sharp or noisy.

### 3.8 Post-Attention LayerNorm

$$
\mathbf{Y}' = \mathrm{LN}_2(\mathbf{Y})
$$

Code:

```cpp
post_lnorm = Normalization(attn_residual, post_ln_scale, post_ln_bias)
```

### 3.9 Feed-Forward Network (FFN / MLP)

ViT uses a 2-layer MLP with expansion ratio 4 and GeLU activation.

#### 3.9.1 First dense layer (expand to 3072)

$$
\mathbf{H} = \mathbf{Y}' \mathbf{W}_1^\top + \mathbf{b}_1
\quad\in\mathbb{R}^{N \times L \times 4D}
$$

Code:

```cpp
inter0 = MatMul(post_lnorm, iw^T); // iw shape conceptually (4D, D)
inter1 = inter0 + ib;
```

#### 3.9.2 GeLU activation

$$
\mathrm{GeLU}(x) = x \Phi(x)
$$

Where (\Phi) is the standard normal CDF.

Common tanh approximation (widely used in implementations):

$$
\mathrm{GeLU}(x) \approx \frac
{x\times \bigg(1+\tanh\Big(\sqrt\frac{2}{\pi}\times (x+0.044715\times x^3)\Big)\bigg)}
{2}
$$

Code calls:

```cpp
inter_act = addGeLU(net, inter1);
```

#### 3.9.3 Second dense layer (project back to 768)

$$
\mathbf{F} = \mathrm{GeLU}(\mathbf{H}) \mathbf{W}_2^\top + \mathbf{b}_2
\quad\in\mathbb{R}^{N \times L \times D}
$$

Code:

```cpp
out0 = MatMul(inter_act, ow^T); // ow conceptually (D, 4D)
out1 = out0 + ob;
```

### 3.10 Final Residual Connection

$$
\mathbf{Z} = \mathbf{Y} + \mathbf{F}
\quad\in\mathbb{R}^{N \times L \times D}
$$

Code:

```cpp
output_residual = out1 + attn_residual;
return output_residual;
```

## 4. Compact Step-by-Step Shape Trace

Below is a shape trace aligned with the main operations (assuming dynamic (L)):

Input

$$ \mathbf{X}: (N, L, 768) $$

Pre-LN

$$ \mathbf{X}': (N, L, 768) $$

Q/K/V projections

$$ \mathbf{Q},\mathbf{K},\mathbf{V}: (N, L, 768) $$

Reshape + transpose to heads

$$ \mathbf{Q}\_h,\mathbf{K}\_h,\mathbf{V}\_h: (N, H, L, d) $$

Attention logits

$$ \mathbf{S}: (N, H, L, L) $$

Softmax weights

$$ \mathbf{P}: (N, H, L, L) $$

Head outputs

$$ \mathbf{O}: (N, H, L, d) $$

Merge heads

$$ \mathbf{O}\_{\text{merged}}: (N, L, 768) $$

Output projection

$$ \mathbf{A}: (N, L, 768) $$

Residual

$$ \mathbf{Y}: (N, L, 768) $$

Post-LN

$$ \mathbf{Y}': (N, L, 768) $$

FFN expand

$$ \mathbf{H}: (N, L, 3072) $$

FFN project

$$ \mathbf{F}: (N, L, 768) $$

Final residual

$$ \mathbf{Z}: (N, L, 768) $$
