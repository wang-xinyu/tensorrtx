"""Multi-variant + dynamic-batch precision & latency experiment for ViT.

For each requested ViT variant and batch size, compare 4 backends:

  1. PyTorch FP32 (HuggingFace)              -> reference logits
  2. ONNX Runtime CUDA EP (FP32 ONNX)
  3. TensorRT engine built from ONNX (FP16, dynamic batch profile 1/2/4)
  4. TensorRT engine built from this repo's vit.cc (FP16, from .wts)

Variants share the table in gen_wts.py / vit.cc:
    ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32, ViT-H/14

Usage:
    python run_experiment.py                                  # all 5 variants, batches 1 2 4
    python run_experiment.py --variants ViT-B/16 ViT-L/16
    python run_experiment.py --batches 1 4

Prerequisites (per variant V):
    models/<safe(V)>.wts       (created by gen_wts.py)
    models/<safe(V)>.engine    (created by build/vit -s <wts> <engine> <V>)

Where safe(V) replaces "/" with "-", e.g. ViT-B/16 -> ViT-B-16.
The ONNX file and TRT-from-ONNX engine are produced lazily on first run.

Output:
    exp/<safe(V)>__bs<B>.npz   per (variant, batch) logits dump
    Final markdown-style summary table to stdout.
"""

from __future__ import annotations

import argparse
import gc
# import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForImageClassification

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
MODELS = ROOT / "models"
EXPDIR = ROOT / "exp"
EXPDIR.mkdir(exist_ok=True)
SAMPLE_IMG = ASSETS / "cats.jpg"
LABELS = ASSETS / "imagenet1000_clsidx_to_labels.txt"

# Mirror of gen_wts.py / vit.cc::getVariantConfig
VARIANTS: Dict[str, Tuple[str, int, int]] = {
    "ViT-B/16": ("google/vit-base-patch16-224",       224, 1000),
    "ViT-B/32": ("google/vit-base-patch32-384",       384, 1000),
    "ViT-L/16": ("google/vit-large-patch16-224",      224, 1000),
    "ViT-L/32": ("google/vit-large-patch32-384",      384, 1000),
    "ViT-H/14": ("google/vit-huge-patch14-224-in21k", 224, 21843),
}
BATCHES_DEFAULT = [1, 2, 4]
PROFILE_MIN, PROFILE_OPT, PROFILE_MAX = 1, 2, 4
N_WARMUP = 20
N_ITERS = 100


def safe_filename(model_type: str) -> str:
    return model_type.replace("/", "-")


# -----------------------------------------------------------------------------
# Inputs & metrics
# -----------------------------------------------------------------------------
def preprocess(batch: int, img_size: int) -> np.ndarray:
    """Same preprocessing as vit.cc: resize bilinear, /255, mean=std=0.5, NCHW."""
    img = cv2.imread(str(SAMPLE_IMG), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_size, img_size), cv2.INTER_LINEAR)
    img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))                    # CHW
    x = np.broadcast_to(img[None, ...], (batch, 3, img_size, img_size))
    return np.ascontiguousarray(x, dtype=np.float32)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# -----------------------------------------------------------------------------
# Backends
# -----------------------------------------------------------------------------
def _load_wts_tensors(wts_path: Path, names) -> Dict[str, torch.Tensor]:
    """Parse subset of tensors from a .wts text file (name N hex hex ...)."""
    want = set(names)
    out: Dict[str, torch.Tensor] = {}
    with open(wts_path, "r") as f:
        next(f)  # header (count)
        for line in f:
            parts = line.rstrip("\n").split(" ")
            name = parts[0]
            if name not in want:
                continue
            n = int(parts[1])
            vals = np.frombuffer(
                bytes.fromhex("".join(parts[2:2 + n])), dtype=">f4"
            ).astype(np.float32).copy()
            out[name] = torch.from_numpy(vals)
            if len(out) == len(want):
                break
    return out


def _load_hf_model(hub_id: str, num_classes: int, model_type: str):
    cfg = AutoConfig.from_pretrained(hub_id)
    cfg._attn_implementation = "eager"
    # Match gen_wts.py: in21k checkpoints (e.g. ViT-H/14) need explicit
    # num_labels override so the random classifier matches our table.
    cfg.num_labels = num_classes
    cfg.id2label = {i: str(i) for i in range(num_classes)}
    cfg.label2id = {str(i): i for i in range(num_classes)}
    model = AutoModelForImageClassification.from_pretrained(
        hub_id, config=cfg, ignore_mismatched_sizes=True,
    ).eval()
    # Sync the (possibly randomly-initialized) classifier with the .wts that
    # the C++ engine consumes. Otherwise different random seeds across
    # processes make torch reference logits inconsistent with TRT outputs
    # (e.g. ViT-H/14 in21k has no public 1k classifier).
    wts = MODELS / f"{safe_filename(model_type)}.wts"
    if wts.exists():
        t = _load_wts_tensors(
            wts, ["classifier.weight", "classifier.bias"]
        )
        if "classifier.weight" in t and "classifier.bias" in t:
            with torch.no_grad():
                w = t["classifier.weight"].view_as(model.classifier.weight)
                b = t["classifier.bias"].view_as(model.classifier.bias)
                model.classifier.weight.copy_(w)
                model.classifier.bias.copy_(b)
    return model


def torch_logits(hub_id: str, num_classes: int, model_type: str, x_np: np.ndarray) -> np.ndarray:
    model = _load_hf_model(hub_id, num_classes, model_type)
    with torch.no_grad():
        # All samples in batch are identical; run batch=1 and tile to save VRAM.
        out = model(torch.from_numpy(x_np[:1])).logits.numpy()[0]
    del model
    gc.collect()
    return np.broadcast_to(out, (x_np.shape[0], out.shape[0])).copy()


def export_onnx(hub_id: str, num_classes: int, model_type: str, onnx_path: Path, img_size: int) -> None:
    if onnx_path.exists():
        return
    print(f"  [onnx] exporting -> {onnx_path}")
    model = _load_hf_model(hub_id, num_classes, model_type)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).logits

    dummy = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
    torch.onnx.export(
        Wrap(model), dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
    )
    del model
    gc.collect()


def ort_logits(onnx_path: Path, x_np: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(
        str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    return sess.run(["logits"], {"input": x_np})[0]


def build_trt_from_onnx(onnx_path: Path, engine_path: Path, img_size: int) -> None:
    if engine_path.exists():
        return
    import tensorrt as trt
    print(f"  [trt-onnx] building -> {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    # parse_from_file lets the parser locate external data files (needed when
    # the ONNX model exceeds 2 GiB and torch.onnx.export spills initializers
    # next to the .onnx file, e.g. for ViT-H/14).
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit(1)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    config.builder_optimization_level = 5
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        (PROFILE_MIN, 3, img_size, img_size),
        (PROFILE_OPT, 3, img_size, img_size),
        (PROFILE_MAX, 3, img_size, img_size),
    )
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(plan)


def trt_infer(engine_path: Path, x_np: np.ndarray, n_warmup: int, n_iters: int):
    """Run TRT engine, return (logits_fp32 batchedN, ms_enqueue, ms_graph)."""
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    in_name = out_name = None
    for i in range(engine.num_io_tensors):
        nm = engine.get_tensor_name(i)
        m = engine.get_tensor_mode(nm)
        if m == trt.TensorIOMode.INPUT and in_name is None:
            in_name = nm
        elif m == trt.TensorIOMode.OUTPUT and out_name is None:
            out_name = nm

    trt_to_np = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT32: np.int32,
        trt.DataType.INT8: np.int8,
        trt.DataType.BOOL: np.bool_,
    }
    in_dtype = trt_to_np[engine.get_tensor_dtype(in_name)]
    out_dtype = trt_to_np[engine.get_tensor_dtype(out_name)]
    x_in = np.ascontiguousarray(x_np.astype(in_dtype))

    ctx.set_input_shape(in_name, x_in.shape)
    out_shape = tuple(ctx.get_tensor_shape(out_name))
    out_buf = np.empty(out_shape, dtype=out_dtype)

    d_in = cuda.mem_alloc(x_in.nbytes)
    d_out = cuda.mem_alloc(out_buf.nbytes)
    stream = cuda.Stream()

    cuda.memcpy_htod_async(d_in, x_in, stream)
    ctx.set_tensor_address(in_name, int(d_in))
    ctx.set_tensor_address(out_name, int(d_out))

    for _ in range(n_warmup):
        ctx.execute_async_v3(stream.handle)
    stream.synchronize()

    s, e = cuda.Event(), cuda.Event()
    s.record(stream)
    for _ in range(n_iters):
        ctx.execute_async_v3(stream.handle)
    e.record(stream)
    e.synchronize()
    ms_enqueue = e.time_since(s) / n_iters

    ms_graph = float("nan")

    try:
        from cuda import cudart

        def _chk(ret):
            err = ret[0]
            if int(err) != 0:
                raise RuntimeError(f"cudart err {err}")
            return ret[1] if len(ret) > 1 else None
        ctx.execute_async_v3(stream.handle)
        stream.synchronize()
        _chk(cudart.cudaStreamBeginCapture(
            stream.handle, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
        ctx.execute_async_v3(stream.handle)
        graph = _chk(cudart.cudaStreamEndCapture(stream.handle))
        exec_graph = _chk(cudart.cudaGraphInstantiate(graph, 0))
        for _ in range(n_warmup):
            _chk(cudart.cudaGraphLaunch(exec_graph, stream.handle))
        stream.synchronize()
        gs, ge = cuda.Event(), cuda.Event()
        gs.record(stream)
        for _ in range(n_iters):
            _chk(cudart.cudaGraphLaunch(exec_graph, stream.handle))
        ge.record(stream)
        ge.synchronize()
        ms_graph = ge.time_since(gs) / n_iters
        cudart.cudaGraphExecDestroy(exec_graph)
        cudart.cudaGraphDestroy(graph)
    except Exception as ex:
        print(f"  [warn] CUDA Graph path failed: {ex}")

    cuda.memcpy_dtoh_async(out_buf, d_out, stream)
    stream.synchronize()
    return out_buf.astype(np.float32), ms_enqueue, ms_graph


# -----------------------------------------------------------------------------
# Per-variant runner
# -----------------------------------------------------------------------------
def run_variant(model_type: str, batches: List[int], tag: str = "") -> List[Dict]:
    hub_id, img_size, num_classes = VARIANTS[model_type]
    safe = safe_filename(model_type)
    suffix = f"__{tag}" if tag else ""
    onnx_path = EXPDIR / f"{safe}.onnx"
    trt_onnx_path = EXPDIR / f"{safe}__from_onnx_fp16{suffix}.engine"
    trt_wts_path = MODELS / f"{safe}{suffix}.engine"
    wts_path = MODELS / f"{safe}.wts"

    if not wts_path.exists():
        print(f"[skip] {model_type}: missing {wts_path} (run gen_wts.py {model_type})")
        return []
    if not trt_wts_path.exists():
        print(f"[skip] {model_type}: missing {trt_wts_path} "
              f"(run ./build/vit -s {wts_path} {trt_wts_path} {model_type})")
        return []

    rows: List[Dict] = []
    print(f"\n=== {model_type}  hub={hub_id}  img={img_size}  classes={num_classes} ===")

    export_onnx(hub_id, num_classes, model_type, onnx_path, img_size)
    build_trt_from_onnx(onnx_path, trt_onnx_path, img_size)

    for B in batches:
        if B > PROFILE_MAX:
            print(f"  [skip] batch={B} > engine max={PROFILE_MAX}")
            continue
        print(f"  -- batch = {B} --")
        x = preprocess(B, img_size)

        torch_out = torch_logits(hub_id, num_classes, model_type, x)            # (B, C)
        ort_out = ort_logits(onnx_path, x)             # (B, C)
        trt_onnx_out, ms_o, ms_o_g = trt_infer(trt_onnx_path, x, N_WARMUP, N_ITERS)
        trt_wts_out, ms_w, ms_w_g = trt_infer(trt_wts_path, x, N_WARMUP, N_ITERS)

        # Per-sample cosine vs torch ref (sample 0 is enough since identical)
        cos_ort = cos_sim(torch_out[0], ort_out[0])
        cos_trt_onnx = cos_sim(torch_out[0], trt_onnx_out[0])
        cos_trt_wts = cos_sim(torch_out[0], trt_wts_out[0])

        print(f"    cos[ort]      = {cos_ort:.6f}")
        print(f"    cos[trt-onnx] = {cos_trt_onnx:.6f}  enqueue={ms_o:.3f} ms  graph={ms_o_g:.3f} ms")
        print(f"    cos[trt-wts]  = {cos_trt_wts:.6f}  enqueue={ms_w:.3f} ms  graph={ms_w_g:.3f} ms")

        np.savez(EXPDIR / f"{safe}__bs{B}{suffix}.npz",
                 torch=torch_out, ort=ort_out,
                 trt_onnx=trt_onnx_out, trt_wts=trt_wts_out)

        rows.append(dict(
            variant=model_type, batch=B,
            cos_ort=cos_ort, cos_trt_onnx=cos_trt_onnx, cos_trt_wts=cos_trt_wts,
            ms_onnx=ms_o, ms_onnx_g=ms_o_g, ms_wts=ms_w, ms_wts_g=ms_w_g,
        ))
    return rows


def print_summary(rows: List[Dict]) -> None:
    if not rows:
        print("\n[summary] no rows.")
        return
    print("\n=== Summary (latency in ms / iter) ===")
    hdr = ("variant", "B",
           "cos[ort]", "cos[trt-onnx]", "cos[trt-wts]",
           "trt-onnx eq", "trt-onnx grph",
           "trt-wts eq",  "trt-wts grph",
           "speedup eq (wts/onnx)", "speedup grph")
    fmt = "| {:<9} | {:>2} | {:>8} | {:>13} | {:>12} | {:>11} | {:>13} | {:>10} | {:>12} | {:>21} | {:>12} |"
    print(fmt.format(*hdr))
    print(fmt.format(*["-" * len(h) for h in hdr]))
    for r in rows:
        sp_eq = r["ms_onnx"] / r["ms_wts"]
        sp_g = (r["ms_onnx_g"] / r["ms_wts_g"]
                if r["ms_wts_g"] == r["ms_wts_g"] and r["ms_onnx_g"] == r["ms_onnx_g"]
                else float("nan"))
        print(fmt.format(
            r["variant"], r["batch"],
            f"{r['cos_ort']:.5f}", f"{r['cos_trt_onnx']:.5f}", f"{r['cos_trt_wts']:.5f}",
            f"{r['ms_onnx']:.3f}", f"{r['ms_onnx_g']:.3f}",
            f"{r['ms_wts']:.3f}",  f"{r['ms_wts_g']:.3f}",
            f"{sp_eq:.2f}x", f"{sp_g:.2f}x",
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    parser.add_argument("--batches", nargs="+", type=int, default=BATCHES_DEFAULT)
    parser.add_argument(
        "--tag", default="",
        help="Suffix appended to engine artifact names (e.g. trt1016) so multiple "
             "TensorRT versions can coexist. .wts and .onnx are NOT suffixed.",
    )
    args = parser.parse_args()

    if not SAMPLE_IMG.exists():
        print(f"missing sample image: {SAMPLE_IMG}")
        sys.exit(1)

    all_rows: List[Dict] = []
    for v in args.variants:
        if v not in VARIANTS:
            print(f"[err] unknown variant: {v} (choose from {list(VARIANTS.keys())})")
            continue
        all_rows.extend(run_variant(v, args.batches, args.tag))

    print_summary(all_rows)


if __name__ == "__main__":
    main()
