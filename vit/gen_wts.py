"""Export ViT weights from HuggingFace to a .wts file for TensorRT.

Usage:
    python gen_wts.py <model_type> [<output_wts_path>]

Examples:
    python gen_wts.py ViT-B/16
    python gen_wts.py ViT-L/16 models/vit_l16.wts
    python gen_wts.py b32                              # alias accepted

Model types supported (must match the table in vit.cc::getVariantConfig):
    ViT-B/16  ->  google/vit-base-patch16-224       (img=224, classes=1000)
    ViT-B/32  ->  google/vit-base-patch32-384       (img=384, classes=1000)
    ViT-L/16  ->  google/vit-large-patch16-224      (img=224, classes=1000)
    ViT-L/32  ->  google/vit-large-patch32-384      (img=384, classes=1000)
    ViT-H/14  ->  google/vit-huge-patch14-224-in21k (img=224, classes=21843,
                                                     ImageNet-21k pretrain only;
                                                     no public 1k fine-tuned ckpt)

Default output path (when omitted) is models/<safe_name>.wts where slashes
are replaced by hyphens, e.g. ViT-B/16 -> models/ViT-B-16.wts. This matches
the recommended layout used by exp/run_experiment.py.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageClassification

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"
MODELS_DIR = SCRIPT_DIR / "models"
LABELS_FILE = ASSETS_DIR / "imagenet1000_clsidx_to_labels.txt"
SAMPLE_IMG = ASSETS_DIR / "cats.jpg"

# Canonical name -> (HF hub id, image size, num classes)
VARIANTS: dict[str, tuple[str, int, int]] = {
    "ViT-B/16": ("google/vit-base-patch16-224",       224, 1000),
    "ViT-B/32": ("google/vit-base-patch32-384",       384, 1000),
    "ViT-L/16": ("google/vit-large-patch16-224",      224, 1000),
    "ViT-L/32": ("google/vit-large-patch32-384",      384, 1000),
    "ViT-H/14": ("google/vit-huge-patch14-224-in21k", 224, 21843),
}

ALIASES: dict[str, str] = {
    "b16": "ViT-B/16", "B16": "ViT-B/16", "B/16": "ViT-B/16",
    "b32": "ViT-B/32", "B32": "ViT-B/32", "B/32": "ViT-B/32",
    "l16": "ViT-L/16", "L16": "ViT-L/16", "L/16": "ViT-L/16",
    "l32": "ViT-L/32", "L32": "ViT-L/32", "L/32": "ViT-L/32",
    "h14": "ViT-H/14", "H14": "ViT-H/14", "H/14": "ViT-H/14",
}


def normalize_model_type(name: str) -> str:
    if name in VARIANTS:
        return name
    if name in ALIASES:
        return ALIASES[name]
    upper = name.upper().replace(" ", "")
    if upper.startswith("VIT-"):
        cand = "ViT-" + upper[4:]
        if cand in VARIANTS:
            return cand
    raise SystemExit(
        f"Unknown model_type: {name!r}. "
        f"Choose from: {', '.join(VARIANTS.keys())}"
    )


def safe_filename(model_type: str) -> str:
    """ViT-B/16 -> ViT-B-16 (filesystem-safe)."""
    return model_type.replace("/", "-")


def read_imagenet_labels() -> dict[int, str]:
    clsid2label: dict[int, str] = {}
    with open(LABELS_FILE, "r") as f:
        for line in f:
            k, v = line.split(": ", 1)
            clsid2label[int(k)] = v.rstrip().rstrip(",").strip().strip("'\"")
    return clsid2label


def maybe_sanity_check(
    model: torch.nn.Module, img_size: int, num_classes: int
) -> None:
    if not SAMPLE_IMG.exists():
        print(f"[skip] sanity check: missing {SAMPLE_IMG}")
        return
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[skip] sanity check: opencv-python / numpy not installed")
        return

    img = cv2.imread(str(SAMPLE_IMG), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_size, img_size), cv2.INTER_LINEAR)
    img = (img.astype(np.float32) / 255.0 - np.array([0.5, 0.5, 0.5])) / np.array(
        [0.5, 0.5, 0.5]
    )
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))[None, ...]).float()

    with torch.no_grad():
        output = model(img)
    print(f"[sanity] logits.shape={tuple(output.logits.shape)}  (expected (1,{num_classes}))")
    if num_classes == 1000 and LABELS_FILE.exists():
        labels = read_imagenet_labels()
        for rank, idx in enumerate(torch.topk(output.logits[0], k=3).indices.tolist()):
            print(f"  Top {rank}: idx={idx} label={labels.get(idx, '?')}")
    else:
        for rank, idx in enumerate(torch.topk(output.logits[0], k=3).indices.tolist()):
            print(f"  Top {rank}: idx={idx}")


def export_wts(model: torch.nn.Module, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    with open(out_path, "w") as f:
        f.write(f"{len(sd)}\n")
        for k, v in sd.items():
            vr = v.detach().reshape(-1).cpu().numpy()
            f.write(f"{k} {vr.size}")
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")
    size_mib = out_path.stat().st_size / 1024 / 1024
    print(f"[ok] wrote {out_path}  ({size_mib:.1f} MiB, {len(sd)} tensors)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("model_type", help="e.g. ViT-B/16, ViT-L/32, b16, h14")
    parser.add_argument("output", nargs="?", default=None,
                        help="output .wts path (default: models/<safe_name>.wts)")
    parser.add_argument("--no-sanity", action="store_true",
                        help="skip the optional forward-pass sanity check")
    args = parser.parse_args()

    model_type = normalize_model_type(args.model_type)
    hub_id, img_size, num_classes = VARIANTS[model_type]
    out_path = (
        Path(args.output) if args.output else MODELS_DIR / f"{safe_filename(model_type)}.wts"
    )

    print(f"[load] model_type={model_type}  hub_id={hub_id}  img={img_size}  classes={num_classes}")
    config = AutoConfig.from_pretrained(hub_id)
    config._attn_implementation = "eager"
    # Force the classifier head size: in21k checkpoints (e.g. ViT-H/14) ship
    # without a classifier and HF would otherwise default num_labels=2,
    # producing a tiny random head incompatible with our engine table.
    config.num_labels = num_classes
    config.id2label = {i: str(i) for i in range(num_classes)}
    config.label2id = {str(i): i for i in range(num_classes)}
    model = AutoModelForImageClassification.from_pretrained(
        hub_id, ignore_mismatched_sizes=True, config=config,
    )
    model.eval()

    if not args.no_sanity:
        maybe_sanity_check(model, img_size, num_classes)
    export_wts(model, out_path)


if __name__ == "__main__":
    main()
