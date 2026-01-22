import argparse
import os

import nvtripy as tp
import requests
import torch
from constants import IMAGE_C, IMAGE_H, IMAGE_W
from model.model import Yolo11Cls
from tqdm import tqdm

CURDIR = os.path.realpath(os.path.dirname(__file__))


def get_model_config(model_variant):
    config = {
        "model_variant": model_variant,
    }
    if model_variant == "n":
        config.update({"gd": 0.50, "gw": 0.25, "max_channels": 1024})
    elif model_variant == "s":
        config.update({"gd": 0.50, "gw": 0.50, "max_channels": 1024})
    elif model_variant == "m":
        config.update({"gd": 0.50, "gw": 1.00, "max_channels": 512})
    elif model_variant == "l":
        config.update({"gd": 1.0, "gw": 1.0, "max_channels": 512})
    elif model_variant == "x":
        config.update({"gd": 1.0, "gw": 1.50, "max_channels": 512})

    return config


def download_weights(model_variant, directory):
    out_path = os.path.join(directory, f"yolo11{model_variant}-cls.pt")

    if os.path.exists(out_path):
        print(f"Checkpoint already exists at: {out_path}, skipping download.")
        return out_path

    URL = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11{model_variant}-cls.pt"

    response = requests.get(URL, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    os.makedirs(directory, exist_ok=True)

    with open(out_path, "wb") as f, tqdm(
        desc=f"Downloading checkpoint: yolo11{model_variant}-cls.pt",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            progress_bar.update(len(chunk))

    return out_path


def load_weights(weights_path, dtype):
    checkpoint = torch.load(weights_path, weights_only=False)
    torch_model = checkpoint["model"].eval()
    if dtype == tp.float16:
        torch_model = torch_model.half()
    else:
        assert dtype == tp.float32, "Unsupported dtype"
        torch_model = torch_model.float()

    state_dict = torch_model.state_dict()

    # Some weights from the training graph are not needed for inference:
    def should_include(key):
        return "num_batches_tracked" not in key

    return {name: tp.Tensor(weight) for name, weight in state_dict.items() if should_include(name)}


def main():
    parser = argparse.ArgumentParser(description="Compiles a YOLO11 classifier model with Tripy.")
    parser.add_argument(
        "--model-variant",
        help="Model variant (n, s, m, l, x)",
        default="n",
        choices=["n", "s", "m", "l", "x"],
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Where to save the Tripy executable",
        default="yolo11-cls.tpymodel",
    )
    parser.add_argument(
        "--checkpoints-dir",
        help="Where to save PyTorch checkpoints",
        default=os.path.join(CURDIR, "checkpoints"),
    )
    parser.add_argument(
        "--max-images",
        help="Maximum number of images the model will be able to classify at once, i.e. the maximum batch size.",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--dtype",
        help="Data type to use for inference",
        default="float16",
        choices=["float32", "float16"],
    )

    args, _ = parser.parse_known_args()

    config = get_model_config(args.model_variant)
    dtype = getattr(tp, args.dtype)
    model = Yolo11Cls(**config, dtype=dtype)

    weights_path = download_weights(args.model_variant, args.checkpoints_dir)

    model.load_state_dict(load_weights(weights_path, dtype))

    # We compile not only the classifier itself, but also accelerate the postprocessing:
    def infer(batch):
        out = model(batch)
        out = tp.softmax(out, dim=1)
        batch_scores, batch_preds = tp.topk(out, 3, dim=-1)
        return batch_scores, batch_preds

    print("Compiling YOLO11 classifier + postprocessing. This may take a few moments...")
    executable = tp.compile(
        infer,
        args=[
            tp.InputInfo(
                [
                    # Support a range of batch sizes from 1 to `max_images`, optimizing for the midpoint:
                    (1, (args.max_images + 1) // 2, args.max_images),
                    IMAGE_C,
                    IMAGE_H,
                    IMAGE_W,
                ],
                dtype=dtype,
            ),
        ],
    )

    print(f"Saving compiled executable to: {args.output}")
    executable.save(args.output)


if __name__ == "__main__":
    main()
