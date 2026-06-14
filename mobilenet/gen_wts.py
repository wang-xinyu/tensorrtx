import argparse
import struct
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights
from torchvision.models.mobilenet import mobilenet_v2, mobilenet_v3_small

SUPPORTED_MODELS = ("mobilenet_v2", "mobilenet_v3_small")


def read_imagenet_labels(path: Path) -> dict[int, str]:
    clsid2label: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f.readlines():
            k, v = line.split(": ")
            clsid2label.setdefault(int(k), v[1:-3])
    return clsid2label


def preprocess(img: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)[None, ...]
    return torch.from_numpy(img)


def create_model(name: str) -> torch.nn.Module:
    if name == "mobilenet_v2":
        return mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    if name == "mobilenet_v3_small":
        return mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    raise ValueError(f"unsupported model: {name}")


def export_model(name: str, output_dir: Path, labels: dict[int, str], image: torch.Tensor) -> None:
    print(f"Now dealing with model: {name}")
    model = create_model(name).eval()
    with torch.inference_mode():
        output = model(image)
        for i, batch in enumerate(torch.topk(output, k=3).indices):
            for j, idx in enumerate(batch):
                print(f"\tBatch: {i}, Top: {j}, logits: {output[i][idx]:.4f}, label: {labels[int(idx)]}")
        print(f"{'=' * 32}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{name}.wts").open("w", encoding="utf-8") as f:
        f.write(f"{len(model.state_dict().keys())}\n")
        for key, value in model.state_dict().items():
            values = value.reshape(-1).cpu().numpy()
            f.write(f"{key} {len(values)} ")
            print(key, value.shape)
            for item in values:
                f.write(" ")
                f.write(struct.pack(">f", float(item)).hex())
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export torchvision MobileNet weights to tensorrtx .wts format.")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, action="append", help="model to export; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    parser.add_argument("--image", type=Path, default=Path(__file__).resolve().parents[1] / "assets" / "cats.jpg")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "imagenet1000_clsidx_to_labels.txt",
    )
    args = parser.parse_args()

    labels = read_imagenet_labels(args.labels)
    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)
    image = preprocess(img)

    for name in args.model or SUPPORTED_MODELS:
        export_model(name, args.output_dir, labels, image)


if __name__ == "__main__":
    main()
