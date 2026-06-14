import argparse
import struct
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import (
    ResNeXt50_32X4D_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnext50_32x4d,
    wide_resnet50_2,
)

SUPPORTED_MODELS = ("resnet18", "resnet34", "resnet50", "resnext50_32x4d", "wide_resnet50_2")


def read_imagenet_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            key, value = line.split(": ", maxsplit=1)
            labels[int(key)] = value.strip()[1:-2]
    return labels


def preprocess(image_path: Path) -> torch.Tensor:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)[None, ...]
    return torch.from_numpy(image)


def create_model(name: str) -> torch.nn.Module:
    if name == "resnet18":
        return resnet18(weights=ResNet18_Weights.DEFAULT)
    if name == "resnet34":
        return resnet34(weights=ResNet34_Weights.DEFAULT)
    if name == "resnet50":
        return resnet50(weights=ResNet50_Weights.DEFAULT)
    if name == "resnext50_32x4d":
        return resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
    if name == "wide_resnet50_2":
        return wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
    raise ValueError(f"unsupported model: {name}")


def export_model(name: str, output_dir: Path, labels: dict[int, str], image: torch.Tensor) -> None:
    print(f"Now dealing with model: {name}")
    model = create_model(name).eval()
    with torch.inference_mode():
        output = model(image)
        for batch_idx, batch in enumerate(torch.topk(output, k=3).indices):
            for top_idx, label_idx in enumerate(batch):
                print(
                    f"\tBatch: {batch_idx}, Top: {top_idx}, "
                    f"logits: {output[batch_idx][label_idx]:.4f}, label: {labels[int(label_idx)]}"
                )
        print("=" * 32)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{name}.wts").open("w", encoding="utf-8") as file:
        file.write(f"{len(model.state_dict().keys())}\n")
        for key, value in model.state_dict().items():
            values = value.reshape(-1).cpu().numpy()
            file.write(f"{key} {len(values)} ")
            print(key, value.shape)
            for item in values:
                file.write(" ")
                file.write(struct.pack(">f", float(item)).hex())
            file.write("\n")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export torchvision ResNet weights to TensorRT .wts files.")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, action="append", help="model to export; repeatable")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "models")
    parser.add_argument("--image", type=Path, default=repo_root / "assets" / "cats.jpg")
    parser.add_argument(
        "--labels", type=Path, default=repo_root / "assets" / "imagenet1000_clsidx_to_labels.txt"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = read_imagenet_labels(args.labels)
    image = preprocess(args.image)
    for name in args.model or SUPPORTED_MODELS:
        export_model(name, args.output_dir, labels, image)


if __name__ == "__main__":
    main()
