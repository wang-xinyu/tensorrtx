import math

import nvtripy as tp

from .block import C2PSA, C3K2, ConvBnSilu

NUM_CLASSES = 1000


def get_width(w, gw, max_channels, divisor=8):
    return int(math.ceil((min(w, max_channels) * gw) / divisor)) * divisor


def get_depth(d, gd):
    if d == 1:
        return d

    r = round(d * gd)
    # Round ties for even numbers down:
    if d * gd - int(d * gd) == 0.5 and (int(d * gd) % 2) == 0:
        r -= 1
    return max(r, 1)


class Yolo11Head(tp.Module):
    def __init__(self, input_channels, dtype):
        super().__init__()
        self.conv = ConvBnSilu(input_channels, 1280, (1, 1), (1, 1), dtype=dtype)
        self.linear = tp.Linear(1280, NUM_CLASSES, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        # Global average pooling:
        x = tp.reshape(tp.mean(x, dim=(2, 3), keepdim=True), (-1, 1280))
        x = self.linear(x)
        return x


class Yolo11Cls(tp.Module):
    def __init__(self, model_variant, gd, gw, max_channels, dtype=tp.float32):
        use_c3k = model_variant in {"m", "l", "x"}

        self.model = tp.Sequential(
            ConvBnSilu(3, get_width(64, gw, max_channels), (3, 3), (2, 2), dtype=dtype),
            ConvBnSilu(
                get_width(64, gw, max_channels),
                get_width(128, gw, max_channels),
                (3, 3),
                (2, 2),
                dtype=dtype,
            ),
            C3K2(
                get_width(128, gw, max_channels),
                get_width(256, gw, max_channels),
                get_depth(2, gd),
                use_c3k,
                True,
                0.25,
                dtype=dtype,
            ),
            ConvBnSilu(
                get_width(256, gw, max_channels),
                get_width(256, gw, max_channels),
                (3, 3),
                (2, 2),
                dtype=dtype,
            ),
            C3K2(
                get_width(256, gw, max_channels),
                get_width(512, gw, max_channels),
                get_depth(2, gd),
                use_c3k,
                True,
                0.25,
                dtype=dtype,
            ),
            ConvBnSilu(
                get_width(512, gw, max_channels),
                get_width(512, gw, max_channels),
                (3, 3),
                (2, 2),
                dtype=dtype,
            ),
            C3K2(
                get_width(512, gw, max_channels),
                get_width(512, gw, max_channels),
                get_depth(2, gd),
                True,
                True,
                0.5,
                dtype=dtype,
            ),
            ConvBnSilu(
                get_width(512, gw, max_channels),
                get_width(1024, gw, max_channels),
                (3, 3),
                (2, 2),
                dtype=dtype,
            ),
            C3K2(
                get_width(1024, gw, max_channels),
                get_width(1024, gw, max_channels),
                get_depth(2, gd),
                True,
                True,
                0.5,
                dtype=dtype,
            ),
            C2PSA(
                get_width(1024, gw, max_channels),
                get_width(1024, gw, max_channels),
                get_depth(2, gd),
                0.5,
                dtype=dtype,
            ),
            Yolo11Head(get_width(1024, gw, max_channels), dtype=dtype),
        )

    def forward(self, x):
        x = self.model(x)
        return x
