import nvtripy as tp


class ConvBnSilu(tp.Module):
    def __init__(self, in_channels, out_channels, kernel_dims, stride, dtype):
        super().__init__()
        self.conv = tp.Conv(
            in_channels,
            out_channels,
            kernel_dims,
            stride=stride,
            padding=[(dim // 2, dim // 2) for dim in kernel_dims],
            bias=False,
            dtype=dtype,
        )
        self.bn = tp.BatchNorm(out_channels, eps=1e-3, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = tp.silu(x)
        return x


class Bottleneck(tp.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut,
        kernel_dims1,
        kernel_dims2,
        expansion_ratio,
        dtype,
    ):
        super().__init__()
        expanded_out_channels = int(out_channels * expansion_ratio)
        self.cv1 = ConvBnSilu(in_channels, expanded_out_channels, kernel_dims1, stride=(1, 1), dtype=dtype)
        self.cv2 = ConvBnSilu(
            expanded_out_channels,
            out_channels,
            kernel_dims2,
            stride=(1, 1),
            dtype=dtype,
        )

        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        out = self.cv1(x)
        out = self.cv2(out)
        if self.shortcut:
            out += x
        return out


class C3k(tp.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers,
        shortcut,
        kernel_dims1,
        kernel_dims2,
        expansion_ratio,
        dtype,
    ):
        super().__init__()
        expanded_out_channels = int(out_channels * expansion_ratio)

        self.cv1 = ConvBnSilu(
            in_channels,
            expanded_out_channels,
            kernel_dims=(1, 1),
            stride=(1, 1),
            dtype=dtype,
        )
        self.cv2 = ConvBnSilu(
            in_channels,
            expanded_out_channels,
            kernel_dims=(1, 1),
            stride=(1, 1),
            dtype=dtype,
        )

        self.m = tp.Sequential(
            *[
                Bottleneck(
                    expanded_out_channels,
                    expanded_out_channels,
                    shortcut,
                    kernel_dims1,
                    kernel_dims2,
                    1.0,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.cv3 = ConvBnSilu(
            2 * expanded_out_channels,
            out_channels,
            kernel_dims=(1, 1),
            stride=(1, 1),
            dtype=dtype,
        )

    def forward(self, x):
        out1 = self.cv1(x)
        out2 = self.cv2(x)

        out1 = self.m(out1)
        out = tp.concatenate((out1, out2), dim=1)
        out = self.cv3(out)
        return out


class C3K2(tp.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers,
        use_c3k,
        shortcut,
        expansion_ratio,
        dtype,
    ):
        super().__init__()

        expanded_out_channels = int(out_channels * expansion_ratio)
        self.cv1 = ConvBnSilu(
            in_channels,
            2 * expanded_out_channels,
            kernel_dims=(1, 1),
            stride=(1, 1),
            dtype=dtype,
        )

        self.m = tp.Sequential(
            *[
                (
                    C3k(
                        expanded_out_channels,
                        expanded_out_channels,
                        2,
                        shortcut,
                        (3, 3),
                        (3, 3),
                        0.5,
                        dtype=dtype,
                    )
                    if use_c3k
                    else Bottleneck(
                        expanded_out_channels,
                        expanded_out_channels,
                        shortcut,
                        (3, 3),
                        (3, 3),
                        0.5,
                        dtype=dtype,
                    )
                )
                for _ in range(num_layers)
            ]
        )

        # Number of input channels to CV2 is the output channels of CV1 plus all
        # output channels from the layers in `m`.
        cv2_in_channels = (2 * expanded_out_channels) + (expanded_out_channels * num_layers)
        self.cv2 = ConvBnSilu(cv2_in_channels, out_channels, (1, 1), (1, 1), dtype=dtype)

    def forward(self, x):
        x = self.cv1(x)

        _, m_inp = tp.split(x, 2, dim=1)

        cat = x
        # We manually iterate over the Sequential module here since we need to access the intermediate outputs.
        for layer in self.m:
            m_inp = layer(m_inp)
            cat = tp.concatenate((cat, m_inp), dim=1)
        out = self.cv2(cat)
        return out


class ConvBn(tp.Module):
    def __init__(self, in_channels, out_channels, kernel_dims, stride, dtype, num_groups=1):
        super().__init__()
        self.conv = tp.Conv(
            in_channels,
            out_channels,
            kernel_dims,
            stride=stride,
            padding=[(dim // 2, dim // 2) for dim in kernel_dims],
            bias=False,
            groups=num_groups,
            dtype=dtype,
        )
        self.bn = tp.BatchNorm(out_channels, eps=1e-3, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Attention(tp.Module):
    def __init__(self, dim, num_heads, attn_ratio, dtype):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.key_dim = int(head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = self.dim + nh_kd * 2

        self.qkv = ConvBn(self.dim, h, (1, 1), (1, 1), dtype=dtype)
        self.pe = ConvBn(self.dim, self.dim, (3, 3), (1, 1), dtype=dtype, num_groups=self.dim)
        self.proj = ConvBn(self.dim, self.dim, (1, 1), (1, 1), dtype=dtype)

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W

        x = self.qkv(x)

        x = tp.reshape(x, (B, self.num_heads, -1, N))

        q, k, v = tp.split(x, [self.key_dim, self.key_dim, self.key_dim * 2], dim=2)

        q_t = tp.transpose(q, 2, 3)

        softmax = tp.softmax((q_t @ k) * self.scale, dim=3)

        attn_t = tp.transpose(softmax, 2, 3)

        matmul2 = v @ attn_t
        reshape = tp.reshape(matmul2, (B, -1, H, W))

        v_reshape = tp.reshape(v, (B, self.dim, H, W))

        pe = self.pe(v_reshape)

        sum = reshape + pe
        proj = self.proj(sum)
        return proj


class PSABlock(tp.Module):
    def __init__(self, dim, attn_ratio, num_heads, shortcut, dtype):
        super().__init__()

        self.attn = Attention(dim, num_heads, attn_ratio, dtype=dtype)
        self.shortcut = shortcut

        self.ffn = tp.Sequential(
            ConvBnSilu(dim, dim * 2, (1, 1), (1, 1), dtype=dtype),
            ConvBn(dim * 2, dim, (1, 1), (1, 1), dtype=dtype),
        )

    def forward(self, x):
        attn_out = self.attn(x)
        if self.shortcut:
            x = x + attn_out
        else:
            x = attn_out

        ffn_out = self.ffn(x)
        if self.shortcut:
            x = x + ffn_out
        else:
            x = ffn_out

        return x


class C2PSA(tp.Module):
    def __init__(self, input_channels, output_channels, num_layers, expansion_ratio, dtype):
        super().__init__()

        expanded_input_channels = int(input_channels * expansion_ratio)

        self.cv1 = ConvBnSilu(input_channels, 2 * expanded_input_channels, (1, 1), (1, 1), dtype=dtype)
        self.m = tp.Sequential(
            *[
                PSABlock(
                    expanded_input_channels,
                    0.5,
                    expanded_input_channels // 64,
                    True,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.cv2 = ConvBnSilu(2 * expanded_input_channels, output_channels, (1, 1), (1, 1), dtype=dtype)

    def forward(self, x):
        x = self.cv1(x)

        split1, y = tp.split(x, 2, dim=1)

        y = self.m(y)

        cat = tp.concatenate((split1, y), dim=1)
        out = self.cv2(cat)
        return out
