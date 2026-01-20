import torch
import struct


def gen_wts(model_path, wts_path):
    print(f"Loading {model_path}...")
    try:
        data = torch.load(model_path, map_location='cpu')
    except FileNotFoundError:
        print(f"Error: {model_path} not found.")
        return

    if isinstance(data, dict) and 'model' in data:
        state_dict = data['model']
    else:
        state_dict = data

    print(f"Exporting to {wts_path}...")

    # Infer architecture
    dims = []
    depths = [0, 0, 0, 0]

    # Check dimensions from downsample layers
    # downsample_layers.0.0 is stem: conv set output to dim[0]
    # downsample_layers.1.0 is conv: dim[0] -> dim[1]
    # ...

    if 'downsample_layers.0.0.weight' in state_dict:
        dims.append(state_dict['downsample_layers.0.0.weight'].shape[0])
    if 'downsample_layers.1.0.weight' in state_dict:
        dims.append(state_dict['downsample_layers.1.0.weight'].shape[0])
    if 'downsample_layers.2.0.weight' in state_dict:
        dims.append(state_dict['downsample_layers.2.0.weight'].shape[0])
    if 'downsample_layers.3.0.weight' in state_dict:
        dims.append(state_dict['downsample_layers.3.0.weight'].shape[0])

    # Count blocks per stage
    for k in state_dict.keys():
        if k.startswith('stages.'):
            parts = k.split('.')
            if len(parts) >= 3:
                stage_idx = int(parts[1])
                block_idx = int(parts[2])
                if stage_idx < 4:
                    depths[stage_idx] = max(depths[stage_idx], block_idx + 1)

    print("Inferred Architecture:")
    print(f"  Dims: {dims}")
    print(f"  Depths: {depths}")

    with open(wts_path, 'w') as f:
        f.write(f"{len(state_dict)}\n")
        for k, v in state_dict.items():
            vr = v.reshape(-1).cpu().numpy()
            f.write(f"{k} {len(vr)}")
            for val in vr:
                f.write(" ")
                f.write(struct.pack('>f', float(val)).hex())
            f.write("\n")

    print("Done.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <pt_path> <wts_path>")
        print(f"Example: python {sys.argv[0]} models/test.pt convnextv2.wts")
        sys.exit(1)

    pt_path = sys.argv[1]
    wts_path = sys.argv[2]
    gen_wts(pt_path, wts_path)
