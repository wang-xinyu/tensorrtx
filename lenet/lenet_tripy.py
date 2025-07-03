import argparse
import os
import struct

import nvtripy as tp

INPUT_SHAPE = (1, 1, 32, 32)
WEIGHT_PATH = "lenet5.wts"
COMPILED_MODEL_PATH = "lenet5.tpymodel"


def load_weights(file):
    if not os.path.exists(file):
        raise FileNotFoundError(f"Weight file: {file} does not exist.")

    with open(file, "r") as f:
        lines = [line.strip() for line in f]

    count = int(lines[0])
    assert count == len(lines) - 1, "Mismatch in weight count."

    return {
        splits[0]: tp.Tensor([struct.unpack(">f", bytes.fromhex(hex_val))[0] for hex_val in splits[2:]])
        for splits in (line.split(" ") for line in lines[1:])
    }


class Lenet5(tp.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = tp.Conv(1, 6, kernel_dims=(5, 5))
        self.conv2 = tp.Conv(6, 16, kernel_dims=(5, 5))
        self.fc1 = tp.Linear(16 * 5 * 5, 120)
        self.fc2 = tp.Linear(120, 84)
        self.fc3 = tp.Linear(84, 10)

    def forward(self, x):
        x = tp.relu(self.conv1(x))
        x = tp.avgpool(x, kernel_dims=(2, 2), stride=(2, 2))
        x = tp.relu(self.conv2(x))
        x = tp.avgpool(x, kernel_dims=(2, 2), stride=(2, 2))

        x = tp.flatten(x, 1)

        x = tp.relu(self.fc1(x))
        x = tp.relu(self.fc2(x))
        x = tp.softmax(self.fc3(x), dim=1)
        return x


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", action="store_true", help="Save the model")
    group.add_argument("-d", action="store_true", help="Load a saved model")
    args = parser.parse_args()

    if args.s:
        model = Lenet5()

        weights = load_weights(WEIGHT_PATH)
        # The weights in the weights file are flattened, so we need to reshape
        # them to the right shape before we can load them:
        for name, tensor in model.state_dict().items():
            weights[name] = tp.reshape(weights[name], tensor.shape)

        model.load_state_dict(weights)

        compiled_model = tp.compile(model, args=[tp.InputInfo(INPUT_SHAPE, dtype=tp.float32)])

        compiled_model.save(COMPILED_MODEL_PATH)
    else:
        compiled_model = tp.Executable.load(COMPILED_MODEL_PATH)

        data = tp.ones(INPUT_SHAPE, dtype=tp.float32).eval()

        output = compiled_model(data)

        print(f"Output: {output}")


if __name__ == "__main__":
    main()
