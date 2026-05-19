import argparse
import os
import struct
from pathlib import Path

import numpy as np
import paddle
from paddle.static.pir_io import get_pir_parameters


DEFAULT_DET_MODEL = "official_models/PP-OCRv5_mobile_det"
DEFAULT_REC_MODEL = "official_models/PP-OCRv5_mobile_rec"

STANDARD_MODELS = [
    ("PP-OCRv5_mobile_det", "ppocrv5_mobile_det"),
    ("PP-OCRv5_mobile_rec", "ppocrv5_mobile_rec"),
    ("PP-OCRv5_server_det", "ppocrv5_server_det"),
    ("PP-OCRv5_server_rec", "ppocrv5_server_rec"),
    ("PP-LCNet_x1_0_doc_ori", "pp_lcnet_x1_0_doc_ori"),
    ("PP-LCNet_x1_0_table_cls", "pp_lcnet_x1_0_table_cls"),
    ("PP-LCNet_x1_0_textline_ori", "pp_lcnet_x1_0_textline_ori"),
    ("PP-DocBlockLayout", "pp_docblocklayout"),
    ("PP-DocLayout_plus-L", "pp_doclayout_plus_l"),
    ("RT-DETR-L_wired_table_cell_det", "rt_detr_l_wired_table_cell_det"),
    ("RT-DETR-L_wireless_table_cell_det", "rt_detr_l_wireless_table_cell_det"),
    ("SLANet_plus", "slanet_plus"),
    ("SLANeXt_wired", "slanext_wired"),
    ("UVDoc", "uvdoc"),
    ("PP-FormulaNet_plus-L", "pp_formulanet_plus_l"),
]

STANDARD_TAGS = {tag for _, tag in STANDARD_MODELS}

MODEL_TYPE_ALIASES = {
    "m_det": "ppocrv5_mobile_det",
    "mobile_det": "ppocrv5_mobile_det",
    "det_m": "ppocrv5_mobile_det",
    "m_rec": "ppocrv5_mobile_rec",
    "mobile_rec": "ppocrv5_mobile_rec",
    "rec_m": "ppocrv5_mobile_rec",
    "s_det": "ppocrv5_server_det",
    "server_det": "ppocrv5_server_det",
    "det_s": "ppocrv5_server_det",
    "s_rec": "ppocrv5_server_rec",
    "server_rec": "ppocrv5_server_rec",
    "rec_s": "ppocrv5_server_rec",
    "doc_ori": "pp_lcnet_x1_0_doc_ori",
    "table_cls": "pp_lcnet_x1_0_table_cls",
    "textline_ori": "pp_lcnet_x1_0_textline_ori",
    "docblock": "pp_docblocklayout",
    "docblocklayout": "pp_docblocklayout",
    "layout": "pp_doclayout_plus_l",
    "doclayout": "pp_doclayout_plus_l",
    "doclayout_plus_l": "pp_doclayout_plus_l",
    "wired_table": "rt_detr_l_wired_table_cell_det",
    "wired_table_cell_det": "rt_detr_l_wired_table_cell_det",
    "wireless_table": "rt_detr_l_wireless_table_cell_det",
    "wireless_table_cell_det": "rt_detr_l_wireless_table_cell_det",
    "slanet": "slanet_plus",
    "slanext": "slanext_wired",
    "formula": "pp_formulanet_plus_l",
}


def _float_to_hex(value):
    return hex(struct.unpack(">I", struct.pack(">f", float(value)))[0])


def _load_pir_model(model_dir):
    os.environ["FLAGS_enable_pir_api"] = "1"
    os.environ["FLAGS_enable_pir_in_executor"] = "1"
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    program, feed_names, fetch_targets = paddle.static.load_inference_model(
        os.path.join(model_dir, "inference"), exe
    )
    params, _ = get_pir_parameters(program)
    return program, feed_names, fetch_targets, params


def _tensor_to_numpy(name):
    var = paddle.static.global_scope().find_var(name)
    if var is None:
        raise RuntimeError(f"parameter not found in Paddle global scope: {name}")
    return np.array(var.get_tensor()).astype(np.float32, copy=False)


def dump_wts(model_dir, out_path):
    program, feed_names, fetch_targets, params = _load_pir_model(model_dir)
    param_names = [p.name for p in params]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{len(param_names)}\n")
        for name in param_names:
            arr = np.ascontiguousarray(_tensor_to_numpy(name).reshape(-1))
            f.write(f"{name} {arr.size}")
            for value in arr:
                f.write(f" {_float_to_hex(value)}")
            f.write("\n")

    return program, feed_names, fetch_targets, param_names


def model_tag(model_dir):
    tag = Path(model_dir).name.lower().replace("pp-ocrv5", "ppocrv5")
    clean = []
    last_underscore = False
    for ch in tag:
        if ch.isalnum():
            clean.append(ch)
            last_underscore = False
        elif not last_underscore:
            clean.append("_")
            last_underscore = True
    return "".join(clean).strip("_")


def normalize_model_type(model_type):
    if not model_type:
        return ""
    tag = model_type.lower().replace("-", "_").replace("pp_ocrv5", "ppocrv5")
    return MODEL_TYPE_ALIASES.get(tag, tag)


def resolve_wts_output(output_arg, model_type):
    output_path = Path(output_arg)
    if str(output_arg).endswith(("/", "\\")) or (output_path.exists() and output_path.is_dir()):
        if not model_type:
            raise RuntimeError("model type is required when -o/--output is a directory")
        output_path = output_path / f"{model_type}.wts"
    return output_path


def export_one_model(model_dir, tag, out_dir):
    model_dir = str(model_dir)
    tag = tag or model_tag(model_dir)
    wts_path = out_dir / f"{tag}.wts"

    print(f"Exporting WTS: {wts_path}")
    _, _, _, param_names = dump_wts(model_dir, wts_path)

    return tag, len(param_names)


def _has_inference_model(model_dir):
    return (model_dir / "inference.pdiparams").exists()


def find_standard_model_dir(root, official_name, tag):
    candidates = [
        root / official_name,
        root / tag,
        root / f"{official_name}_infer",
        root / f"{tag}_infer",
    ]
    for candidate in candidates:
        if _has_inference_model(candidate):
            return candidate
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Export PP-OCRv5 Paddle PIR weights to WTS")
    parser.add_argument(
        "model_type",
        nargs="?",
        default="",
        help="Optional model type for -w/-o, for example m_det, s_rec, layout, formula",
    )
    parser.add_argument("-w", "--weights", default="", help="Paddle inference model directory for one WTS export")
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Output WTS path, or an output directory when model_type is set",
    )
    parser.add_argument("--det-model", default=DEFAULT_DET_MODEL)
    parser.add_argument("--rec-model", default=DEFAULT_REC_MODEL)
    parser.add_argument("--det-tag", default="")
    parser.add_argument("--rec-tag", default="")
    parser.add_argument("--model", action="append", default=[], help="Export an arbitrary Paddle inference model dir")
    parser.add_argument("--tag", action="append", default=[], help="Tag for the matching --model entry")
    parser.add_argument(
        "--official-model-dir",
        default="",
        help="Export the standard supported model set below this root",
    )
    parser.add_argument(
        "--all-official-models",
        default="",
        help="Export all Paddle inference models below this directory",
    )
    parser.add_argument("--out-dir", default="build")
    return parser.parse_args()


def main():
    args = parse_args()
    normalized_type = normalize_model_type(args.model_type)

    if bool(args.weights) != bool(args.output):
        raise RuntimeError("-w/--weights and -o/--output must be used together")

    if args.weights:
        output_path = resolve_wts_output(args.output, normalized_type)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shown_type = normalized_type or model_tag(args.weights)
        known = "standard" if shown_type in STANDARD_TAGS else "custom"
        print(f"model type: {shown_type} ({known})")
        print(f"Exporting WTS: {output_path}")
        _, _, _, param_names = dump_wts(args.weights, output_path)
        print(f"params: {len(param_names)}")
        return

    if normalized_type:
        raise RuntimeError("trailing model_type is only valid with -w/--weights and -o/--output")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generic_models = list(args.model)
    generic_tags = list(args.tag)
    if args.official_model_dir:
        root = Path(args.official_model_dir)
        missing = []
        for official_name, tag in STANDARD_MODELS:
            model_dir = find_standard_model_dir(root, official_name, tag)
            if model_dir is None:
                missing.append(official_name)
                continue
            generic_models.append(str(model_dir))
            generic_tags.append(tag)
        if missing:
            raise RuntimeError("missing official model dirs: " + ", ".join(missing))

    if args.all_official_models:
        root = Path(args.all_official_models)
        for item in sorted(root.iterdir()):
            if _has_inference_model(item):
                generic_models.append(str(item))
                generic_tags.append("")
            else:
                print(f"Skipping non-PIR model: {item}")

    if generic_models:
        while len(generic_tags) < len(generic_models):
            generic_tags.append("")
        report = []
        for model_dir, tag in zip(generic_models, generic_tags):
            report.append(export_one_model(model_dir, tag, out_dir))
        print("exported models:")
        for tag, param_count in report:
            print(f"  {tag}: params={param_count}")
        return

    det_tag = args.det_tag or model_tag(args.det_model)
    rec_tag = args.rec_tag or model_tag(args.rec_model)

    det_wts = out_dir / f"{det_tag}.wts"
    rec_wts = out_dir / f"{rec_tag}.wts"
    print(f"Exporting det WTS: {det_wts}")
    _, _, _, det_names = dump_wts(args.det_model, det_wts)

    print(f"Exporting rec WTS: {rec_wts}")
    _, _, _, rec_names = dump_wts(args.rec_model, rec_wts)

    print(f"det params: {len(det_names)}")
    print(f"rec params: {len(rec_names)}")


if __name__ == "__main__":
    main()
