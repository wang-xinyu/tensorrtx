Validation images are stored outside this module to keep the source tree small.

Clone the sample repository and point commands to its PP-OCR sample directory:

```bash
git clone https://github.com/lindsayshuo/infer_pic.git /path/to/infer_pic
export PPOCRV5_SAMPLE_DIR=/path/to/infer_pic/ppocr/samples/validation
```

Each standard engine has a validation directory named after the engine basename, for example:

```text
${PPOCRV5_SAMPLE_DIR}/ppocrv5_mobile_det
${PPOCRV5_SAMPLE_DIR}/ppocrv5_mobile_rec
${PPOCRV5_SAMPLE_DIR}/pp_formulanet_plus_l
${PPOCRV5_SAMPLE_DIR}/pp_formulanet_plus_l.decoder
```

The full image set and mapping table live in:

```text
https://github.com/lindsayshuo/infer_pic/tree/main/ppocr/samples/validation
```
