# State Farm Distracted Driver - Full Pipeline

This is an end-to-end training + inference pipeline implementing your requested strategy:

- Skin/hand/face-focused preprocessing (HSV + YCrCb skin mask).
- Upper-body ROI bias crop.
- ResNet18 + ConvNeXt-Tiny dual-backbone training.
- Label smoothing, cosine LR decay, mixed precision.
- TTA prediction + softmax ensemble.
- Embedding concat (`512 + 768 = 1280`) and L2 normalization.
- FAISS cosine KNN graph construction.
- Attention-weighted neighbor smoothing (graph message passing style).
- Multi-step propagation with configurable number of smoothing steps.

## Expected input CSV format

`train_labels.csv`
- `img` (e.g. `img_1.jpg`)
- `label` (0..9)
- optional `img_path` (if missing, script auto-uses `train/<img>`)

`sample_submission.csv`
- `img`
- `c0..c9`
- optional `img_path` (if missing, script auto-uses `test/<img>`)

## Run

```bash
python state_farm_distracted_driver_detection/pipeline.py \
  --data-dir /path/to/state-farm \
  --out-dir outputs \
  --save-dir checkpoints \
  --epochs-resnet 6 \
  --epochs-convnext 4 \
  --tta 2 \
  --knn-k 10 \
  --alpha 0.7 \
  --temperature 0.1 \
  --smooth-steps 2
```

## Tuning notes

- Keep `knn-k` around `8-15` (too high can over-smooth).
- Tune `temperature` in `0.05-0.2`.
- Tune `alpha` in `0.6-0.8`.
- `smooth-steps=2` is usually strong; avoid too many steps.
