# MR-YOLO

Multi-Resolution YOLO for RF signal detection.

MR-YOLO processes several spectrograms of the **same signal at different time-frequency resolutions** simultaneously. Each resolution is handled by an independent branch backbone. Branch features are fused, then fed into a shared FPN/PAN neck and YOLO detection head (P3/P4/P5, DFL regression, Task-Aligned Assigner).

---

## Installation

```bash
git clone https://github.com/<you>/mr_yolo.git
cd mr_yolo
pip install -r requirements.txt
```

---

## Dataset format

Each `.pt` file contains a Python list of tensors — one tensor per resolution, in the same order as `--res-keys`. Each `.json` file contains the annotations for one sample.

```
<data_dir>/
├── train/
│   ├── data/            *.pt  files  (list of tensors)
│   └── labels_detect/   *.json files
└── val/
    ├── data/
    └── labels_detect/
```

Label JSON format (one object per detection):
```json
[
  {"class": 3, "xc": 0.52, "yc": 0.38, "w": 0.12, "h": 0.08, "snr": 12.4},
  ...
]
```

---

## Training

```bash
python train.py \
    --data-dir /data/rf_dataset \
    --res-keys cfg512 cfg256 cfg128 cfg1024 cfg2048
```

The resolutions are **auto-detected** from the `.pt` files. `--res-keys` just provides names for them, in the same order.

**Common options:**

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | — | Dataset root (required) |
| `--res-keys` | — | Ordered resolution key names (required) |
| `--scale` | `n` | `n` (nano, width=0.25) · `s` (small, 0.50) · `m` (medium, 0.75) |
| `--epochs` | 100 | Max training epochs |
| `--batch-size` | 32 | |
| `--lr` | 1e-3 | Learning rate |
| `--patience` | 10 | Early stopping patience |
| `--monitor` | `val_loss` | `val_loss` · `map50` · `map50_95` |
| `--num-classes` | 20 | Number of target classes |
| `--preprocessing` | `none` | `none` · `spectrogram_psnr` · `complex_real_imag` · … |
| `--output-dir-parent` | `runs` | Parent folder for experiment outputs |
| `--dry-run` | — | Print config without training |

**Example — small model, 3 resolutions, monitor mAP50:**
```bash
python train.py \
    --data-dir /data/rf_dataset \
    --res-keys cfg512 cfg256 cfg1024 \
    --scale s \
    --epochs 150 --batch-size 64 \
    --monitor map50 --full-eval-every 5
```

Training outputs (in `runs/<experiment>/`):

```
best.pt              Best checkpoint
last.pt              Latest checkpoint
train_log.csv        Per-epoch metrics
loss_curves.png      Train / val loss
map_curves.png       mAP50 / mAP50:95 curves
```

---

## Evaluation

```bash
python predict.py \
    --checkpoint runs/mr_yolo_n_cfg512_cfg256_cfg1024/best.pt \
    --data-dir /data/rf_dataset \
    --res-keys cfg512 cfg256 cfg1024
```

Results are written to `<checkpoint_dir>/eval_val.json` by default.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | — | Path to `.pt` checkpoint (required) |
| `--data-dir` | — | Dataset root (required) |
| `--res-keys` | — | Same as training (required) |
| `--split` | `val` | `train` · `val` · `test` |
| `--scale` | `n` | Must match training scale |
| `--iou-thresh` | 0.5 | IoU threshold for matching |
| `--false-alarm-target` | 0.01 | Target false-alarm rate for ROC |
| `--output-json` | auto | Path for the JSON results file |

---

## Python API

```python
from mr_yolo import MR_YOLO

# Build model
model = MR_YOLO(
    input_resolutions=[(512, 512), (256, 1024), (128, 2048)],
    output_dir="runs/my_experiment",
    num_classes=20,
    device="cuda:0",
    in_ch=1,
    width_mult=0.25,        # nano
    backbone_mode="TFSep_pyramid",
)

# Train
model.fit(
    data_dir="/data/rf_dataset",
    epochs=100,
    batch_size=32,
    lr=1e-3,
    patience=10,
    dataset="fused",
    select_res={"res_keys": ["cfg512", "cfg256", "cfg1024"]},
    monitor="val_loss",
)

# Inference
import torch
imgs = [torch.randn(1, 1, 512, 512), torch.randn(1, 1, 256, 1024)]
model.eval()
predictions, _, _ = model.predict(imgs, conf_threshold=0.3)
# predictions: list of (N, 6) tensors [x1, y1, x2, y2, score, class]
```

---

## Architecture

```
Input: [spec_res0, spec_res1, ..., spec_resN]   (B, C, H_i, W_i)
         │
         ├─ BranchBackbone_0 ──┐
         ├─ BranchBackbone_1 ──┤
         │  ...                 ├─► Feature Fusion  ──► P3/P4/P5 Neck  ──► YOLO Head
         └─ BranchBackbone_N ──┘                         (FPN/PAN)          (DFL)
```

Each branch backbone is a lightweight CNN with transformer blocks (`TFSep_pyramid` mode). Feature fusion concatenates branch outputs and projects to a common channel dimension.

**Scale shortcuts:**

| Scale | `width_mult` | Approx. params |
|---|---|---|
| n (nano) | 0.25 | ~3 M |
| s (small) | 0.50 | ~10 M |
| m (medium) | 0.75 | ~22 M |