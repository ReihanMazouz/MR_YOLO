# MR-YOLO

**Multi-Resolution YOLO for RF signal detection**

MR-YOLO is a PyTorch detection framework for radio-frequency signals observed
through multiple time-frequency resolutions. Instead of forcing a single
spectrogram scale, the model consumes several synchronized representations of
the same signal, extracts resolution-specific features, fuses them, and predicts
signal locations with a YOLO-style dense detection head.

The implementation is designed for research-grade RF detection experiments:
multi-resolution inputs, automatic shape discovery, configurable model scale,
YOLO detection losses, full validation metrics, and reproducible experiment
outputs.

---

## Highlights

- **Multi-resolution signal modeling**: each input resolution has its own branch
  backbone before feature fusion.
- **YOLO detection stack**: shared P3/P4/P5 neck, distribution focal loss
  regression, task-aligned assignment, mAP and recall-oriented evaluation.
- **RF-aware evaluation**: metrics include detection quality and recall by SNR.
- **Experiment-ready CLI**: training and evaluation scripts auto-detect input
  tensor shapes from the dataset.
- **Composable Python API**: instantiate, train, evaluate, and run inference
  directly from Python.

---

## Architecture

![MR-YOLO architecture](assets/MRS_YOLO_architecture.png)

MR-YOLO processes a list of spectrogram tensors from the same RF sample:

```text
Input: [spec_res0, spec_res1, ..., spec_resN]   (B, C, H_i, W_i)
         |
         +-- BranchBackbone_0 --+
         +-- BranchBackbone_1 --+
         |                      +--> Feature Fusion --> P3/P4/P5 Neck --> YOLO Head
         +-- BranchBackbone_N --+                         FPN/PAN          DFL
```

Each resolution is handled by a lightweight branch backbone with
time-frequency transformer blocks. Branch outputs are aligned into common
feature pyramids, fused, and passed to a shared FPN/PAN neck and detection head.

---

## Performance View

![Recall by SNR](assets/recall_snr.png)

The evaluation pipeline reports standard detection metrics and signal-domain
diagnostics such as recall across SNR bins. This makes it possible to inspect
model behavior under low-SNR and high-SNR operating regimes rather than relying
only on aggregate mAP.

---

## Benchmark Summary

The benchmark figures below compare detection quality against model cost. They
are intended to make the accuracy-efficiency trade-off explicit when selecting
an MR-YOLO scale or comparing against alternative detectors.

### mAP vs Compute

![Benchmark mAP vs FLOPs](assets/benchmark_map_vs_flops.png)

### mAP vs Model Size

![Benchmark mAP vs parameters](assets/benchmark_map_vs_params.png)

### SNR Recall vs Compute

![Benchmark recall by SNR vs FLOPs](assets/benchmark_recall_snr_vs_flops.png)

### SNR Recall vs Model Size

![Benchmark recall by SNR vs parameters](assets/benchmark_recall_snr_vs_params.png)

---

## Repository Layout

```text
.
├── assets/                    Figures used by the README
├── mr_yolo/
│   ├── models/                MR-YOLO model, backbones, detection head
│   ├── nn/                    Core neural network blocks
│   └── utils/                 Dataset, preprocessing, loss, metrics, plotting
├── train.py                   Training entry point
├── predict.py                 Evaluation and inference entry point
├── requirements.txt           Python dependencies
└── README.md
```

---

## Installation

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/<owner>/mr_yolo.git
cd mr_yolo
python -m pip install -r requirements.txt
```

Recommended environment:

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU for training larger experiments

The minimal dependency set is intentionally small:

```text
torch, torchinfo, scipy, numpy, matplotlib, seaborn, tqdm
```

---

## Dataset Format

MR-YOLO expects one tensor file and one annotation file per sample. Each `.pt`
file contains a Python list of tensors, with one tensor per resolution. The
order of tensors must match the order passed through `--res-keys`.

```text
<data_dir>/
├── train/
│   ├── data/            *.pt files
│   └── labels_detect/   *.json files
└── val/
    ├── data/            *.pt files
    └── labels_detect/   *.json files
```

Example label file:

```json
[
  {
    "class": 3,
    "xc": 0.52,
    "yc": 0.38,
    "w": 0.12,
    "h": 0.08,
    "snr": 12.4
  }
]
```

Annotation fields:

| Field | Description |
|---|---|
| `class` | Integer class index |
| `xc`, `yc` | Normalized bounding-box center coordinates |
| `w`, `h` | Normalized bounding-box width and height |
| `snr` | Signal-to-noise ratio used for SNR-conditioned analysis |

---

## Training

Basic training run:

```bash
python train.py \
  --data-dir /data/rf_dataset \
  --res-keys cfg512 cfg256 cfg128 cfg1024 cfg2048
```

MR-YOLO auto-detects tensor shapes from the first `.pt` file in
`<data_dir>/train/data`. The `--res-keys` argument provides stable names for
the resolutions and must match the tensor order in every sample.

Example with a larger model and mAP-based checkpointing:

```bash
python train.py \
  --data-dir /data/rf_dataset \
  --res-keys cfg512 cfg256 cfg1024 \
  --scale s \
  --epochs 150 \
  --batch-size 64 \
  --lr 1e-3 \
  --monitor map50 \
  --full-eval-every 5
```

Common options:

| Flag | Default | Description |
|---|---:|---|
| `--data-dir` | required | Dataset root |
| `--res-keys` | required | Ordered resolution names |
| `--scale` | `n` | Model scale: `n`, `s`, or `m` |
| `--num-classes` | `20` | Number of target classes |
| `--epochs` | `100` | Maximum number of epochs |
| `--batch-size` | `32` | Training batch size |
| `--lr` | `1e-3` | Learning rate |
| `--patience` | `10` | Early-stopping patience |
| `--monitor` | `val_loss` | Metric used for `best.pt`: `val_loss`, `map50`, `map50_95` |
| `--preprocessing` | `none` | Input preprocessing mode |
| `--output-dir-parent` | `runs` | Parent directory for experiment outputs |
| `--dry-run` | off | Print the resolved configuration without training |

Model scales:

| Scale | Width multiplier | Typical use |
|---|---:|---|
| `n` | `0.25` | Fast baselines and ablations |
| `s` | `0.50` | Balanced accuracy and cost |
| `m` | `0.75` | Larger experiments |

Training artifacts are written to `runs/<experiment>/`:

```text
best.pt              Best checkpoint according to --monitor
last.pt              Latest checkpoint
train_log.csv        Per-epoch metrics
loss_curves.png      Training and validation losses
map_curves.png       mAP50 and mAP50:95 curves
```

---

## Evaluation

Evaluate a trained checkpoint on a dataset split:

```bash
python predict.py \
  --checkpoint runs/mr_yolo_n_cfg512_cfg256_cfg1024/best.pt \
  --data-dir /data/rf_dataset \
  --res-keys cfg512 cfg256 cfg1024
```

Evaluate on a test split and write results to a custom location:

```bash
python predict.py \
  --checkpoint runs/mr_yolo_n_cfg512_cfg256_cfg1024/best.pt \
  --data-dir /data/rf_dataset \
  --res-keys cfg512 cfg256 cfg1024 \
  --split test \
  --output-json results/mr_yolo_test.json
```

Evaluation options:

| Flag | Default | Description |
|---|---:|---|
| `--checkpoint` | required | Path to `.pt` checkpoint |
| `--data-dir` | required | Dataset root |
| `--res-keys` | required | Same ordered resolution keys used during training |
| `--split` | `val` | Dataset split: `train`, `val`, or `test` |
| `--scale` | `n` | Must match the training scale |
| `--num-classes` | `20` | Number of target classes |
| `--preprocessing` | `none` | Must match the training preprocessing |
| `--iou-thresh` | `0.5` | IoU threshold for detection matching |
| `--false-alarm-target` | `0.01` | False-alarm target used by ROC-style analysis |
| `--output-json` | auto | Defaults to `<checkpoint_dir>/eval_<split>.json` |

The JSON output contains model configuration, evaluation settings, mAP metrics,
precision, recall, and SNR-conditioned diagnostics.

---

## Python API

```python
import torch

from mr_yolo import MR_YOLO

model = MR_YOLO(
    input_resolutions=[(512, 512), (256, 1024), (128, 2048)],
    output_dir="runs/my_experiment",
    num_classes=20,
    device="cuda:0",
    in_ch=1,
    width_mult=0.25,
)

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

model.eval()
inputs = [
    torch.randn(1, 1, 512, 512, device="cuda:0"),
    torch.randn(1, 1, 256, 1024, device="cuda:0"),
    torch.randn(1, 1, 128, 2048, device="cuda:0"),
]

predictions, _, _ = model.predict(inputs, conf_threshold=0.3)
```

`predictions` is a list of tensors in `[x1, y1, x2, y2, score, class]` format.

---

## Reproducibility Checklist

For comparable experiments, keep the following values fixed and documented:

- Dataset split and sample generation process
- Resolution list and `--res-keys` order
- Model scale and `width_mult`
- Number of classes
- Preprocessing mode
- Training schedule, batch size, learning rate, and early-stopping monitor
- Checkpoint used for evaluation
- IoU threshold and false-alarm target

---

## License

This repository is distributed under the terms of the license included in
[LICENSE](LICENSE).
