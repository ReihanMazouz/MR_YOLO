#!/usr/bin/env python3
"""
Train MR-YOLO on a multi-resolution dataset.

MR-YOLO processes multiple spectrograms at different time-frequency resolutions
simultaneously. Each resolution is handled by an independent branch backbone,
and the branch features are fused before a shared P3/P4/P5 neck and YOLO head.

Dataset format
--------------
Each .pt file in data/ contains a list of tensors, one per resolution, in the
same order as --res-keys. Each .json file in labels_detect/ contains YOLO-format
annotations: xc, yc, w, h (normalized), class, snr.

  <data_dir>/
  ├── train/
  │   ├── data/           .pt files  (list of tensors)
  │   └── labels_detect/  .json files
  └── val/
      ├── data/
      └── labels_detect/

Examples
--------
  python train.py \\
      --data-dir /data/rf_dataset \\
      --res-keys cfg512 cfg256 cfg128 cfg1024 cfg2048

  python train.py \\
      --data-dir /data/rf_dataset \\
      --res-keys cfg512 cfg256 cfg1024 \\
      --scale s --epochs 150 --batch-size 64 --lr 1e-3

  # Dry run — print config without training
  python train.py --data-dir /data/rf_dataset \\
      --res-keys cfg512 cfg256 --dry-run
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mr_yolo import MR_YOLO
from mr_yolo.utils.preprocess import preprocessing_num_channels

Resolution = Tuple[int, int]

SCALE_WIDTH = {"n": 0.25, "s": 0.50, "m": 0.75}

def detect_resolutions(data_dir: str, res_keys: List[str]) -> Dict[str, Resolution]:
    """Read one .pt file to infer (H, W) for each resolution key."""
    images_dir = Path(data_dir) / "train" / "data"
    pt_file = next(images_dir.glob("*.pt"), None)
    if pt_file is None:
        raise FileNotFoundError(f"No .pt file found in {images_dir}")
    specs = torch.load(pt_file, map_location="cpu")
    if not isinstance(specs, list):
        raise ValueError(f"Expected a list of tensors in {pt_file}")
    if len(specs) != len(res_keys):
        raise ValueError(
            f"Dataset has {len(specs)} resolution tensors but "
            f"{len(res_keys)} res_keys provided: {res_keys}.\n"
            "Make sure --res-keys lists all resolutions in the same order as "
            "the tensors in each .pt file."
        )
    return {key: (int(s.shape[-2]), int(s.shape[-1])) for key, s in zip(res_keys, specs)}


def auto_output_name(scale: str, res_keys: List[str]) -> str:
    return f"mr_yolo_{scale}_{'_'.join(res_keys)}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MR-YOLO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True, help="Dataset root directory.")
    p.add_argument(
        "--res-keys", nargs="+", required=True,
        help="Ordered resolution keys matching the tensor order in each .pt file "
             "(e.g. cfg512 cfg256 cfg128 cfg1024 cfg2048).",
    )
    p.add_argument(
        "--scale", default="n", choices=["n", "s", "m"],
        help="Model scale: n=nano (width=0.25), s=small (0.50), m=medium (0.75).",
    )
    p.add_argument("--num-classes", type=int, default=20)
    p.add_argument("--reg-max", type=int, default=16)
    p.add_argument("--preprocessing", default="none",
                   help="Input preprocessing: none | spectrogram_psnr | complex_real_imag | …")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--full-eval-every", type=int, default=5,
                   help="Run full detection metrics every N epochs.")
    p.add_argument("--save-last-every", type=int, default=5)
    p.add_argument(
        "--monitor", default="val_loss",
        choices=["val_loss", "map50", "map50_95"],
        help="Metric used to save best.pt and trigger early stopping.",
    )
    p.add_argument("--output-dir-parent", default="runs")
    p.add_argument("--output-dir-name", default=None,
                   help="Experiment folder name. Auto-generated if omitted.")
    p.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print configuration and exit without training.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    resolutions = detect_resolutions(args.data_dir, args.res_keys)
    input_channels = preprocessing_num_channels(args.preprocessing)
    dir_name = args.output_dir_name or auto_output_name(args.scale, args.res_keys)
    output_dir = str(Path(args.output_dir_parent) / dir_name)

    print("MR-YOLO Training")
    print(f"  Scale       : {args.scale}  (width_mult={SCALE_WIDTH[args.scale]})")
    print(f"  Device      : {args.device}")
    print(f"  Output      : {output_dir}")
    print(f"  Channels    : {input_channels}  (preprocessing={args.preprocessing})")
    print(f"  Resolutions :")
    for key, hw in resolutions.items():
        print(f"    {key}: {hw[0]}×{hw[1]}")
    print(f"  Epochs      : {args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  Monitor     : {args.monitor}  patience={args.patience}")

    if args.dry_run:
        print("\n[dry-run] Exiting without training.")
        return

    if Path(output_dir).exists():
        print(f"\n[SKIP] Output directory already exists: {output_dir}")
        return

    model = MR_YOLO(
        input_resolutions=list(resolutions.values()),
        output_dir=output_dir,
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        in_ch=input_channels,
        width_mult=SCALE_WIDTH[args.scale],
    )

    try:
        model.fit(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            dataset="fused",
            preprocessing=args.preprocessing,
            select_res={"res_keys": args.res_keys},
            num_workers=args.num_workers,
            full_eval_every=args.full_eval_every,
            save_last_every=args.save_last_every,
            monitor=args.monitor,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
