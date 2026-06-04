#!/usr/bin/env python3
"""
Run MR-YOLO inference on a dataset split and compute detection metrics.

Outputs a JSON file with mAP50, mAP50:95, precision, recall, and
per-SNR-bin detection probability.

Examples
--------
  # Evaluate on val split
  python predict.py \\
      --checkpoint runs/mr_yolo_n_cfg512_cfg256/best.pt \\
      --data-dir /data/rf_dataset \\
      --res-keys cfg512 cfg256

  # Evaluate on test split, custom output path
  python predict.py \\
      --checkpoint runs/mr_yolo_n_cfg512_cfg256/best.pt \\
      --data-dir /data/rf_dataset \\
      --res-keys cfg512 cfg256 \\
      --split test \\
      --output-json results/mr_yolo_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mr_yolo import MR_YOLO
from mr_yolo.utils.analysing_results import dataset_analysis_with_metrics
from mr_yolo.utils.dataset import YOLODatasetFusedMultiRes, load_class_index_to_name
from mr_yolo.utils.preprocess import preprocessing_num_channels

Resolution = Tuple[int, int]
SCALE_WIDTH = {"n": 0.25, "s": 0.50, "m": 0.75}

def detect_resolutions(data_dir: str, res_keys: List[str], split: str) -> Dict[str, Resolution]:
    images_dir = Path(data_dir) / split / "data"
    pt_file = next(images_dir.glob("*.pt"), None)
    if pt_file is None:
        raise FileNotFoundError(f"No .pt file found in {images_dir}")
    specs = torch.load(pt_file, map_location="cpu")
    if not isinstance(specs, list):
        raise ValueError(f"Expected a list of tensors in {pt_file}")
    if len(specs) != len(res_keys):
        raise ValueError(
            f"Dataset has {len(specs)} resolution tensors but "
            f"{len(res_keys)} res_keys provided: {res_keys}."
        )
    return {key: (int(s.shape[-2]), int(s.shape[-1])) for key, s in zip(res_keys, specs)}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MR-YOLO inference and evaluate detection metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    p.add_argument("--data-dir", required=True, help="Dataset root directory.")
    p.add_argument(
        "--res-keys", nargs="+", required=True,
        help="Ordered resolution keys (must match those used at training time).",
    )
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument(
        "--scale", default="n", choices=["n", "s", "m"],
        help="Must match the scale used at training time.",
    )
    p.add_argument("--num-classes", type=int, default=20)
    p.add_argument("--reg-max", type=int, default=16)
    p.add_argument("--preprocessing", default="none")
    p.add_argument("--iou-thresh", type=float, default=0.5)
    p.add_argument("--false-alarm-target", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--output-json", default=None,
        help="Path for the JSON results file. "
             "Defaults to <checkpoint_dir>/eval_<split>.json.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    resolutions = detect_resolutions(args.data_dir, args.res_keys, args.split)
    input_channels = preprocessing_num_channels(args.preprocessing)
    img_size = max(resolutions.values(), key=lambda hw: hw[0] * hw[1])

    print("MR-YOLO Evaluation")
    print(f"  Checkpoint  : {checkpoint}")
    print(f"  Split       : {args.split}")
    print(f"  Device      : {args.device}")
    print(f"  Resolutions :")
    for key, hw in resolutions.items():
        print(f"    {key}: {hw[0]}×{hw[1]}")

    # Build model
    print("\n[1/4] Building model...")
    model = MR_YOLO(
        input_resolutions=list(resolutions.values()),
        output_dir=str(checkpoint.parent),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        in_ch=input_channels,
        width_mult=SCALE_WIDTH[args.scale],
    )

    # Load weights
    print("[2/4] Loading weights...")
    missing, unexpected = model.load_weights(str(checkpoint), device=args.device, eval_mode=True)
    if missing:
        print(f"  [warning] {len(missing)} missing key(s)")
    if unexpected:
        print(f"  [warning] {len(unexpected)} unexpected key(s)")
    model.eval()

    # Build dataloader
    print("[3/4] Building dataloader...")
    dataset = YOLODatasetFusedMultiRes(
        data_dir=str(Path(args.data_dir) / args.split / "data"),
        labels_dir=str(Path(args.data_dir) / args.split / "labels_detect"),
        res_keys=tuple(args.res_keys),
        preprocessing=args.preprocessing,
    )
    num_workers = max(0, args.num_workers)
    loader_kwargs: dict = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": dataset.collate_fn,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    print(f"  {len(dataset)} samples | batch={args.batch_size}")

    # Evaluate
    print("[4/4] Evaluating...")
    metrics = dataset_analysis_with_metrics(
        model=model,
        val_loader=loader,
        iou_thresh=args.iou_thresh,
        fa=args.false_alarm_target,
        img_size=img_size,
        to_save=False,
        to_plot=False,
        class_index_to_name=load_class_index_to_name(args.data_dir),
    )

    # Save results
    output_json = Path(args.output_json) if args.output_json else (
        checkpoint.parent / f"eval_{args.split}.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint": str(checkpoint),
        "dataset": args.data_dir,
        "split": args.split,
        "device": args.device,
        "model": {
            "scale": args.scale,
            "num_classes": args.num_classes,
            "reg_max": args.reg_max,
            "preprocessing": args.preprocessing,
            "res_keys": args.res_keys,
            "input_resolutions": {k: list(v) for k, v in resolutions.items()},
        },
        "eval": {
            "iou_thresh": args.iou_thresh,
            "false_alarm_target": args.false_alarm_target,
            "batch_size": args.batch_size,
        },
        "metrics": metrics,
    }

    output_json.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )

    map_stats = metrics.get("map_stats", {})
    print("\n[Done]")
    print(f"  Results : {output_json}")
    print(f"  mAP50   : {map_stats.get('mAP50')}")
    print(f"  mAP50:95: {map_stats.get('mAP50:95')}")


if __name__ == "__main__":
    main()
