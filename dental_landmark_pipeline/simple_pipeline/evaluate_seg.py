"""
evaluate_seg.py
---------------
Compute segmentation evaluation metrics on a held-out test set.

Metrics reported (matching Teeth3DS benchmark):
    OA    — Overall Accuracy
    mACC  — Mean Class Accuracy
    mIoU  — Mean Intersection over Union  (primary metric)

Usage:
    python evaluate_seg.py \
        --data-root   data/teeth3ds \
        --split-file  splits/test.txt \
        --ckpt        checkpoints/seg/best.pt \
        --batch-size  2

Output is printed to stdout and optionally saved as a JSON file.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from datasets.seg_dataset import TeethSegDataset
from models.seg_model import PointNet2SegModel
from utils.metrics import SegMetrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Stage-1 segmentation")
    p.add_argument("--data-root",  required=True)
    p.add_argument("--split-file", default=None)
    p.add_argument("--ckpt",       required=True, help="Path to checkpoint (.pt)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--npoints",    type=int, default=10_000)
    p.add_argument("--num-workers",type=int, default=4)
    p.add_argument("--out",        default=None, help="Save results to this JSON path")
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    ds = TeethSegDataset(
        root=args.data_root,
        split_file=args.split_file,
        npoints=args.npoints,
        augment=False,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Test set: {len(ds)} scans")

    # Model
    model = PointNet2SegModel(num_classes=17, in_channels=3, dropout=0.0).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # Evaluate
    metrics = SegMetrics(num_classes=17)
    with torch.no_grad():
        for i, (xyz, normals, labels) in enumerate(loader):
            xyz     = xyz.to(device)
            normals = normals.to(device)
            labels  = labels.to(device)
            logits  = model(xyz, normals)        # (B, 17, N)
            preds   = logits.argmax(dim=1)       # (B, N)
            metrics.update(preds, labels)
            if (i + 1) % 10 == 0:
                print(f"  Processed {(i+1)*args.batch_size}/{len(ds)} scans …", flush=True)

    results = metrics.compute()

    # Per-class IoU breakdown
    conf = metrics._conf
    import numpy as np
    tp   = np.diag(conf)
    fn   = conf.sum(axis=1) - tp
    fp   = conf.sum(axis=0) - tp
    denom = tp + fp + fn
    iou_per_class = np.where(denom > 0, tp / denom, float("nan"))

    print("\n" + "="*55)
    print(f"  Overall Accuracy  (OA)  : {results['OA']:.4f}")
    print(f"  Mean Class Accuracy     : {results['mACC']:.4f}")
    print(f"  Mean IoU           (mIoU): {results['mIoU']:.4f}")
    print("="*55)

    class_names = ["gingiva"] + [f"FDI_{i}" for i in range(1, 17)]
    print("\nPer-class IoU:")
    for i, name in enumerate(class_names):
        iou = iou_per_class[i]
        flag = "" if not (iou < 0.5) else "  ← low"
        print(f"  [{i:2d}] {name:<12s}  IoU={iou:.4f}{flag}")

    if args.out:
        out = {
            "OA":   results["OA"],
            "mACC": results["mACC"],
            "mIoU": results["mIoU"],
            "iou_per_class": iou_per_class.tolist(),
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
