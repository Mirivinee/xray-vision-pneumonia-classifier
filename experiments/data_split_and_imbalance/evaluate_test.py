"""Evaluate one already selected checkpoint on the held-out test manifest rows."""

import argparse
import json
from pathlib import Path

import torch

from training_common import DEFAULT_MANIFEST, make_test_loader, run_epoch
from model_integration import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--after-selection", action="store_true", required=True,
                        help="confirm model selection is complete before test evaluation")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    try:
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                              else "cpu" if args.device == "auto" else args.device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        config = checkpoint["config"]
        if config.get("label_to_id") != {"NORMAL": 0, "PNEUMONIA": 1}:
            raise ValueError("Checkpoint label mapping does not match NORMAL=0, PNEUMONIA=1")
        if config.get("input_shape") != [3, 224, 224] or config.get("output_shape_per_batch") != "[B] one raw PNEUMONIA logit":
            raise ValueError("Checkpoint model input/output contract does not match this evaluator")
        model = build_model(dropout=config["dropout"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        test_loader, test_rows = make_test_loader(
            args.manifest, args.dataset_dir, args.batch_size, args.num_workers
        )
        pos_weight_value = config.get("pos_weight")
        pos_weight = (torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
                      if pos_weight_value is not None else None)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        metrics = run_epoch(model, test_loader, criterion, device)
        result = {"test_images": len(test_rows), "test_metrics": metrics,
                  "checkpoint": str(args.checkpoint), "positive_label": "PNEUMONIA",
                  "confusion_matrix_order": ["NORMAL", "PNEUMONIA"]}
        print(json.dumps(result, indent=2))
        print("Final test evaluation only. Do not use these results to tune or select models.")
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
