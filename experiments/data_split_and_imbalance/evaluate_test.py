"""Evaluate a selected checkpoint once on the held-out test split."""

import argparse
import json
from pathlib import Path

from model_integration import build_partner_model, resolve_factory_spec
from training_common import IMAGE_SIZE, evaluate, make_split_loader, require_torch, validate_model_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-factory", help="override the saved partner model factory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--after-selection", action="store_true", required=True,
                        help="confirm model/strategy selection is complete before test evaluation")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    try:
        torch, _, _, _, _ = require_torch()
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                              else "cpu" if args.device == "auto" else args.device)
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(args.checkpoint, map_location=device)
        if checkpoint.get("image_size") != list(IMAGE_SIZE):
            raise ValueError("Checkpoint image size does not match the current 224x224 evaluation pipeline.")
        if checkpoint.get("label_to_id") != {"NORMAL": 0, "PNEUMONIA": 1}:
            raise ValueError("Checkpoint label mapping does not match NORMAL=0, PNEUMONIA=1.")
        factory_spec = resolve_factory_spec(args.model_factory or checkpoint.get("model_factory"))
        model = build_partner_model(factory_spec).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        validate_model_output(model, torch, device)
        torch, loader, _ = make_split_loader(args.manifest, args.dataset_dir, "test", args.batch_size)
        result = evaluate(model, loader, torch, device, torch.nn.CrossEntropyLoss())
        result.update({"evaluated_split": "test", "checkpoint": str(args.checkpoint),
                       "image_size": list(IMAGE_SIZE), "positive_class": "PNEUMONIA"})
        print(json.dumps(result, indent=2))
        print("Use this held-out result for final reporting only; do not select or tune models with it.")
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
