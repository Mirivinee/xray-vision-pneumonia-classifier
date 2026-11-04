"""Train with configurable mild rotation augmentation on training images only."""

import argparse

from training_common import add_training_arguments, train, validate_rotation_degrees


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    parser.add_argument("--rotation-degrees", type=float, default=3,
                        help="maximum absolute train-only rotation (0 to 10 degrees; default 3)")
    args = parser.parse_args()
    try:
        validate_rotation_degrees(args.rotation_degrees)
        train(args, approach="approach2", augment_train=True, rotation_degrees=args.rotation_degrees)
    except (RuntimeError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
