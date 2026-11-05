"""Train the same CNN with mild dynamic augmentation on training images only."""

import argparse

from training_common import add_train_arguments, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_train_arguments(parser)
    parser.add_argument("--rotation-degrees", type=float, default=3,
                        help="maximum absolute train-only rotation (0 to 10; default 3)")
    args = parser.parse_args()
    try:
        run_training(args, "approach2", augment_train=True)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
