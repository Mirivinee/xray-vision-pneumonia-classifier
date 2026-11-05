"""Train with one imbalance strategy: class-weighted loss OR weighted sampling."""

import argparse

from training_common import STRATEGIES, add_train_arguments, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_train_arguments(parser)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True,
                        help="select exactly one strategy per run")
    args = parser.parse_args()
    try:
        run_training(args, "approach1", strategy=args.strategy)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
