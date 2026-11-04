"""Train one imbalance strategy at a time; test data is never loaded here."""

import argparse

from training_common import STRATEGIES, add_training_arguments, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True,
                        help="choose class-weighted-loss OR weighted-sampling")
    args = parser.parse_args()
    try:
        train(args, approach="approach1", strategy=args.strategy)
    except (RuntimeError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
