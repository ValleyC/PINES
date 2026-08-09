from __future__ import annotations

import argparse

from pines.reproduce import run_toy_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    path = run_toy_study(args.output, args.samples, args.seed)
    print(path)


if __name__ == "__main__":
    main()

