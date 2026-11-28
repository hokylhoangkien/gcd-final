"""Queue an epoch-boundary D1 stop or pause."""

import argparse
from pathlib import Path

from relational.control import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pause", "stop"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    request(args.root, args.action.upper(), variant="D1")


if __name__ == "__main__":
    main()
