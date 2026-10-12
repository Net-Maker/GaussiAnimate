"""Launch bundled reconstruction scripts with their upstream import layout."""

import argparse
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", choices=("train", "render"))
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    root = Path(__file__).resolve().parent
    path = root / ("train_gui.py" if args.script == "train" else "render.py")
    sys.path.insert(0, str(root))
    sys.argv = [str(path), *remaining]
    runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
