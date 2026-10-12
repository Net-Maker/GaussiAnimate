#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PREFIX="$ROOT/.native"
PYTHON=${PYTHON:-python}
JOBS=${JOBS:-2}
INSTALL_CURVE=1
INSTALL_PYSSDR=1
INSTALL_PYTHON_DEPS=1
MODULE_DIR=

usage() {
    cat <<'EOF'
Usage: bash scripts/install_geometry.sh [options]

Build the small native backends used by Skelebone extraction.

Options:
  --prefix PATH       install native artifacts under PATH (default: .native)
  --python PATH       Python interpreter to receive pyssdr
  --module-dir PATH   override the Python extension install directory
  --jobs COUNT        parallel compiler jobs (default: 2)
  --curve-only        build only the CGAL curve_skeleton executable
  --pyssdr-only       build only the lightweight Dem Bones pyssdr module
  --no-python-deps    do not install the geometry Python extra or pybind11
  --help              show this message

System prerequisites:
  Ubuntu: sudo apt-get install build-essential cmake libcgal-dev libeigen3-dev python3-dev libx11-6 libgl1 git fonts-dejavu-core libegl1 libegl-mesa0
  macOS:  brew install cmake cgal eigen
EOF
}

while (($#)); do
    case "$1" in
        --prefix) PREFIX=$2; shift 2 ;;
        --python) PYTHON=$2; shift 2 ;;
        --module-dir) MODULE_DIR=$2; shift 2 ;;
        --jobs) JOBS=$2; shift 2 ;;
        --curve-only) INSTALL_PYSSDR=0; shift ;;
        --pyssdr-only) INSTALL_CURVE=0; shift ;;
        --no-python-deps) INSTALL_PYTHON_DEPS=0; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

command -v cmake >/dev/null || {
    echo "cmake is required. See --help for the platform install command." >&2
    exit 1
}
command -v "$PYTHON" >/dev/null || {
    echo "Python interpreter not found: $PYTHON" >&2
    exit 1
}

mkdir -p "$PREFIX"

if ((INSTALL_PYSSDR)); then
    if ((INSTALL_PYTHON_DEPS)); then
        if ((INSTALL_CURVE)); then
            "$PYTHON" -m pip install --no-build-isolation -e "$ROOT[geometry]"
        else
            "$PYTHON" -m pip install --no-build-isolation -e "$ROOT[ssdr]"
        fi
    fi
    "$PYTHON" -c 'import numpy as np, sys; sys.exit(0 if int(np.__version__.split(".")[0]) < 2 else "pyssdr requires NumPy < 2; install the GaussiAnimate dependencies first")'
    PYTHON_SITE=${MODULE_DIR:-$($PYTHON -c 'import sysconfig; print(sysconfig.get_path("platlib"))')}
    test -n "$PYTHON_SITE"
    mkdir -p "$PYTHON_SITE"
    cmake \
        -S "$ROOT/native/pyssdr" \
        -B "$PREFIX/build/pyssdr" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPython3_EXECUTABLE="$PYTHON"
    cmake --build "$PREFIX/build/pyssdr" --parallel "$JOBS"
    cmake --install "$PREFIX/build/pyssdr" --prefix "$PYTHON_SITE"
    PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON" -c 'import pyssdr; print("pyssdr:", pyssdr.__file__)'
fi

if ((INSTALL_CURVE)); then
    cmake \
        -S "$ROOT/native/curve_skeleton" \
        -B "$PREFIX/build/curve_skeleton" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX"
    cmake --build "$PREFIX/build/curve_skeleton" --parallel "$JOBS"
    cmake --install "$PREFIX/build/curve_skeleton"
    "$PREFIX/bin/curve_skeleton" --help >/dev/null
    echo "curve_skeleton: $PREFIX/bin/curve_skeleton"
fi

echo "Geometry backends are ready."
