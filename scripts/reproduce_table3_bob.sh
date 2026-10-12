#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULT_ROOT=${GA_MULTIVIEW_RELEASE_ROOT:?Set GA_MULTIVIEW_RELEASE_ROOT to the Table 3 run directory}
PYTHON_BIN=${GA_D3DGS_PYTHON:-python}

export GA_EXPERIMENT_RUNS_ROOT="$RESULT_ROOT/runs"
export GAUSSIANIMATE_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RESULT_ROOT/logs"
if ! "$PYTHON_BIN" -m experiments.paper \
  "$REPO_ROOT/experiments/multiview_gaussian/novel_pose.yaml" \
  --method smpl_bob --gpus 0 \
  > "$RESULT_ROOT/logs/table3_bob.log" 2>&1; then
  printf 'BoB failed; see %s\n' "$RESULT_ROOT/logs/table3_bob.log" >&2
  exit 1
fi

"$PYTHON_BIN" -m experiments.multiview_gaussian.table3_psnr \
  --root "$RESULT_ROOT" --method bob
