# SSDR Protocol Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the historical SSDR protocol for table reproduction and provide a robust point-cloud protocol that preserves requested bones with auditable diagnostics.

**Architecture:** `SSDRConfig.protocol` selects either `paper_legacy` or `motion_pointcloud`. Python owns protocol validation, motion initialization, strict bone-count checks and final diagnostics; the vendored DemBones binding exposes the small set of native initialization controls required to reproduce the June 9 historical behavior.

**Tech Stack:** Python 3, NumPy, pytest/unittest, C++14, Eigen, pybind11, DemBones.

**Spec:** `docs/ssdr_bone_collapse_analysis.md`

## Global Constraints

- `motion_pointcloud` is the public default and must initialize exactly the requested number of nonempty motion clusters.
- `paper_legacy` uses threshold 3, single-point split seeds and global best-error reassignment.
- Strict experiments fail when final and requested bone counts differ.
- Every saved experiment records requested, initialized, final and effective bone statistics.
- Existing `initialization="motion"` and `initialization="backend"` callers remain compatible.

---

### Task 1: Protocol contract and diagnostics

**Files:**
- Modify: `tests/test_skelebone_extraction.py`
- Modify: `skelebone/extraction/ssdr.py`
- Modify: `skelebone/pipeline.py`

**Interfaces:**
- Produces: `SSDRConfig.protocol`, `SSDRResult.diagnostics`, and protocol-aware `decompose_ssdr()`.

- [x] Write failing tests for protocol validation, legacy backend controls, exact motion initialization, and diagnostic counts.
- [x] Run the focused tests and verify failures are caused by the missing protocol contract.
- [x] Implement the minimal protocol resolver and diagnostics.
- [x] Run the focused tests to green.

### Task 2: Native historical and robust split controls

**Files:**
- Modify: `native/pyssdr/include/DemBones/DemBones.h`
- Modify: `native/pyssdr/pyssdr.cpp`
- Test: `tests/test_native_geometry.py`

**Interfaces:**
- Produces: native properties `clusterThreshold`, `labelAssignmentMode`, `splitSeedNeighbors`, `pruneDelay`, and `zeroGrowthPatience`.

- [x] Write a failing native contract test for all controls and binding exports.
- [x] Run it and verify the missing fields fail.
- [x] Implement configurable threshold, assignment, multi-point seed, delayed prune and zero-growth patience.
- [x] Build the vendored binding and run its small synthetic protocol checks.

### Task 3: Horse experiment entry and artifacts

**Files:**
- Modify: `tests/test_monocular_ssdr_evaluate.py`
- Modify: `experiments/monocular_gaussian/ssdr_evaluate.py`
- Modify: `experiments/monocular_gaussian/horse_ssdr.yaml`

**Interfaces:**
- Produces: working `run_ssdr(..., require_num_bones=...)` and NPZ diagnostic arrays.

- [x] Write a failing test that calls `run_ssdr` with an injected backend and verifies strict-count propagation.
- [x] Run it and confirm the current undefined variable fails.
- [x] Add the missing parameter, protocol config and diagnostic artifact fields.
- [x] Run the focused tests to green.

### Task 4: Documentation and reproducibility metadata

**Files:**
- Modify: `experiments/README.md`
- Modify: `docs/ssdr_bone_collapse_analysis.md`

**Interfaces:**
- Documents exact commands and the distinction between table reproduction and robust public extraction.

- [x] Document both protocol names and their intended use.
- [x] Document requested/initialized/final/effective bone fields.
- [x] Check Markdown and config formatting.

### Task 5: Verification

**Files:**
- Verify only.

**Interfaces:**
- Consumes all repaired protocol interfaces and artifacts.

- [x] Run SSDR and monocular focused tests.
- [x] Build the native binding from the vendored source.
- [x] Run the complete non-GPU test suite and report unrelated environment failures separately.
- [x] Refit the cached full Horse trajectory with `motion_pointcloud` and verify exactly 50 final bones.
- [x] Validate the saved diagnostic arrays and compare geometry RMSE with the previous repaired result.
