# Non-Novel-View Table Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recompute and package every quantitative camera-ready GaussiAnimate table except novel-view synthesis, without presenting copied paper values or incomplete case coverage as reproduced results.

**Architecture:** Keep the existing declarative experiment engine. Add one focused experiment config and evaluator for each missing ablation table, normalize the camera-ready table identities, and use immutable raw artifacts as evaluator inputs. Table 3 remains a staged D3DGS -> animation -> render -> metrics pipeline; downstream jobs run only when their checkpoint-backed inputs validate.

**Tech Stack:** Python 3, NumPy, PyYAML, pytest, existing `experiments.engine`, D3DGS, HOOD/VTO, PartMM.

**Spec:** `/home/ubuntu/Downloads/GaussiAnimate_formal_camera_ready_v40.pdf`

## Global Constraints

- Exclude Table 2, novel-view synthesis, from this execution scope.
- Reproduce Table 3, Table 4, Table 5, and supplemental Table R.1 from executable code and declared inputs.
- Treat Table 1 as a literature comparison, not an experiment.
- Never substitute a paper value when a recomputed value differs.
- Record configuration, dataset identity, seed, source artifact, metrics, and provenance for every result.
- Do not average incomplete case coverage.

---

### Task 1: Camera-Ready Experiment Matrix

**Files:**
- Modify: `experiments/catalog.yaml`
- Modify: `experiments/README.md`
- Modify: `experiments/multiview_gaussian/novel_pose.yaml`
- Modify: `experiments/multiview_gaussian/reconstruction.yaml`
- Modify: `experiments/mesh/vto_animation.yaml`
- Modify: `experiments/mesh/dt4d_animation.yaml`
- Test: `tests/test_paper_experiments.py`

**Interfaces:**
- Consumes: camera-ready table captions and protocols from the PDF.
- Produces: stable target IDs mapped to the correct Table 3 and Table 4 labels.

- [ ] Update stale arXiv-v1 table and figure numbers to camera-ready-v40 identities.
- [ ] Add tests asserting that Table 2 is the only novel-view target and that all quantitative non-NVS targets are declared.
- [ ] Run `pytest -q tests/test_paper_experiments.py`.

### Task 2: Table 5 Patch-Length Ablation

**Files:**
- Create: `experiments/mesh/patch_length.yaml`
- Create: `experiments/mesh/patch_length.py`
- Modify: `experiments/catalog.yaml`
- Test: `tests/test_paper_experiments.py`

**Interfaces:**
- Consumes: five-case VTO T-shirt PartMM inputs and patch lengths `[3, 5, 7, 11, 15]`.
- Produces: one `geometry_metrics.json` per patch length and a coverage-aware aggregate matching Table 5's schema.

- [ ] Write tests for matrix size, exact patch lengths, artifact validation, and recomputed means.
- [ ] Verify the new tests fail before implementation.
- [ ] Implement an evaluator that reads per-case numerical records, verifies the frozen five-case set, recomputes the mean, and records source SHA-256.
- [ ] Add the schema-v2 config and catalog entry.
- [ ] Run the evaluator against recovered artifacts when available; otherwise record the exact missing raw artifact rather than a paper number.
- [ ] Run `pytest -q tests/test_paper_experiments.py`.

### Task 3: Supplemental Table R.1 Database Coverage and Cost

**Files:**
- Create: `experiments/mesh/database_scaling.yaml`
- Create: `experiments/mesh/database_scaling.py`
- Modify: `experiments/catalog.yaml`
- Test: `tests/test_paper_experiments.py`

**Interfaces:**
- Consumes: VTO scalability outputs for 5, 20, and 50 training sequences, SSDR cache identities, and runtime records.
- Produces: RMSE, minimum SO(3) distance, build time, prediction time, GPU memory, and provenance for each database size.

- [ ] Write tests for exact sequence counts and all five reported metrics.
- [ ] Verify the tests fail before implementation.
- [ ] Implement strict parsing and cross-file consistency checks for the recovered scalability records.
- [ ] Add the schema-v2 config and catalog entry.
- [ ] Recompute and package Table R.1 from the recovered source records.
- [ ] Run `pytest -q tests/test_paper_experiments.py`.

### Task 4: Table 4 Mesh Reproduction

**Files:**
- Modify: `experiments/mesh/vto_animation.yaml`
- Modify: `experiments/mesh/dt4d_animation.yaml`
- Modify: `experiments/catalog.yaml`
- Outputs: `experiments/runs/vto_animation/**/geometry_metrics.json`
- Outputs: `experiments/runs/dt4d_animation/**/geometry_metrics.json`

**Interfaces:**
- Consumes: frozen VTO per-case arrays and D4D benchmark records.
- Produces: recomputed Table 4 values with paper deltas and explicit protocol classifications.

- [ ] Validate both configs and enumerate the exact job matrix.
- [ ] Run all VTO methods from packaged per-case arrays.
- [ ] Run all D4D methods from the hash-frozen 26-instance artifact.
- [ ] Generate coverage-aware summaries and preserve deviations from the paper.
- [ ] Run focused historical-result tests.

### Task 5: Table 3 Clothed-Human Reproduction

**Files:**
- Modify: `experiments/multiview_gaussian/resources.yaml`
- Modify: `experiments/multiview_gaussian/novel_pose.yaml`
- Modify: `experiments/catalog.yaml`
- Outputs: `${GA_EXPERIMENT_RUNS_ROOT}/multiview_novel_pose/**`

**Interfaces:**
- Consumes: 8 DNA and the declared ActorsHQ case set, D3DGS `params.npz`, SMPL/SMPL-X poses, cameras, masks, and method caches.
- Produces: D3DGS-GT, SMPL-LBS, SMPL-BoB, and SMPL-PartMM held-out rendering metrics.

- [ ] Run full resource validation and record checkpoint coverage.
- [ ] Finish missing ActorsHQ D3DGS checkpoints once CUDA is available.
- [ ] Run strict LBS, v2 BoB, training-only SSDR, and paper-spec PartMM for every checkpoint-backed case.
- [ ] Render only the held-out 20 percent split and evaluate PSNR, SSIM, and LPIPS.
- [ ] Aggregate only after every declared case is complete; separately report the unresolved paper 8-subject versus recovered 10-sequence identity mismatch.

### Task 6: Release Audit

**Files:**
- Modify: `experiments/catalog.yaml`
- Modify: `experiments/README.md`
- Test: `tests/test_experiment_audit.py`
- Test: `tests/test_reproductions.py`

**Interfaces:**
- Consumes: completed job manifests and table summaries from Tasks 1-5.
- Produces: one current, honest release-readiness report for the non-NVS tables.

- [ ] Update every target with recomputed values, coverage, source hashes, and remaining blockers.
- [ ] Add an audit mode that can exclude `synthetic_novel_view` from the readiness gate.
- [ ] Run the full test suite.
- [ ] Run the non-NVS release audit and preserve its machine-readable output.
