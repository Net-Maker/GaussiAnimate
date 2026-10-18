# Third-party and provenance notices for `experiments.backends.partmm_research`

This package vendors a bounded, minimal closure of research source code so
the PartMM experiment stages run as normal packaged imports without an
external implementation checkout on `sys.path`. Local changes include
packaged imports and bug fixes; the original source attribution is retained.

## Author-owned research code (author's `smpl_to_ssdr` snapshot)

The files under `core/`, `math/`, `patch/`, `pyramid/`, `reconstruction/`,
`skeletal/`, and `utils/` are the author's own research code, copied from the
uncommitted local research snapshot historically kept outside this repository
(the `smpl_to_ssdr` package).

- **License status: no license file or license statement was found with the
  snapshot.** The code is copyrighted by its author and may be redistributed
  only with the author's permission. It is vendored here for the authorized
  research use of this repository; no open-source license is claimed or
  implied. This must be resolved (e.g. by freezing the snapshot under a
  redistributable license) before the repository can be open-sourced.
- Unused `io` / `visualization` modules, the adaptive-partition utilities,
  and `core/adaptive_mask.py` were **not** copied.
  `SMPLToSSDRMatcher.build_combined_target_database`, the legacy
  combined-target path that required `core.adaptive_mask`, is not on the
  part-based retargeting path and now raises `NotImplementedError` saying so.
- The debug-only `DpViewer` visualization helper
  (`skeletal/part_binding.py::visualize_binding_debug`) was removed rather
  than vendored.

## Motion2Motion code (https://github.com/LinghaoChan/Motion2Motion_codes)

- `_thirdparty_motion.py` — copy of `dataset/motion.py` from the Motion2Motion
  code release (SIGGRAPH Asia 2025 paper "Motion2Motion: Cross-topology
  Motion Transfer with Sparse Correspondence"). Unmodified except a
  provenance docstring.
- `_thirdparty_nearest_neighbor.py` — copy of `nearest_neighbor/utils.py`
  from the same release. Unmodified except a provenance docstring.

- **License status: the Motion2Motion repository carries no LICENSE file and
  no license statement in its README** (verified against the local copy and
  the upstream repository page). The code is under default copyright;
  redistribution requires the authors' permission. Redistribution status is
  **unresolved**; the files are vendored here for authorized research use
  only, and no open-source license is claimed or implied.

## Efficient-GPNN borrowed code (Apache-2.0)

`_thirdparty_nearest_neighbor.py` retains the original in-file notice: it
borrows code from
https://github.com/ariel415el/Efficient-GPNN/blob/main/utils/NN.py
(`efficient_cdist`, `get_col_mins_efficient`, `get_NNs_Dists`, and the patch
extraction approach). Per the upstream repository, Efficient-GPNN is
**Apache-2.0 licensed**; the original attribution notice is preserved in the
file header above.

## External runtime dependencies (not vendored)

- Python packages: `torch`, `numpy`, `scipy`, `tqdm`, `unfoldNd`, `smplx`
- Licensed SMPL body model data (e.g. `SMPL_NEUTRAL.pkl`), loaded at runtime
  from an external directory configured explicitly via
  `experiments.backends.partmm_research.set_smpl_model_dir`. SMPL model files
  are subject to their own license from the Max Planck Institute and must not
  be redistributed with this code.
