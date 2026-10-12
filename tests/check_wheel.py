"""Fail a release build if non-public research modules enter the wheel."""

import sys
from pathlib import Path
from zipfile import ZipFile


PUBLIC_ROOTS = {"partmm", "experiments", "skelebone"}
REQUIRED_CONFIGS = {
    "experiments/monocular_gaussian/config.yaml",
    "experiments/multiview_gaussian/config.yaml",
    "experiments/mesh/config.yaml",
    "experiments/catalog.yaml",
    "experiments/reproductions/catalog.yaml",
    "experiments/multiview_gaussian/methods/d3dgs_gt.yaml",
    "experiments/multiview_gaussian/methods/smpl_lbs.yaml",
    "experiments/multiview_gaussian/methods/smpl_bob.yaml",
    "experiments/multiview_gaussian/methods/smpl_partmm.yaml",
}
PRIVATE_PREFIXES = (
    "experiments/runs/",
    "experiments/multiview_gaussian/diagnostics/",
)
REQUIRED_LICENSE_FILES = {"LICENSE", "THIRD_PARTY_NOTICES.md"}


def audit(path: Path) -> None:
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        private = [
            name for name in names if name.startswith(PRIVATE_PREFIXES)
        ]
        package_roots = {
            name.split("/", 1)[0]
            for name in names
            if "/" in name and ".dist-info/" not in name
        }
        unexpected = package_roots - PUBLIC_ROOTS
        missing = REQUIRED_CONFIGS - names
        license_names = {
            name.rsplit("/", 1)[-1]
            for name in names
            if ".dist-info/" in name
        }
        missing_licenses = REQUIRED_LICENSE_FILES - license_names
        local_paths = [
            name
            for name in names
            if not name.endswith((".png", ".npz"))
            and b"/home/" in archive.read(name)
        ]
    if private:
        raise RuntimeError(f"wheel contains ignored local artifacts: {sorted(private)}")
    if unexpected:
        raise RuntimeError(f"unexpected wheel package roots: {sorted(unexpected)}")
    if missing:
        raise RuntimeError(f"missing experiment configs: {sorted(missing)}")
    if missing_licenses:
        raise RuntimeError(f"missing license files: {sorted(missing_licenses)}")
    if local_paths:
        raise RuntimeError(f"wheel contains local filesystem paths: {local_paths}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_wheel.py path/to/gaussianimate.whl")
    wheel = Path(sys.argv[1])
    audit(wheel)
    print(f"wheel boundary valid: {wheel}")
