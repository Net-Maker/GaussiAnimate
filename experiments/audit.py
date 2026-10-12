"""Report whether the checked-in experiment matrix meets the paper protocol."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


VALID_STATES = {"verified", "partial", "in_progress", "blocked"}
TARGET_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


def audit_catalog(
    path: Path,
    exclude_target_ids: Iterable[str] = (),
) -> Dict[str, Any]:
    path = Path(path)
    catalog = yaml.safe_load(path.read_text(encoding="utf-8"))
    excluded = set(exclude_target_ids)
    records = []
    target_ids = set()
    for entry in catalog.get("experiments", []):
        target_id = entry.get("target_id")
        if not isinstance(target_id, str) or not TARGET_ID_PATTERN.fullmatch(
            target_id
        ):
            raise ValueError(
                f"{entry.get('target', '<unknown>')} has invalid target_id"
            )
        if target_id in target_ids:
            raise ValueError(f"duplicate target_id: {target_id}")
        target_ids.add(target_id)
        readiness = entry.get("readiness")
        if not isinstance(readiness, dict):
            raise ValueError(f"{entry.get('target', '<unknown>')} has no readiness record")
        state = readiness.get("state")
        if state not in VALID_STATES:
            raise ValueError(f"{entry['target']} has invalid readiness state {state!r}")
        for field in ("verified", "blockers"):
            statements = readiness.get(field, [])
            if not isinstance(statements, list) or any(
                not isinstance(statement, str) for statement in statements
            ):
                raise ValueError(
                    f"{target_id} readiness.{field} must be a list of strings"
                )
        config = path.parent / entry["config"]
        config_target_id = None
        if config.is_file():
            raw_config = yaml.safe_load(config.read_text(encoding="utf-8"))
            config_target_id = raw_config.get("experiment", {}).get(
                "target_id"
            )
            if config_target_id != target_id:
                raise ValueError(
                    f"{target_id} does not match {config}: "
                    f"{config_target_id!r}"
                )
        records.append(
            {
                "target_id": target_id,
                "target": entry["target"],
                "state": state,
                "config": str(config),
                "config_exists": config.is_file(),
                "verified": list(readiness.get("verified", [])),
                "blockers": list(readiness.get("blockers", [])),
            }
        )
    unknown_exclusions = excluded - target_ids
    if unknown_exclusions:
        raise ValueError(
            f"unknown excluded target ids: {', '.join(sorted(unknown_exclusions))}"
        )
    records = [
        record for record in records if record["target_id"] not in excluded
    ]
    missing_configs = [record["target"] for record in records if not record["config_exists"]]
    state_counts = {
        state: sum(record["state"] == state for record in records)
        for state in sorted(VALID_STATES)
    }
    ready = bool(records) and not missing_configs and all(
        record["state"] == "verified" for record in records
    )
    declared_ready = bool(catalog.get("audit", {}).get("overall_ready", False))
    if not excluded and declared_ready != ready:
        raise ValueError(
            f"catalog audit.overall_ready={declared_ready} disagrees with computed readiness={ready}"
        )
    return {
        "paper": catalog.get("paper"),
        "as_of": str(catalog.get("audit", {}).get("as_of", "")),
        "ready": ready,
        "targets": len(records),
        "state_counts": state_counts,
        "missing_configs": missing_configs,
        "excluded_target_ids": sorted(excluded),
        "records": records,
    }


def _print_report(report: Dict[str, Any]) -> None:
    verdict = "READY" if report["ready"] else "NOT READY"
    print(f"{report['paper']} paper reproduction: {verdict} ({report['as_of']})")
    for record in report["records"]:
        print(
            f"[{record['state']}] {record['target_id']}: "
            f"{record['target']}"
        )
        for blocker in record["blockers"]:
            print(f"  - {blocker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit paper experiment readiness")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(__file__).resolve().parent / "catalog.yaml",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--exclude-target",
        action="append",
        default=[],
        help="Exclude one target_id from this audit gate; may be repeated.",
    )
    parser.add_argument(
        "--non-novel-view",
        action="store_true",
        help="Exclude the Table 2 synthetic_novel_view target.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit unsuccessfully unless every paper target is verified.",
    )
    args = parser.parse_args()
    excluded = set(args.exclude_target)
    if args.non_novel_view:
        excluded.add("synthetic_novel_view")
    report = audit_catalog(args.catalog, exclude_target_ids=excluded)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
    if args.require_ready and not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
