"""Compose paper qualitative figures from declared historical image panels."""

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from PIL import Image, ImageDraw, ImageFont


ENV_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}"
)
BACKGROUND_POLICIES = {"alpha_to_white", "preserve"}
SOURCE_POLICIES = {
    "historical_only",
    "historical_or_explicit_external",
    "explicit_external",
}


class FigureBlockedError(RuntimeError):
    """Raised when a declared figure cannot be built from available inputs."""

    def __init__(self, report: Dict[str, Any]):
        self.report = report
        reasons = report.get("blockers", ["unknown input failure"])
        super().__init__("figure blocked: " + "; ".join(reasons))


def _expand_environment(value: Any, require: bool) -> Any:
    if isinstance(value, dict):
        return {
            key: _expand_environment(item, require)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_expand_environment(item, require) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        if require:
            raise ValueError(
                "required environment variable is unset: {}".format(name)
            )
        return match.group(0)

    return os.path.expanduser(ENV_PATTERN.sub(replace, value))


def _ids(items: List[Dict[str, Any]], label: str) -> List[str]:
    values = [item.get("id") for item in items]
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("every {} must have a non-empty id".format(label))
    duplicates = sorted(
        {value for value in values if values.count(value) > 1}
    )
    if duplicates:
        raise ValueError(
            "duplicate {} ids: {}".format(label, ", ".join(duplicates))
        )
    return values


def _panel_path(panel: Any) -> Optional[str]:
    if isinstance(panel, str):
        return panel
    if isinstance(panel, dict):
        return panel.get("path")
    return None


def validate_figure_manifest(manifest: Dict[str, Any]) -> None:
    """Validate layout and exact row/column coverage without reading inputs."""

    if manifest.get("schema_version") != 1:
        raise ValueError("qualitative figures require schema_version: 1")
    if manifest.get("kind") != "qualitative_figure":
        raise ValueError("qualitative figures require kind: qualitative_figure")
    figure = manifest.get("figure")
    if not isinstance(figure, dict):
        raise ValueError("figure must be a mapping")
    for key in ("id", "paper_target", "source_policy"):
        if not figure.get(key):
            raise ValueError("figure.{} is required".format(key))
    if figure["source_policy"] not in SOURCE_POLICIES:
        raise ValueError(
            "figure.source_policy must declare historical/external inputs"
        )
    methods = manifest.get("methods")
    rows = manifest.get("rows")
    if not isinstance(methods, list) or not methods:
        raise ValueError("methods must be a non-empty list")
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list")
    method_ids = _ids(methods, "method")
    row_ids = _ids(rows, "row")
    if any(not item.get("label") for item in methods):
        raise ValueError("every method must have a label")
    if any(not item.get("label") for item in rows):
        raise ValueError("every row must have a label")

    layout = manifest.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("layout must be a mapping")
    if layout.get("method_order") != method_ids:
        raise ValueError("layout.method_order must match methods exactly")
    if layout.get("row_order") != row_ids:
        raise ValueError("layout.row_order must match rows exactly")
    if layout.get("background") not in ("white", "#ffffff", "#fff"):
        raise ValueError("qualitative figure background must be white")
    cell_size = layout.get("cell_size")
    if (
        not isinstance(cell_size, list)
        or len(cell_size) != 2
        or any(not isinstance(value, int) or value <= 0 for value in cell_size)
    ):
        raise ValueError("layout.cell_size must contain two positive integers")
    orientation = layout.get("row_label_orientation", "horizontal")
    if orientation not in ("horizontal", "vertical"):
        raise ValueError(
            "layout.row_label_orientation must be horizontal or vertical"
        )

    panels = manifest.get("panels")
    if not isinstance(panels, dict):
        raise ValueError("panels must be a mapping")
    if set(panels) != set(row_ids):
        raise ValueError("panels must define every declared row exactly")
    for row_id in row_ids:
        cells = panels[row_id]
        if not isinstance(cells, dict) or set(cells) != set(method_ids):
            raise ValueError(
                "panels.{} must define every method exactly".format(row_id)
            )
        for method_id in method_ids:
            panel = cells[method_id]
            if panel is None:
                continue
            if isinstance(panel, str):
                if not panel:
                    raise ValueError("panel path cannot be empty")
                continue
            if not isinstance(panel, dict) or not panel.get("path"):
                raise ValueError(
                    "panel {}/{} must be a path, mapping, or null".format(
                        row_id,
                        method_id,
                    )
                )
            background = panel.get("background", "alpha_to_white")
            if background not in BACKGROUND_POLICIES:
                raise ValueError(
                    "panel {}/{} has invalid background policy".format(
                        row_id,
                        method_id,
                    )
                )
            crop = panel.get("crop_fraction")
            if crop is not None:
                if (
                    not isinstance(crop, list)
                    or len(crop) != 4
                    or any(not isinstance(value, (int, float)) for value in crop)
                    or not (
                        0 <= crop[0] < crop[2] <= 1
                        and 0 <= crop[1] < crop[3] <= 1
                    )
                ):
                    raise ValueError(
                        "panel {}/{} has invalid crop_fraction".format(
                            row_id,
                            method_id,
                        )
                    )

    output = manifest.get("output")
    if not isinstance(output, dict):
        raise ValueError("output must be a mapping")
    if not output.get("image") or not output.get("provenance"):
        raise ValueError("output.image and output.provenance are required")


def _resolve_path(value: str, manifest_dir: Path) -> str:
    if ENV_PATTERN.search(value):
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return str(path.resolve())


def load_figure_manifest(
    path: Path,
    require_environment: bool = False,
) -> Dict[str, Any]:
    """Load, expand, validate, and resolve a qualitative figure manifest."""

    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("figure manifest must be a mapping: {}".format(path))
    manifest = _expand_environment(manifest, require_environment)
    validate_figure_manifest(manifest)
    for cells in manifest["panels"].values():
        for method_id, panel in cells.items():
            if isinstance(panel, str):
                cells[method_id] = _resolve_path(panel, path.parent)
            elif isinstance(panel, dict):
                panel["path"] = _resolve_path(panel["path"], path.parent)
    for key in ("image", "provenance"):
        manifest["output"][key] = _resolve_path(
            manifest["output"][key],
            path.parent,
        )
    reference = manifest.get("reference")
    if isinstance(reference, dict) and reference.get("path"):
        reference["path"] = _resolve_path(reference["path"], path.parent)
    manifest["_manifest_path"] = str(path)
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_figure_manifest(
    manifest: Dict[str, Any],
    check_inputs: bool = True,
) -> Dict[str, Any]:
    """Return ready or blocked with per-panel input evidence."""

    validate_figure_manifest(manifest)
    inputs = []
    blockers = []
    for row in manifest["rows"]:
        row_id = row["id"]
        for method in manifest["methods"]:
            method_id = method["id"]
            panel = manifest["panels"][row_id][method_id]
            path_value = _panel_path(panel)
            if path_value is None:
                continue
            source = {
                "row": row_id,
                "method": method_id,
                "path": path_value,
                "background": (
                    panel.get("background", "alpha_to_white")
                    if isinstance(panel, dict)
                    else "alpha_to_white"
                ),
                "crop_fraction": (
                    panel.get("crop_fraction")
                    if isinstance(panel, dict)
                    else None
                ),
            }
            if ENV_PATTERN.search(path_value):
                source["status"] = "unresolved_environment"
                blockers.append(
                    "unresolved input {}/{}: {}".format(
                        row_id,
                        method_id,
                        path_value,
                    )
                )
                inputs.append(source)
                continue
            if not check_inputs:
                source["status"] = "declared"
                inputs.append(source)
                continue
            path = Path(path_value)
            if not path.is_file():
                source["status"] = "missing"
                blockers.append(
                    "missing input {}/{}: {}".format(
                        row_id,
                        method_id,
                        path,
                    )
                )
                inputs.append(source)
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                    mode = image.mode
                if width <= 0 or height <= 0:
                    raise ValueError("image has zero extent")
                source.update(
                    status="ready",
                    bytes=path.stat().st_size,
                    sha256=_sha256(path),
                    width=width,
                    height=height,
                    mode=mode,
                )
            except Exception as exc:
                source.update(status="unreadable", error=repr(exc))
                blockers.append(
                    "unreadable input {}/{}: {}".format(
                        row_id,
                        method_id,
                        path,
                    )
                )
            inputs.append(source)
    return {
        "schema_version": 1,
        "figure_id": manifest["figure"]["id"],
        "status": "blocked" if blockers else "ready",
        "ready": not blockers,
        "blockers": blockers,
        "inputs": inputs,
    }


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    return draw.textsize(text, font=font)


def _centered_text(
    draw: ImageDraw.ImageDraw,
    bounds: Tuple[int, int, int, int],
    text: str,
    font,
) -> None:
    width, height = _text_size(draw, text, font)
    left, top, right, bottom = bounds
    draw.text(
        (
            left + (right - left - width) // 2,
            top + (bottom - top - height) // 2,
        ),
        text,
        fill=(24, 28, 31),
        font=font,
    )


def _open_panel(panel: Any) -> Image.Image:
    path = Path(_panel_path(panel))
    options = panel if isinstance(panel, dict) else {}
    with Image.open(path) as source:
        source.load()
        image = source.convert("RGBA")
    crop = options.get("crop_fraction")
    if crop is not None:
        width, height = image.size
        image = image.crop(
            (
                int(round(crop[0] * width)),
                int(round(crop[1] * height)),
                int(round(crop[2] * width)),
                int(round(crop[3] * height)),
            )
        )
    background = options.get("background", "alpha_to_white")
    if background == "alpha_to_white":
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        white.alpha_composite(image)
        image = white
    return image.convert("RGB")


def _fit_panel(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image)
    image.thumbnail(size, resampling.LANCZOS)
    cell = Image.new("RGB", size, (255, 255, 255))
    cell.paste(
        image,
        (
            (size[0] - image.size[0]) // 2,
            (size[1] - image.size[1]) // 2,
        ),
    )
    return cell


def _draw_row_label(
    canvas: Image.Image,
    bounds: Tuple[int, int, int, int],
    label: str,
    font,
    orientation: str,
) -> None:
    if orientation == "horizontal":
        _centered_text(ImageDraw.Draw(canvas), bounds, label, font)
        return
    left, top, right, bottom = bounds
    label_image = Image.new(
        "RGB",
        (bottom - top, right - left),
        (255, 255, 255),
    )
    _centered_text(
        ImageDraw.Draw(label_image),
        (0, 0, label_image.size[0], label_image.size[1]),
        label,
        font,
    )
    label_image = label_image.rotate(90, expand=True)
    canvas.paste(label_image, (left, top))


def compose_figure(
    manifest_or_path: Any,
    output_path: Optional[Path] = None,
    provenance_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compose a white-background figure and write source provenance."""

    manifest = (
        load_figure_manifest(Path(manifest_or_path))
        if not isinstance(manifest_or_path, dict)
        else manifest_or_path
    )
    report = audit_figure_manifest(manifest, check_inputs=True)
    if not report["ready"]:
        raise FigureBlockedError(report)

    layout = manifest["layout"]
    cell_width, cell_height = layout["cell_size"]
    padding = int(layout.get("padding", 24))
    gap = int(layout.get("gap", 12))
    header_height = int(layout.get("header_height", 48))
    row_label_width = int(layout.get("row_label_width", 96))
    method_count = len(manifest["methods"])
    row_count = len(manifest["rows"])
    width = (
        2 * padding
        + row_label_width
        + method_count * cell_width
        + max(0, method_count - 1) * gap
    )
    height = (
        2 * padding
        + header_height
        + row_count * cell_height
        + max(0, row_count - 1) * gap
    )
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    header_font = _font(int(layout.get("header_font_size", 22)), bold=True)
    row_font = _font(int(layout.get("row_font_size", 19)), bold=False)
    panel_left = padding + row_label_width
    for column, method in enumerate(manifest["methods"]):
        left = panel_left + column * (cell_width + gap)
        _centered_text(
            draw,
            (left, padding, left + cell_width, padding + header_height),
            method["label"],
            header_font,
        )

    orientation = layout.get("row_label_orientation", "horizontal")
    for row_index, row in enumerate(manifest["rows"]):
        top = padding + header_height + row_index * (cell_height + gap)
        _draw_row_label(
            canvas,
            (padding, top, padding + row_label_width, top + cell_height),
            row["label"],
            row_font,
            orientation,
        )
        for column, method in enumerate(manifest["methods"]):
            panel = manifest["panels"][row["id"]][method["id"]]
            if panel is None:
                continue
            image = _fit_panel(
                _open_panel(panel),
                (cell_width, cell_height),
            )
            left = panel_left + column * (cell_width + gap)
            canvas.paste(image, (left, top))

    output = Path(
        output_path if output_path is not None else manifest["output"]["image"]
    ).expanduser().resolve()
    provenance = Path(
        provenance_path
        if provenance_path is not None
        else manifest["output"]["provenance"]
    ).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".partial")
    canvas.save(temporary_output, format="PNG")
    os.replace(str(temporary_output), str(output))

    manifest_path = manifest.get("_manifest_path")
    record = {
        "schema_version": 1,
        "figure_id": manifest["figure"]["id"],
        "paper_target": manifest["figure"]["paper_target"],
        "source_policy": manifest["figure"]["source_policy"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": manifest_path,
            "sha256": (
                _sha256(Path(manifest_path))
                if manifest_path and Path(manifest_path).is_file()
                else None
            ),
        },
        "method_order": layout["method_order"],
        "row_order": layout["row_order"],
        "canvas": {
            "width": width,
            "height": height,
            "background": "white",
        },
        "sources": report["inputs"],
        "reference": manifest.get("reference"),
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
            "width": width,
            "height": height,
        },
    }
    provenance.parent.mkdir(parents=True, exist_ok=True)
    temporary_provenance = provenance.with_name(
        provenance.name + ".partial"
    )
    temporary_provenance.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary_provenance), str(provenance))
    return record


def validate_figure_artifact(
    image_path: Path,
    provenance_path: Path,
    check_sources: bool = True,
) -> Dict[str, Any]:
    """Validate a composed figure against its immutable provenance record."""

    image_path = Path(image_path).resolve()
    provenance_path = Path(provenance_path).resolve()
    record = {
        "image": str(image_path),
        "provenance": str(provenance_path),
        "ready": False,
    }
    if not image_path.is_file() or not provenance_path.is_file():
        record["missing"] = [
            str(path)
            for path in (image_path, provenance_path)
            if not path.is_file()
        ]
        return record
    try:
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        with Image.open(image_path) as image:
            image.load()
            size = list(image.size)
            corners = [
                image.convert("RGB").getpixel(point)
                for point in (
                    (0, 0),
                    (image.width - 1, 0),
                    (0, image.height - 1),
                    (image.width - 1, image.height - 1),
                )
            ]
        output = provenance.get("output", {})
        hash_matches = _sha256(image_path) == output.get("sha256")
        dimensions_match = size == [
            output.get("width"),
            output.get("height"),
        ]
        white_background = all(pixel == (255, 255, 255) for pixel in corners)
        source_checks = []
        for source in provenance.get("sources", []):
            source_path = Path(str(source.get("path", ""))).expanduser()
            exists = source_path.is_file()
            source_hash_matches = bool(
                exists
                and source.get("sha256")
                and _sha256(source_path) == source.get("sha256")
            )
            source_checks.append(
                {
                    "row": source.get("row"),
                    "method": source.get("method"),
                    "path": str(source_path),
                    "exists": exists,
                    "hash_matches": source_hash_matches,
                }
            )
        sources_match = bool(
            source_checks
            and all(item["hash_matches"] for item in source_checks)
        )
        record.update(
            readable=True,
            schema_version=provenance.get("schema_version"),
            figure_id=provenance.get("figure_id"),
            method_order=provenance.get("method_order"),
            row_order=provenance.get("row_order"),
            source_count=len(provenance.get("sources", [])),
            hash_matches=hash_matches,
            dimensions_match=dimensions_match,
            white_background=white_background,
            sources_checked=check_sources,
            sources_match=sources_match,
            source_checks=source_checks,
        )
        record["ready"] = bool(
            record["schema_version"] == 1
            and record["figure_id"]
            and isinstance(record["method_order"], list)
            and bool(record["method_order"])
            and isinstance(record["row_order"], list)
            and bool(record["row_order"])
            and hash_matches
            and dimensions_match
            and white_background
            and (not check_sources or sources_match)
        )
    except Exception as exc:
        record.update(readable=False, error=repr(exc))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("manifest", type=Path)
    audit_parser.add_argument("--report", type=Path)
    compose_parser = subparsers.add_parser("compose")
    compose_parser.add_argument("manifest", type=Path)
    compose_parser.add_argument("--output", type=Path)
    compose_parser.add_argument("--provenance", type=Path)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("image", type=Path)
    validate_parser.add_argument("provenance", type=Path)
    args = parser.parse_args()

    if args.command == "audit":
        result = audit_figure_manifest(
            load_figure_manifest(args.manifest),
            check_inputs=True,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text + "\n", encoding="utf-8")
        print(text)
        if not result["ready"]:
            raise SystemExit(2)
        return
    if args.command == "compose":
        result = compose_figure(
            args.manifest,
            output_path=args.output,
            provenance_path=args.provenance,
        )
    else:
        result = validate_figure_artifact(
            args.image,
            args.provenance,
        )
        if not result["ready"]:
            print(json.dumps(result, indent=2, sort_keys=True))
            raise SystemExit(2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
