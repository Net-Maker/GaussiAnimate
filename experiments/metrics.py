"""Reproducible image and geometry metrics shared by paper experiments."""

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _write_json(path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _image_files(directory) -> Dict[str, Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")
    files = {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not files:
        raise ValueError(f"no supported images found under {directory}")
    return files


def _load_image(path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("image metrics require the 'evaluation' extra") from exc
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return array


def _load_mask(path, shape: Tuple[int, int]) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("masked image metrics require Pillow") from exc
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if mask.shape != shape:
        raise ValueError(f"mask {path} has shape {mask.shape}, expected {shape}")
    return mask > 0.5


def psnr(prediction: np.ndarray, reference: np.ndarray, mask=None) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if prediction.shape != reference.shape:
        raise ValueError("PSNR inputs must have matching shapes")
    error = (prediction - reference) ** 2
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != prediction.shape[:2] or not mask.any():
            raise ValueError("PSNR mask must select pixels with shape [H,W]")
        error = error[mask]
    mse = float(error.mean())
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def ssim(prediction: np.ndarray, reference: np.ndarray, mask=None) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError("SSIM requires scikit-image from the 'evaluation' extra") from exc
    prediction = np.asarray(prediction, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if prediction.shape != reference.shape:
        raise ValueError("SSIM inputs must have matching shapes")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        prediction = prediction.copy()
        reference = reference.copy()
        prediction[~mask] = 0.0
        reference[~mask] = 0.0
    return float(
        structural_similarity(
            reference,
            prediction,
            channel_axis=-1,
            data_range=1.0,
        )
    )


class _LPIPS:
    def __init__(self, network: str = "vgg", device: Optional[str] = None):
        try:
            import lpips
            import torch
        except ImportError as exc:
            raise ImportError("LPIPS requires torch and lpips from the 'evaluation' extra") from exc
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = lpips.LPIPS(net=network).to(self.device).eval()

    def __call__(self, prediction: np.ndarray, reference: np.ndarray, mask=None) -> float:
        prediction = np.asarray(prediction, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            prediction = prediction.copy()
            reference = reference.copy()
            prediction[~mask] = 0.0
            reference[~mask] = 0.0
        torch = self.torch
        pred = torch.from_numpy(prediction).permute(2, 0, 1)[None].to(self.device)
        ref = torch.from_numpy(reference).permute(2, 0, 1)[None].to(self.device)
        with torch.no_grad():
            value = self.model(pred * 2.0 - 1.0, ref * 2.0 - 1.0)
        return float(value.item())


def evaluate_image_directories(
    prediction_dir,
    reference_dir,
    output_path,
    metrics: Sequence[str] = ("psnr", "ssim", "lpips"),
    mask_dir=None,
    lpips_network: str = "vgg",
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Pair images by stem and report per-frame and mean rendering metrics."""

    requested = tuple(metrics)
    supported = {"psnr", "ssim", "lpips"}
    unknown = set(requested) - supported
    if unknown:
        raise ValueError(f"unsupported image metrics: {', '.join(sorted(unknown))}")
    predictions = _image_files(prediction_dir)
    references = _image_files(reference_dir)
    stems = sorted(set(predictions) & set(references))
    if not stems:
        raise ValueError("prediction and reference directories have no matching image stems")
    missing_prediction = sorted(set(references) - set(predictions))
    missing_reference = sorted(set(predictions) - set(references))
    if missing_prediction or missing_reference:
        raise ValueError(
            "image sets differ: "
            f"missing predictions={missing_prediction[:5]}, "
            f"missing references={missing_reference[:5]}"
        )
    masks = _image_files(mask_dir) if mask_dir else None
    lpips_metric = _LPIPS(lpips_network, device) if "lpips" in requested else None
    rows: List[Dict[str, object]] = []
    for stem in stems:
        prediction = _load_image(predictions[stem])
        reference = _load_image(references[stem])
        if prediction.shape != reference.shape:
            raise ValueError(
                f"image shape mismatch for {stem}: {prediction.shape} vs {reference.shape}"
            )
        mask = None
        if masks is not None:
            if stem not in masks:
                raise ValueError(f"missing mask for image {stem}")
            mask = _load_mask(masks[stem], prediction.shape[:2])
        row: Dict[str, object] = {"frame": stem}
        if "psnr" in requested:
            row["psnr"] = psnr(prediction, reference, mask)
        if "ssim" in requested:
            row["ssim"] = ssim(prediction, reference, mask)
        if lpips_metric is not None:
            row["lpips"] = lpips_metric(prediction, reference, mask)
        rows.append(row)
    aggregate = {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in requested
    }
    result = {
        "kind": "image",
        "prediction_dir": str(Path(prediction_dir).resolve()),
        "reference_dir": str(Path(reference_dir).resolve()),
        "mask_dir": str(Path(mask_dir).resolve()) if mask_dir else None,
        "frames": len(rows),
        "metrics": aggregate,
        "per_frame": rows,
    }
    _write_json(output_path, result)
    return result


def _load_points(path, key: str) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        points = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=False)
        if key not in data:
            raise ValueError(f"{path} does not contain point key {key!r}")
        points = data[key]
    else:
        raise ValueError(f"unsupported geometry metric input: {path.suffix}")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"{path} points must have shape [T,N,3]")
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains non-finite points")
    return points


def evaluate_geometry(
    prediction_path,
    reference_path,
    output_path,
    prediction_key: str = "points",
    reference_key: str = "points",
    unit_scale: float = 1.0,
) -> Dict[str, object]:
    """Evaluate registered point trajectories with paper geometry metrics."""

    prediction = _load_points(prediction_path, prediction_key)
    reference = _load_points(reference_path, reference_key)
    if prediction.shape != reference.shape:
        raise ValueError(
            f"geometry shapes differ: prediction={prediction.shape}, reference={reference.shape}"
        )
    distance = np.linalg.norm(prediction - reference, axis=-1) * float(unit_scale)
    per_frame_rmse = np.sqrt(np.mean(distance * distance, axis=1))
    rmse = float(np.sqrt(np.mean(distance * distance)))
    mean_error = float(distance.mean())
    extent = np.ptp(reference.reshape(-1, 3), axis=0) * float(unit_scale)
    diagonal = float(np.linalg.norm(extent))
    result = {
        "kind": "geometry",
        "prediction": str(Path(prediction_path).resolve()),
        "reference": str(Path(reference_path).resolve()),
        "frames": int(prediction.shape[0]),
        "points_per_frame": int(prediction.shape[1]),
        "unit_scale": float(unit_scale),
        "metrics": {
            "rmse": rmse,
            "mean_error": mean_error,
            "normalized_rmse": rmse / diagonal if diagonal > 1e-12 else None,
        },
        "per_frame": [
            {"frame": index, "rmse": float(value)}
            for index, value in enumerate(per_frame_rmse)
        ],
    }
    _write_json(output_path, result)
    return result


def collect_json_metrics(source_path, output_path, selectors: Dict[str, object]):
    """Select and optionally reduce JSON fields into the standard metric envelope.

    A selector may be a dotted path string, or ``{"path": "...", "reduce":
    "mean"}``. The ``*`` path component maps over every list item or dictionary
    value, which makes per-sequence result files directly aggregatable.
    """

    source_path = Path(source_path)
    data = json.loads(source_path.read_text(encoding="utf-8"))

    def lookup(value, components, full_path):
        if not components:
            return value
        component, remaining = components[0], components[1:]
        if component == "*":
            items = value.values() if isinstance(value, dict) else value
            if not isinstance(value, (dict, list)):
                raise KeyError(f"{source_path} has no JSON field {full_path!r}")
            return [lookup(item, remaining, full_path) for item in items]
        if isinstance(value, dict) and component in value:
            return lookup(value[component], remaining, full_path)
        if isinstance(value, list) and component.isdigit():
            index = int(component)
            if index < len(value):
                return lookup(value[index], remaining, full_path)
        raise KeyError(f"{source_path} has no JSON field {full_path!r}")

    reducers = {
        "mean": lambda values: float(np.mean(values)),
        "sum": lambda values: float(np.sum(values)),
        "min": lambda values: float(np.min(values)),
        "max": lambda values: float(np.max(values)),
        "count": lambda values: int(np.asarray(values).size),
    }
    metrics = {}
    for name, selector in selectors.items():
        if isinstance(selector, str):
            path, reduce = selector, None
        elif isinstance(selector, dict) and isinstance(selector.get("path"), str):
            path, reduce = selector["path"], selector.get("reduce")
        else:
            raise ValueError(f"metric selector {name!r} must be a path or mapping")
        value = lookup(data, path.split("."), path)
        if reduce is not None:
            if reduce not in reducers:
                raise ValueError(f"unsupported JSON metric reduction: {reduce}")
            value = reducers[reduce](value)
        metrics[name] = value
    result = {
        "kind": "imported",
        "source": str(source_path.resolve()),
        "metrics": metrics,
    }
    _write_json(output_path, result)
    return result
