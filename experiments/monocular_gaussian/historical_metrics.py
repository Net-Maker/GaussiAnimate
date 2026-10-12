"""Evaluate synthetic renders with the exact historical RigGS metric code path."""

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as functional
from torchvision.transforms.functional import to_tensor

from .riggs_loader import bundled_engine_root


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _files(path: Path):
    return {
        item.stem: item
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    }


def _gaussian(window_size, sigma, device):
    values = [
        math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
        for x in range(window_size)
    ]
    result = torch.tensor(values, dtype=torch.float32, device=device)
    return result / result.sum()


def _ssim(first, second, window_size=11):
    channel = first.shape[1]
    one_d = _gaussian(window_size, 1.5, first.device).unsqueeze(1)
    window = one_d.mm(one_d.t()).unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    mu_first = functional.conv2d(
        first, window, padding=window_size // 2, groups=channel
    )
    mu_second = functional.conv2d(
        second, window, padding=window_size // 2, groups=channel
    )
    mu_first_sq = mu_first.square()
    mu_second_sq = mu_second.square()
    mu_both = mu_first * mu_second
    sigma_first = (
        functional.conv2d(
            first * first, window, padding=window_size // 2, groups=channel
        )
        - mu_first_sq
    )
    sigma_second = (
        functional.conv2d(
            second * second, window, padding=window_size // 2, groups=channel
        )
        - mu_second_sq
    )
    sigma_both = (
        functional.conv2d(
            first * second, window, padding=window_size // 2, groups=channel
        )
        - mu_both
    )
    result = ((2 * mu_both + 0.01**2) * (2 * sigma_both + 0.03**2)) / (
        (mu_first_sq + mu_second_sq + 0.01**2)
        * (sigma_first + sigma_second + 0.03**2)
    )
    return result.mean()


def evaluate(
    prediction_dir: Path,
    reference_dir: Path,
    output: Path,
    metrics,
    device: str,
    riggs_root: Path = None,
):
    requested = tuple(metrics)
    unknown = set(requested) - {"psnr", "ssim", "lpips"}
    if unknown:
        raise ValueError(f"unsupported metrics: {sorted(unknown)}")
    predictions = _files(Path(prediction_dir))
    references = _files(Path(reference_dir))
    if set(predictions) != set(references):
        raise ValueError("prediction and ground-truth image stems differ")
    if not predictions:
        raise ValueError("no image pairs found")

    lpips_model = None
    if "lpips" in requested:
        sys.path.insert(0, str(Path(riggs_root or bundled_engine_root()).resolve()))
        from lpipsPyTorch.modules.lpips import LPIPS

        lpips_model = LPIPS("vgg", "0.1").to(device).eval()

    rows = []
    for stem in sorted(predictions):
        with Image.open(predictions[stem]) as image:
            prediction = to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
        with Image.open(references[stem]) as image:
            reference = to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
        if prediction.shape != reference.shape:
            raise ValueError(f"image shape mismatch for {stem}")
        row = {"frame": stem}
        with torch.no_grad():
            if "psnr" in requested:
                mse = (prediction - reference).square().reshape(1, -1).mean(1)
                row["psnr"] = float((20 * torch.log10(1.0 / torch.sqrt(mse))).item())
            if "ssim" in requested:
                row["ssim"] = float(_ssim(prediction, reference).item())
            if lpips_model is not None:
                # Historical metrics.py passes [0,1] tensors directly.
                row["lpips"] = float(lpips_model(prediction, reference).item())
        rows.append(row)

    aggregate = {
        metric: float(sum(row[metric] for row in rows) / len(rows))
        for metric in requested
    }
    result = {
        "kind": "image",
        "protocol": {
            "implementation": "RigGS metrics.py",
            "ssim": "11x11 Gaussian window, sigma 1.5, zero padding",
            "lpips": "lpipsPyTorch v0.1 VGG, historical direct [0,1] input",
        },
        "prediction_dir": str(Path(prediction_dir).resolve()),
        "reference_dir": str(Path(reference_dir).resolve()),
        "frames": len(rows),
        "device": device,
        "metrics": aggregate,
        "per_frame": rows,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metrics", nargs="+", default=["psnr", "ssim", "lpips"]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--riggs-root",
        type=Path,
        default=bundled_engine_root(),
    )
    args = parser.parse_args()
    result = evaluate(
        args.prediction_dir,
        args.reference_dir,
        args.output,
        args.metrics,
        args.device,
        args.riggs_root,
    )
    print(json.dumps(result["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
