"""Render spatial SSIM maps for matching rendered and ground-truth images."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.loss_utils import ssim_map


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# Edit these values before running: python3 visualize_ssim.py
RENDER_DIR = Path("models/bicycle/train/ours_30000/renders")
GT_DIR = Path("models/bicycle/train/ours_30000/gt")
OUTPUT_DIR = Path("models/bicycle/train/ours_30000/ssim_dissimilarity/renders")

WINDOW_SIZE = 11
MAP_MODE = "dissimilarity"  # "dissimilarity" (1 - SSIM) or "ssim"
DEVICE = "auto"  # "auto", "cuda", or "cpu"


def image_files(folder):
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError("Image folder not found: {}".format(folder))
    return {
        path.name: path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE requests CUDA, but CUDA is unavailable")
    return device


def load_rgb_tensor(path, device):
    with Image.open(str(path)) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def heatmap(values):
    """Apply the same blue-cyan-yellow-red map used by KNN visualization."""
    values = torch.clamp(values, 0.0, 1.0)
    red = torch.clamp(1.5 - torch.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = torch.clamp(1.5 - torch.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = torch.clamp(1.5 - torch.abs(4.0 * values - 1.0), 0.0, 1.0)
    return torch.stack((red, green, blue), dim=0)


def save_rgb_tensor(image, path):
    array = (
        image.detach().clamp(0.0, 1.0).mul(255).round()
        .to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    )
    Image.fromarray(array, mode="RGB").save(str(path))


def visualization_values(channel_mean_ssim, mode):
    if mode == "ssim":
        # SSIM is theoretically in [-1, 1]; negative similarity is shown as blue.
        return torch.clamp(channel_mean_ssim, 0.0, 1.0)
    if mode == "dissimilarity":
        # Match the KNN visualization semantics: red means a larger error signal.
        return torch.clamp(1.0 - channel_mean_ssim, 0.0, 1.0)
    raise ValueError("MAP_MODE must be 'ssim' or 'dissimilarity'")


def main():
    if WINDOW_SIZE <= 0 or WINDOW_SIZE % 2 == 0:
        raise ValueError("WINDOW_SIZE must be a positive odd integer")
    if MAP_MODE not in ("ssim", "dissimilarity"):
        raise ValueError("MAP_MODE must be 'ssim' or 'dissimilarity'")

    render_files = image_files(RENDER_DIR)
    gt_files = image_files(GT_DIR)
    filenames = sorted(set(render_files) & set(gt_files))
    skipped_filenames = sorted((set(render_files) | set(gt_files)) - set(filenames))
    if not filenames:
        raise ValueError("Render and GT folders have no matching image filenames")
    if skipped_filenames:
        print("Skipping {} unmatched files".format(len(skipped_filenames)))

    device = resolve_device(DEVICE)
    output_dir = OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = {}
    with torch.no_grad():
        for index, filename in enumerate(filenames, 1):
            rendered = load_rgb_tensor(render_files[filename], device)
            gt = load_rgb_tensor(gt_files[filename], device)
            if rendered.shape != gt.shape:
                raise ValueError(
                    "Image shape mismatch for {}: render {}, GT {}".format(
                        filename, tuple(rendered.shape), tuple(gt.shape)
                    )
                )
            spatial_ssim = ssim_map(rendered, gt, window_size=WINDOW_SIZE)
            scores[filename] = float(spatial_ssim.mean().item())
            channel_mean_ssim = spatial_ssim.mean(dim=1).squeeze(0)
            colored = heatmap(visualization_values(channel_mean_ssim, MAP_MODE))
            save_rgb_tensor(colored, output_dir / filename)
            print(
                "SSIM maps: {}/{}".format(index, len(filenames)),
                end="\r", flush=True,
            )
    print()

    mean_ssim = float(np.mean(list(scores.values())))
    metadata = {
        "render_dir": str(RENDER_DIR.expanduser().resolve()),
        "gt_dir": str(GT_DIR.expanduser().resolve()),
        "output_dir": str(output_dir),
        "map_mode": MAP_MODE,
        "visualized_value": "SSIM" if MAP_MODE == "ssim" else "1 - SSIM",
        "window_size": WINDOW_SIZE,
        "device": str(device),
        "colormap": "blue-cyan-yellow-red",
        "low_color": "blue",
        "high_color": "red",
        "mean_ssim": mean_ssim,
        "image_count": len(filenames),
        "skipped_filenames": skipped_filenames,
        "per_image_ssim": scores,
    }
    metadata_path = output_dir.parent / "ssim_metadata.json"
    with metadata_path.open("w") as stream:
        json.dump(metadata, stream, indent=2)
    print("Mean SSIM: {:.7f}".format(mean_ssim))
    print("SSIM visualization written to {}".format(output_dir))


if __name__ == "__main__":
    main()
