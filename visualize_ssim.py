"""Render spatial SSIM maps for matching rendered and ground-truth images."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.loss_utils import ssim_map


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render spatial SSIM maps for matching render and GT images."
    )
    parser.add_argument("render_dir", type=Path, help="Rendered image directory.")
    parser.add_argument("gt_dir", type=Path, help="Ground-truth image directory.")
    parser.add_argument(
        "-o", "--output-dir", required=True, type=Path,
        help="Directory in which colored SSIM maps are written.",
    )
    parser.add_argument(
        "--window-size", default=11, type=int,
        help="Positive odd SSIM window size (default: 11).",
    )
    parser.add_argument(
        "--map-mode", choices=("dissimilarity", "ssim"), default="dissimilarity",
        help="Visualize 1 - SSIM or SSIM itself (default: dissimilarity).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Torch device: auto, cpu, cuda, or cuda:<index> (default: auto).",
    )
    return parser.parse_args()


def validate_args(args):
    if args.window_size <= 0 or args.window_size % 2 == 0:
        raise ValueError("--window-size must be a positive odd integer")


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
    raise ValueError("--map-mode must be 'ssim' or 'dissimilarity'")


def main():
    args = parse_args()
    validate_args(args)

    render_files = image_files(args.render_dir)
    gt_files = image_files(args.gt_dir)
    filenames = sorted(set(render_files) & set(gt_files))
    skipped_filenames = sorted((set(render_files) | set(gt_files)) - set(filenames))
    if not filenames:
        raise ValueError("Render and GT folders have no matching image filenames")
    if skipped_filenames:
        print("Skipping {} unmatched files".format(len(skipped_filenames)))

    device = resolve_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
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
            spatial_ssim = ssim_map(rendered, gt, window_size=args.window_size)
            scores[filename] = float(spatial_ssim.mean().item())
            channel_mean_ssim = spatial_ssim.mean(dim=1).squeeze(0)
            colored = heatmap(visualization_values(channel_mean_ssim, args.map_mode))
            save_rgb_tensor(colored, output_dir / filename)
            print(
                "SSIM maps: {}/{}".format(index, len(filenames)),
                end="\r", flush=True,
            )
    print()

    mean_ssim = float(np.mean(list(scores.values())))
    metadata = {
        "render_dir": str(args.render_dir.expanduser().resolve()),
        "gt_dir": str(args.gt_dir.expanduser().resolve()),
        "output_dir": str(output_dir),
        "map_mode": args.map_mode,
        "visualized_value": "SSIM" if args.map_mode == "ssim" else "1 - SSIM",
        "window_size": args.window_size,
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
