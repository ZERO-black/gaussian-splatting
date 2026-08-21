"""Combine matching render images from multiple analysis folders."""

import argparse
import json
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine matching images from render folders into a labelled grid."
    )
    parser.add_argument(
        "folders", nargs="+", type=Path,
        help="Input render folders in row-major display order.",
    )
    parser.add_argument("-o", "--output-dir", required=True, type=Path)
    parser.add_argument(
        "--labels", nargs="+",
        help="Labels in input-folder order; defaults to each folder's parent name.",
    )
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--cols", required=True, type=int)
    parser.add_argument("--gap", default=0, type=int)
    parser.add_argument("--background", nargs=3, default=(0, 0, 0), type=int)
    parser.add_argument("--label-color", nargs=3, default=(255, 0, 0), type=int)
    parser.add_argument("--label-position", nargs=2, default=(12, 12), type=int)
    parser.add_argument("--label-font-size", default=32, type=int)
    parser.add_argument("--label-stroke-width", default=1, type=int)
    parser.add_argument("--label-stroke-color", nargs=3, default=(0, 0, 0), type=int)
    parser.add_argument(
        "--label-font-path", type=Path,
        help="Optional .ttf/.otf font path, required for unsupported characters.",
    )
    return parser.parse_args()


def image_files(folder):
    if not folder.is_dir():
        raise FileNotFoundError("Render folder not found: {}".format(folder))
    return {
        path.name: path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def load_label_font(image_font, font_size, font_path):
    if font_path is not None:
        return image_font.truetype(str(font_path), font_size)
    try:
        return image_font.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        return image_font.load_default()


def combine_images(
    paths, labels, output_path, rows, cols, gap, background, label_font,
    label_position, label_color, label_stroke_width, label_stroke_color,
):
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to combine render folders.") from exc

    images = []
    try:
        for path, label in zip(paths, labels):
            image = Image.open(str(path)).convert("RGB")
            ImageDraw.Draw(image).text(
                label_position, label, fill=label_color, font=label_font,
                stroke_width=label_stroke_width, stroke_fill=label_stroke_color,
            )
            images.append(image)
        cell_width = max(image.width for image in images)
        cell_height = max(image.height for image in images)
        width = cell_width * cols + gap * (cols - 1)
        height = cell_height * rows + gap * (rows - 1)
        canvas = Image.new("RGB", (width, height), tuple(background))
        for index, image in enumerate(images):
            row, column = divmod(index, cols)
            x = column * (cell_width + gap) + (cell_width - image.width) // 2
            y = row * (cell_height + gap) + (cell_height - image.height) // 2
            canvas.paste(image, (x, y))
        canvas.save(str(output_path))
    finally:
        for image in images:
            image.close()


def main():
    args = parse_args()
    try:
        from PIL import ImageFont
    except ImportError as exc:
        raise RuntimeError("Pillow is required to combine render folders.") from exc

    if len(args.folders) < 2:
        raise ValueError("Provide at least two input folders")
    labels = args.labels
    if labels is None:
        labels = [
            folder.parent.name if folder.name == "renders" else folder.name
            for folder in args.folders
        ]
    if len(labels) != len(args.folders):
        raise ValueError("--labels and input folders must have the same length")
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("--rows and --cols must be positive")
    if args.rows * args.cols < len(args.folders):
        raise ValueError("--rows * --cols must be at least the number of folders")
    if args.gap < 0:
        raise ValueError("--gap must be non-negative")
    color_channels = args.background + args.label_color + args.label_stroke_color
    if any(channel < 0 or channel > 255 for channel in color_channels):
        raise ValueError("BACKGROUND and LABEL_COLOR channels must be in [0, 255]")
    if args.label_font_size <= 0 or args.label_stroke_width < 0:
        raise ValueError("Label font size must be positive and stroke width non-negative")

    folders = [folder.expanduser().resolve() for folder in args.folders]
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir in folders:
        raise ValueError("Output directory must differ from every input folder")
    file_maps = [image_files(folder) for folder in folders]
    filename_sets = [set(files) for files in file_maps]
    if not all(filename_sets):
        empty = [str(folder) for folder, files in zip(folders, file_maps) if not files]
        raise ValueError("No images found in: {}".format(empty))
    filenames = sorted(set.intersection(*filename_sets))
    skipped_filenames = sorted(set.union(*filename_sets) - set(filenames))
    if not filenames:
        raise ValueError("Input folders have no matching image filenames")
    if skipped_filenames:
        print(
            "Skipping {} files missing from at least one folder".format(
                len(skipped_filenames)
            )
        )

    font_path = (
        None if args.label_font_path is None
        else args.label_font_path.expanduser().resolve()
    )
    label_font = load_label_font(ImageFont, args.label_font_size, font_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, filename in enumerate(filenames, 1):
        combine_images(
            [files[filename] for files in file_maps], labels, output_dir / filename,
            args.rows, args.cols, args.gap, args.background, label_font,
            args.label_position, args.label_color, args.label_stroke_width,
            args.label_stroke_color,
        )
        print("Combining: {}/{}".format(index, len(filenames)), end="\r", flush=True)
    print()
    manifest = {
        "input_folders": [str(folder) for folder in folders],
        "labels": labels,
        "rows": args.rows,
        "cols": args.cols,
        "gap": args.gap,
        "background": list(args.background),
        "label_color": list(args.label_color),
        "label_position": list(args.label_position),
        "label_font_size": args.label_font_size,
        "image_count": len(filenames),
        "skipped_file_count": len(skipped_filenames),
        "skipped_filenames": skipped_filenames,
    }
    with (output_dir / "combination.json").open("w") as stream:
        json.dump(manifest, stream, indent=2)
    print("Combined images written to {}".format(output_dir))


if __name__ == "__main__":
    main()
