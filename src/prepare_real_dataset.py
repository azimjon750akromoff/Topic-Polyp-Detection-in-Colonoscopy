import os
import shutil
import random
from pathlib import Path
import numpy as np
from PIL import Image


def mask_to_bbox(mask_path):
    mask = Image.open(mask_path).convert("L")
    m = np.array(mask)

    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    w, h = mask.size
    x_center = (xmin + xmax) / 2 / w
    y_center = (ymin + ymax) / 2 / h
    width = (xmax - xmin) / w
    height = (ymax - ymin) / h

    return x_center, y_center, width, height


def prepare_dataset(source_dir, output_dir, n_shot=-1, seed=42):
    random.seed(seed)

    source = Path(source_dir)
    out = Path(output_dir)

    img_dir = source / "images"
    mask_dir = source / "masks"

    images = [
        p for p in img_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    ]

    print(f"Found {len(images)} images total.")

    if n_shot > 0:
        images = random.sample(images, n_shot)
        print(f"Using only {n_shot} images (few-shot mode).")

    random.shuffle(images)
    split = int(0.8 * len(images))
    train_imgs = images[:split]
    val_imgs = images[split:]

    for s in ["train", "val"]:
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)

    def process(split, imgs):
        for img_path in imgs:
            mask_path = mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                mask_path = mask_dir / (img_path.stem + ".jpg")
            if not mask_path.exists():
                print(f"⚠ No mask found for {img_path.stem}, skipping...")
                continue

            shutil.copy2(img_path, out / split / "images" / img_path.name)
            bbox = mask_to_bbox(mask_path)

            label_path = out / split / "labels" / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                if bbox:
                    x, y, w, h = bbox
                    f.write(f"0 {x} {y} {w} {h}\n")

    process("train", train_imgs)
    process("val", val_imgs)

    with open(out / "dataset.yaml", "w") as f:
        f.write(f"""
path: {out.absolute()}
train: train/images
val: val/images
nc: 1
names: ['polyp']
""")

    print("Dataset created at:", out)


if __name__ == "__main__":
    prepare_dataset("data/real_dataset", "data/full_real", n_shot=-1)
