from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import config


# Repository root
ROOT = Path(__file__).resolve().parent.parent

# Input/output paths
INPUT = ROOT / "assets" / config.PORTRAIT_FILE
OUTPUT = ROOT / "data" / "portrait_processed.png"


def process_portrait():
    print(f"Loading portrait: {INPUT}")

    image = Image.open(INPUT).convert("RGB")

    print(f"Original size: {image.size}")

    # --------------------------------------------
    # 1. Explicit head + shoulders crop
    # --------------------------------------------
    #
    # Coordinates are based on the supplied
    # 505 x 1111 portrait.
    #
    # This avoids the distracting building on the
    # right and removes excessive sky while keeping
    # the head and both shoulders.
    # --------------------------------------------

    crop_box = (25, 320, 455, 807)

    image = image.crop(crop_box)

    # --------------------------------------------
    # 2. Resize to required 300 x 340 grid
    # --------------------------------------------

    image = image.resize(
        (300, 340),
        Image.Resampling.LANCZOS,
    )

    # --------------------------------------------
    # 3. Autocontrast
    # --------------------------------------------

    image = ImageOps.autocontrast(
        image,
        cutoff=1,
    )

    # --------------------------------------------
    # 4. Contrast x 1.3
    # --------------------------------------------

    image = ImageEnhance.Contrast(image).enhance(1.3)

    # --------------------------------------------
    # 5. UnsharpMask
    # --------------------------------------------

    image = image.filter(
        ImageFilter.UnsharpMask(
            radius=3,
            percent=140,
        )
    )

    # --------------------------------------------
    # 6. Save intermediate result
    # --------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(OUTPUT)

    print(f"Saved processed portrait: {OUTPUT}")
    print(f"Processed size: {image.size}")


if __name__ == "__main__":
    process_portrait()