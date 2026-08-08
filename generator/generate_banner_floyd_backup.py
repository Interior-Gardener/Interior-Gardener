from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import config


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "assets" / config.PORTRAIT_FILE
PROCESSED_OUTPUT = ROOT / "data" / "portrait_processed.png"
DITHER_OUTPUT = ROOT / "data" / "portrait_dither.png"


# ============================================================
# Portrait preprocessing
# ============================================================

def process_portrait():
    print(f"Loading portrait: {INPUT}")

    image = Image.open(INPUT).convert("RGB")

    print(f"Original size: {image.size}")

    # --------------------------------------------------------
    # 1. Explicit head + shoulders crop
    # --------------------------------------------------------

    crop_box = (25, 320, 455, 807)

    image = image.crop(crop_box)

    # --------------------------------------------------------
    # 2. Resize to required 300 x 340 processing grid
    # --------------------------------------------------------

    image = image.resize(
        (300, 340),
        Image.Resampling.LANCZOS,
    )

    # --------------------------------------------------------
    # 3. Autocontrast
    # --------------------------------------------------------

    image = ImageOps.autocontrast(
        image,
        cutoff=1,
    )

    # --------------------------------------------------------
    # 4. Contrast x 1.3
    # --------------------------------------------------------

    image = ImageEnhance.Contrast(image).enhance(1.3)

    # --------------------------------------------------------
    # 5. UnsharpMask
    # --------------------------------------------------------

    image = image.filter(
        ImageFilter.UnsharpMask(
            radius=3,
            percent=140,
        )
    )

    PROCESSED_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(PROCESSED_OUTPUT)

    print(f"Saved processed portrait: {PROCESSED_OUTPUT}")
    print(f"Processed size: {image.size}")

    return image


# ============================================================
# Serpentine Floyd-Steinberg 1-bit dithering
# ============================================================

def floyd_steinberg_serpentine(image):
    """
    Convert a grayscale image to 1-bit using
    serpentine Floyd-Steinberg error diffusion.

    White  -> background
    Black  -> future portrait dot
    """

    grayscale = image.convert("L")

    # Float array is essential because error diffusion
    # produces intermediate values.
    pixels = np.asarray(
        grayscale,
        dtype=np.float64,
    ).copy()

    height, width = pixels.shape

    result = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for y in range(height):

        # Alternate scan direction every row.
        if y % 2 == 0:
            x_range = range(width)
            direction = 1
        else:
            x_range = range(width - 1, -1, -1)
            direction = -1

        for x in x_range:

            old_value = pixels[y, x]

            # 1-bit quantization.
            if old_value >= 128:
                new_value = 255
            else:
                new_value = 0

            result[y, x] = new_value

            error = old_value - new_value

            # ------------------------------------------------
            # Floyd-Steinberg error diffusion
            #
            #            X   7
            #        3   5   1
            #
            # divided by 16
            #
            # The pattern is mirrored when scanning right-to-left.
            # ------------------------------------------------

            if direction == 1:

                # Right
                if x + 1 < width:
                    pixels[y, x + 1] += error * 7 / 16

                # Bottom-left
                if y + 1 < height and x - 1 >= 0:
                    pixels[y + 1, x - 1] += error * 3 / 16

                # Bottom
                if y + 1 < height:
                    pixels[y + 1, x] += error * 5 / 16

                # Bottom-right
                if y + 1 < height and x + 1 < width:
                    pixels[y + 1, x + 1] += error * 1 / 16

            else:

                # Left
                if x - 1 >= 0:
                    pixels[y, x - 1] += error * 7 / 16

                # Bottom-right
                if y + 1 < height and x + 1 < width:
                    pixels[y + 1, x + 1] += error * 3 / 16

                # Bottom
                if y + 1 < height:
                    pixels[y + 1, x] += error * 5 / 16

                # Bottom-left
                if y + 1 < height and x - 1 >= 0:
                    pixels[y + 1, x - 1] += error * 1 / 16

    return Image.fromarray(
        result,
        mode="L",
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    processed = process_portrait()

    # Convert the processed portrait into 1-bit
    # serpentine Floyd-Steinberg dither.
    dithered = floyd_steinberg_serpentine(processed)

    dithered.save(DITHER_OUTPUT)

    # --------------------------------------------------------
    # Measurements
    # --------------------------------------------------------

    pixels = np.asarray(dithered)

    black_pixels = np.count_nonzero(pixels == 0)
    total_pixels = pixels.size

    ink_coverage = black_pixels / total_pixels

    print()
    print("DITHER COMPLETE")
    print("----------------")
    print(f"Grid: {pixels.shape[1]} x {pixels.shape[0]}")
    print(f"Total pixels: {total_pixels}")
    print(f"Black pixels: {black_pixels}")
    print(f"Ink coverage: {ink_coverage:.4f}")
    print(f"Saved: {DITHER_OUTPUT}")