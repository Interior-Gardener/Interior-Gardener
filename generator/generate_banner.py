from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "assets" / config.PORTRAIT_FILE

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_OUTPUT = DATA_DIR / "portrait_processed.png"
MASK_OUTPUT = DATA_DIR / "portrait_mask.png"
DITHER_OUTPUT = DATA_DIR / "portrait_dither.png"
SVG_OUTPUT = DATA_DIR / "portrait_dither.svg"


# ============================================================
# SETTINGS
# ============================================================

# Final processing resolution.
# Keeping this moderate prevents an unnecessarily huge SVG.
OUTPUT_SIZE = (420, 520)

# Dot size in the final SVG.
DOT_RADIUS = 1.25

# Spacing between dots.
DOT_SPACING = 3

# Dithering threshold.
DITHER_THRESHOLD = 128

# How strongly the original portrait is enhanced before dithering.
CONTRAST = 1.15

# Slightly brighten dark facial regions.
GAMMA = 0.92


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image():
    print(f"Loading portrait: {INPUT}")

    if not INPUT.exists():
        raise FileNotFoundError(
            f"\nPortrait not found:\n{INPUT}\n\n"
            f"Check config.py and make sure PORTRAIT_FILE is correct."
        )

    image = cv2.imread(str(INPUT), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Could not read image: {INPUT}")

    print(f"Original size: {image.shape[1]} x {image.shape[0]}")

    return image


# ============================================================
# CROP
# ============================================================

def crop_portrait(image):
    """
    Remove only the large empty space above the head.

    We deliberately keep the shoulders and lower body area.
    The source image already has a good composition, so we
    don't perform an aggressive face crop.
    """

    height, width = image.shape[:2]

    # Based on the supplied 1696 x 2082 portrait.
    #
    # The subject begins around y ~= 59, but the very top
    # contains useful hair separation. Therefore we only
    # remove a small amount of empty space.
    top = max(0, int(height * 0.015))

    # Keep the bottom intact.
    bottom = height

    cropped = image[top:bottom, :]

    print(
        f"Crop: x=0:{width}, "
        f"y={top}:{bottom}"
    )

    return cropped


def create_subject_mask(image):
    """
    Use MediaPipe Selfie Segmentation for a robust human silhouette.
    This effectively eliminates the rectangular background block caused by GrabCut,
    as MediaPipe is specifically trained to recognize the human body, hair, and face.
    """
    import mediapipe as mp

    print("Running MediaPipe subject segmentation...")

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(image_rgb)
        
        # results.segmentation_mask is float32 [0.0, 1.0].
        # We threshold it at a slightly conservative value like 0.4 to keep hair details.
        subject_mask = (results.segmentation_mask > 0.4).astype(np.uint8) * 255
        
    # ========================================================
    # CLEAN MASK
    # ========================================================

    # Slight edge smoothing to remove jaggedness
    subject_mask = cv2.GaussianBlur(
        subject_mask,
        (5, 5),
        0,
    )

    return subject_mask



# ============================================================
# PREPARE PORTRAIT
# ============================================================

def prepare_portrait(image, mask):
    """
    Resize image + mask while preserving the portrait's aspect ratio.

    The background is removed before grayscale processing.
    """

    target_width, target_height = OUTPUT_SIZE

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(
        image_rgb,
        (target_width, target_height),
        interpolation=cv2.INTER_LANCZOS4,
    )

    resized_mask = cv2.resize(
        mask,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    # Convert to grayscale.
    gray = cv2.cvtColor(
        resized_image,
        cv2.COLOR_RGB2GRAY,
    )

    # ========================================================
    # CONTRAST
    # ========================================================

    gray_float = gray.astype(np.float32) / 255.0

    # Gamma adjustment.
    gray_float = np.power(
        gray_float,
        GAMMA,
    )

    # Contrast around midpoint.
    gray_float = (
        (gray_float - 0.5) * CONTRAST
    ) + 0.5

    gray_float = np.clip(
        gray_float,
        0.0,
        1.0,
    )

    gray = (
        gray_float * 255.0
    ).astype(np.uint8)

    # ========================================================
    # APPLY MASK
    # ========================================================

    # Outside the subject becomes white.
    #
    # This is intentional because the final dither only
    # generates dots for the actual subject.
    gray_masked = np.full_like(
        gray,
        255,
    )

    foreground = resized_mask > 20

    gray_masked[foreground] = gray[foreground]

    return gray_masked, resized_mask


# ============================================================
# FLOYD-STEINBERG DITHER
# ============================================================

def floyd_steinberg_dither(gray):
    """
    Floyd-Steinberg dithering.

    Dark pixels become black dots.
    Light pixels remain mostly empty.

    This preserves facial features much better than
    simple thresholding.
    """

    img = gray.astype(np.float32).copy()

    height, width = img.shape

    for y in range(height):
        for x in range(width):

            old_pixel = img[y, x]

            new_pixel = (
                0.0
                if old_pixel < DITHER_THRESHOLD
                else 255.0
            )

            img[y, x] = new_pixel

            error = old_pixel - new_pixel

            if x + 1 < width:
                img[y, x + 1] += error * 7 / 16

            if y + 1 < height and x > 0:
                img[y + 1, x - 1] += error * 3 / 16

            if y + 1 < height:
                img[y + 1, x] += error * 5 / 16

            if (
                y + 1 < height
                and x + 1 < width
            ):
                img[y + 1, x + 1] += error * 1 / 16

    return np.where(
        img < 128,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# SAVE DITHER PNG
# ============================================================

def save_dither(gray_dither):
    image = Image.fromarray(
        gray_dither,
        mode="L",
    )

    image.save(
        DITHER_OUTPUT,
        optimize=True,
    )

    print(
        f"Saved dithered portrait: "
        f"{DITHER_OUTPUT}"
    )


# ============================================================
# CREATE SVG DOT PORTRAIT
# ============================================================

def create_svg(gray_dither, mask):
    """
    Convert the dithered image into an SVG made from circles.

    Only black dither pixels inside the subject mask
    become SVG dots.
    """

    height, width = gray_dither.shape

    circles = []

    for y in range(
        0,
        height,
        DOT_SPACING,
    ):
        for x in range(
            0,
            width,
            DOT_SPACING,
        ):

            # Outside subject -> no dot.
            if mask[y, x] < 80:
                continue

            # White pixel -> no dot.
            if gray_dither[y, x] >= 128:
                continue

            # Slightly vary dot radius based on darkness.
            #
            # This gives the final SVG a richer visual texture.
            local_value = gray_dither[y, x]

            radius = DOT_RADIUS

            if local_value < 40:
                radius *= 1.15

            circles.append(
                f'<circle cx="{x}" cy="{y}" '
                f'r="{radius:.2f}" />'
            )

    svg = f'''<svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 {width} {height}"
    width="{width}"
    height="{height}"
>
    <rect
        width="100%"
        height="100%"
        fill="transparent"
    />

    <g
        fill="currentColor"
        shape-rendering="geometricPrecision"
    >
        {"".join(circles)}
    </g>
</svg>
'''

    SVG_OUTPUT.write_text(
        svg,
        encoding="utf-8",
    )

    print(
        f"Saved SVG portrait: "
        f"{SVG_OUTPUT}"
    )

    print(
        f"SVG dots generated: "
        f"{len(circles):,}"
    )


# ============================================================
# SAVE MASK
# ============================================================

def save_mask(mask):
    Image.fromarray(
        mask,
        mode="L",
    ).save(
        MASK_OUTPUT,
        optimize=True,
    )

    print(
        f"Saved subject mask: "
        f"{MASK_OUTPUT}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_portrait():

    print()
    print("=" * 60)
    print("PORTRAIT PROCESSOR")
    print("=" * 60)
    print()

    # 1. Load.
    image = load_image()

    # 2. Crop only excess top space.
    image = crop_portrait(image)

    # 3. Separate subject from dark background.
    mask = create_subject_mask(image)

    # 4. Resize + grayscale + tonal processing.
    gray, resized_mask = prepare_portrait(
        image,
        mask,
    )

    # 5. Save mask.
    save_mask(resized_mask)

    # 6. Dither.
    print("Applying Floyd-Steinberg dithering...")

    dither = floyd_steinberg_dither(
        gray,
    )

    # 7. Save processed portrait.
    Image.fromarray(
        gray,
        mode="L",
    ).save(
        PROCESSED_OUTPUT,
        optimize=True,
    )

    print(
        f"Saved processed portrait: "
        f"{PROCESSED_OUTPUT}"
    )

    # 8. Save dither PNG.
    save_dither(dither)

    # 9. Generate SVG.
    create_svg(
        dither,
        resized_mask,
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print()


if __name__ == "__main__":
    process_portrait()