from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import random
import config
import animation_engine


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

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BANNER_OUTPUT = OUTPUT_DIR / "banner.svg"



# ============================================================
# SETTINGS
# ============================================================

# Final processing resolution.
# Keeping this moderate prevents an unnecessarily huge SVG.
OUTPUT_SIZE = (420, 520)

# Dot size in the final SVG.
DOT_RADIUS = 0.9

# Spacing between dots.
DOT_SPACING = 2

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
# CREATE SVG DOT PORTRAIT & BANNER
# ============================================================

def generate_dots(gray_dither, mask):
    height, width = gray_dither.shape
    pts = []
    
    for y in range(0, height, DOT_SPACING):
        for x in range(0, width, DOT_SPACING):
            if mask[y, x] < 80:
                continue
            if gray_dither[y, x] >= 128:
                continue
            pts.append([x, y])
            
    return np.array(pts, dtype=np.float32), width, height

def create_svg(pts, width, height):
    circles = []
    for x, y in pts:
        circles.append(f'<circle cx="{x}" cy="{y}" r="{DOT_RADIUS:.2f}" />')

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
    SVG_OUTPUT.write_text(svg, encoding="utf-8")
    print(f"Saved SVG portrait: {SVG_OUTPUT}")
    print(f"SVG dots generated: {len(pts):,}")


def compose_banner(portrait_pts, portrait_width, portrait_height):
    """
    Compose the final premium portfolio hero banner.
    """
    print("Generating final animated banner SVG...")
    
    w = config.BANNER_WIDTH
    h = config.BANNER_HEIGHT
    
    # Portrait placement: align to the right and bottom
    portrait_x = w - portrait_width - 80
    portrait_y = h - portrait_height

    # Calculate optimal transport and bands
    logo_paths = [
        ROOT / "assets" / "react-logo.webp",
        ROOT / "assets" / "node-logo.webp",
        ROOT / "assets" / "mongodb-logo.webp"
    ]
    
    # Travellers
    routes = animation_engine.compute_traveller_routes(portrait_pts, logo_paths, portrait_width, portrait_height, 900)
    
    # Drift bands
    # Remove traveller dots from portrait points to avoid overlap? The PDF suggests they are independent.
    logo_center = (portrait_width / 2, portrait_height / 2)
    bands = animation_engine.compute_drift_bands(portrait_pts, logo_center, 94)
    
    # Intro layer: random groups
    intro_groups = []
    shuffled_pts = portrait_pts.copy()
    np.random.shuffle(shuffled_pts)
    group_size = len(shuffled_pts) // 60
    for i in range(60):
        intro_groups.append(shuffled_pts[i*group_size:(i+1)*group_size])

    # Let's generate the SVG contents!
    
    # 1. Intro Layer
    intro_svg = ""
    for i, grp in enumerate(intro_groups):
        delay = random.uniform(0, 2.0)
        dots = "".join([f'<circle cx="{x}" cy="{y}" r="{DOT_RADIUS:.2f}"/>' for x, y in grp])
        intro_svg += f'<g opacity="0"><animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.5;0.99;1" dur="3.2s" begin="{delay}s" fill="freeze" />{dots}</g>'
        
    # 2. Loop Layer (Drift Bands)
    # Loop duration: 13.9s, begin: 3.2s
    # Keytimes: 0.000;0.194;0.288;0.432;0.525;0.669;0.763;0.906;1.000
    loop_svg = ""
    for b in bands:
        dx, dy = b['dx'], b['dy']
        dots = "".join([f'<circle cx="{x}" cy="{y}" r="{DOT_RADIUS:.2f}"/>' for x, y in b['pts']])
        loop_svg += f'''
        <g display="none">
            <set attributeName="display" to="inline" begin="3.2s" />
            <animate attributeName="opacity" values="1;1;0;0;0;0;0;0;1" keyTimes="0.000;0.194;0.288;0.432;0.525;0.669;0.763;0.906;1.000" dur="13.9s" begin="3.2s" repeatCount="indefinite"/>
            <animateTransform attributeName="transform" type="translate" values="0 0;0 0;{dx:.1f} {dy:.1f};{dx:.1f} {dy:.1f};{dx:.1f} {dy:.1f};{dx:.1f} {dy:.1f};{dx:.1f} {dy:.1f};{dx:.1f} {dy:.1f};0 0" keyTimes="0.000;0.194;0.288;0.432;0.525;0.669;0.763;0.906;1.000" dur="13.9s" begin="3.2s" repeatCount="indefinite"/>
            {dots}
        </g>'''
        
    # 3. Traveller Layer
    traveller_svg = ""
    if routes:
        P0, L1, L2, L3 = routes
        for i in range(len(P0)):
            p0x, p0y = P0[i]
            l1x, l1y = L1[i]
            l2x, l2y = L2[i]
            l3x, l3y = L3[i]
            
            traveller_svg += f'''
            <use href="#traveller" x="0" y="0" opacity="0">
                <animate attributeName="opacity" values="0;0;1;1;1;1;1;1;0" keyTimes="0.000;0.194;0.288;0.432;0.525;0.669;0.763;0.906;1.000" dur="13.9s" begin="3.2s" repeatCount="indefinite"/>
                <animateTransform attributeName="transform" type="translate" values="{p0x:.1f} {p0y:.1f};{p0x:.1f} {p0y:.1f};{l1x:.1f} {l1y:.1f};{l1x:.1f} {l1y:.1f};{l2x:.1f} {l2y:.1f};{l2x:.1f} {l2y:.1f};{l3x:.1f} {l3y:.1f};{l3x:.1f} {l3y:.1f};{p0x:.1f} {p0y:.1f}" keyTimes="0.000;0.194;0.288;0.432;0.525;0.669;0.763;0.906;1.000" dur="13.9s" begin="3.2s" repeatCount="indefinite"/>
            </use>'''

    # We will build the SVG contents step-by-step for a technical, modern look.
    
    # Text helper function for right-side dotted lines
    def format_row(label, value, y_offset, color=config.UI_DARK, val_color="#ffffff"):
        # We need a fixed width for the label so the dots line up.
        # But SVG text doesn't auto-wrap or fill dots. We'll use monospace for the label.
        dots = "." * max(2, (18 - len(label)))
        return f'''
        <text x="500" y="{y_offset}" class="mono label">{label} {dots}</text>
        <text x="680" y="{y_offset}" class="mono value" fill="{val_color}">{value}</text>
        '''

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
    <defs>
        <linearGradient id="portraitGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="{config.PORTRAIT_LIGHT}" />
            <stop offset="100%" stop-color="{config.PORTRAIT_DARK}" />
        </linearGradient>
        
        <circle id="traveller" cx="0" cy="0" r="{DOT_RADIUS * 1.5:.2f}" fill="{config.PORTRAIT_LIGHT}" />
        
        <style>
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&amp;display=swap');
            text {{
                font-family: 'JetBrains Mono', monospace;
            }}
            .mono {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 14px;
            }}
            .label {{
                fill: {config.UI_LIGHT};
            }}
            .value {{
                font-weight: 700;
            }}
            .title {{
                font-size: 12px;
                fill: #4B5563;
                letter-spacing: 2px;
            }}
            .top-bar-text {{
                font-size: 12px;
                fill: #6B7280;
            }}
            .accent-bg {{
                fill: {config.PORTRAIT_LIGHT};
            }}
            .accent-text {{
                fill: #ffffff;
                font-weight: 700;
                font-size: 12px;
            }}
        </style>
    </defs>

    <!-- Background Frame -->
    <rect x="0" y="0" width="{w}" height="{h}" fill="#050914" />
    <rect x="2" y="2" width="{w-4}" height="{h-4}" rx="8" fill="#060b13" stroke="{config.UI_DARK}" stroke-width="1.5" />

    <!-- Top Bar macOS dots -->
    <circle cx="24" cy="24" r="6" fill="#FF5F56" />
    <circle cx="44" cy="24" r="6" fill="#FFBD2E" />
    <circle cx="64" cy="24" r="6" fill="#27C93F" />

    <!-- Top Bar Text -->
    <text x="{w/2}" y="28" class="top-bar-text" text-anchor="middle">{config.EMAIL} - % ./profile.sh --live</text>

    <!-- Visual Map (Left side) -->
    <text x="40" y="80" class="title">VISUAL.MAP</text>
    
    <!-- Framing brackets for portrait -->
    <path d="M 40 100 L 40 90 L 50 90" fill="none" stroke="{config.UI_LIGHT}" stroke-width="2" />
    <path d="M {40 + portrait_width} 90 L {40 + portrait_width + 10} 90 L {40 + portrait_width + 10} 100" fill="none" stroke="{config.UI_LIGHT}" stroke-width="2" />
    <path d="M 40 {90 + portrait_height} L 40 {90 + portrait_height + 10} L 50 {90 + portrait_height + 10}" fill="none" stroke="{config.UI_LIGHT}" stroke-width="2" />
    <path d="M {40 + portrait_width} {90 + portrait_height + 10} L {40 + portrait_width + 10} {90 + portrait_height + 10} L {40 + portrait_width + 10} {90 + portrait_height}" fill="none" stroke="{config.UI_LIGHT}" stroke-width="2" />

    <rect x="35" y="85" width="{portrait_width + 15}" height="{portrait_height + 15}" fill="none" stroke="{config.UI_DARK}" stroke-width="1" opacity="0.3" />

    <!-- System Info (Right side) -->
    <text x="500" y="100" class="title">SYSTEM.INFO</text>
    <text x="{w - 40}" y="100" class="title" text-anchor="end" fill="#EF4444">• LIVE</text>
    
    <!-- Highlighted Email block -->
    <rect x="500" y="115" width="280" height="24" rx="4" class="accent-bg" />
    <text x="510" y="132" class="accent-text">{config.EMAIL}</text>

    <!-- Data Rows -->
    {format_row("Subject", config.NAME, 170, val_color="#ffffff")}
    {format_row("Role", config.ROLE, 195, val_color="#CBD5E1")}
    {format_row("Origin", config.LOCATION, 220, val_color="#CBD5E1")}
    {format_row("Education", config.EDUCATION, 245, val_color="#CBD5E1")}
    {format_row("Status", "Building + Learning + Shipping", 270, val_color="{config.UI_LIGHT}")}
    {format_row("ToolChain", config.TOOLCHAIN, 295, val_color="#CBD5E1")}
    
    <text x="500" y="335" class="title" style="fill: {config.ACCENT}">-----------</text>
    
    {format_row("Core.Lang", config.LANGUAGES, 360, val_color="#ffffff")}
    {format_row("Core.Frontend", config.FRONTEND, 385, val_color="#CBD5E1")}
    {format_row("Core.Backend", config.BACKEND, 410, val_color="#CBD5E1")}
    {format_row("Core.Database", config.DATABASE, 435, val_color="#CBD5E1")}
    {format_row("Core.Infra", config.INFRA, 460, val_color="#CBD5E1")}
    
    <text x="500" y="500" class="mono" fill="#94A3B8">- Contact -</text>
    
    {format_row("Grid.Mail", config.EMAIL, 530, val_color="#ffffff")}
    {format_row("Grid.Portfolio", config.PORTFOLIO.replace("https://", ""), 555, val_color="#CBD5E1")}
    {format_row("Grid.LinkedIn", config.LINKEDIN.split("in/")[1].strip("/"), 580, val_color="#CBD5E1")}

    <text x="500" y="615" class="title" style="fill: #6B7280">▶ More about me &amp; projects below in README ↓ <rect x="0" y="-10" width="8" height="12" fill="{config.ACCENT}"><animate attributeName="opacity" values="1;0;1" dur="1s" repeatCount="indefinite"/></rect></text>

    <!-- Portrait Animation Layer -->
    <g transform="translate(42, 92)" fill="url(#portraitGrad)" shape-rendering="crispEdges">
        <!-- Intro fade-in -->
        {intro_svg}
        
        <!-- Continuous Loop Drift -->
        {loop_svg}
        
        <!-- morphing travellers -->
        {traveller_svg}
    </g>
    
</svg>
'''

    BANNER_OUTPUT.write_text(svg, encoding="utf-8")
    print(f"Saved Final Banner SVG: {BANNER_OUTPUT}")


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

    # 9. Generate SVG portrait.
    pts, p_width, p_height = generate_dots(dither, resized_mask)
    create_svg(pts, p_width, p_height)

    # 10. Compose Final Animated Banner
    compose_banner(pts, p_width, p_height)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print()


if __name__ == "__main__":
    process_portrait()