# Hero Animation Architecture & Pipeline

## Overview
This hero animation uses a sophisticated **Pre-Calculated Flow-Field SVG Path Morphing** technique to animate 1,500 particles between a portrait and multiple technology logos (React, Node.js, MongoDB, Docker, AWS). 

Instead of relying on heavy runtime JavaScript/Canvas (which GitHub README blocks) or massive Animated WebP/GIF files (which are 15MB+ and slow to load), this architecture pre-computes an entire fluid dynamics simulation in Python during build-time and bakes it into a single highly optimized SVG SMIL payload.

## Architecture Pipeline

1. **Logo Geometry Extraction (`svglib` / `resvg-js`)**
   - We parse the exact SVG logos from `assets/` and mathematically rasterize them using `@resvg/resvg-js`. This guarantees that internal geometries (like the MongoDB leaf and Docker whale) are perfectly preserved.
   - We extract an alpha mask and compute a Euclidean Distance Transform using OpenCV to sample points precisely within the boundaries of the logos, prioritizing denser geometry.

2. **Organic Particle Correspondence**
   - Standard randomization creates chaotic "swarms" of crossing particles.
   - We map the 1,500 points from one state to the next by sorting them into spatial bands and using the Hungarian matching algorithm (`scipy.optimize.linear_sum_assignment`) on local clusters. This ensures particles flow to their nearest structural neighbor without exploding across the screen.

3. **Fluid Dynamics via Curl Noise**
   - SMIL path morphing natively interpolates in a straight line, which looks mechanical.
   - To achieve natural motion, we calculate 5 intermediate keyframes for every transition. We apply a 2D Simplex Noise curl field (fluid simulation) to displace the particles sideways. The browser then natively interpolates across these swirling frames, resulting in an organic, smoke-like motion.

4. **DOM Optimization & Color System**
   - The original architecture created 4,500 DOM elements (1,500 circles + animations), which crashes weak browsers.
   - We group the particles into exactly 5 master `<path>` elements based on their color. The `d` attribute of these 5 paths holds all 1,500 coordinates. This guarantees 60 FPS.
   - We tint the 5 paths smoothly during the animation to match the brand colors of the active logo using `<animate attributeName="fill">`.

## Configuration & Auto-Discovery

The entire animation timeline, particle settings, and paths are controlled by a single file: `generator/config.py`.

### Adding a new logo:
1. Drop a valid SVG into `assets/logos/`, named `NN-slug.svg` (the engine processes them in filename order).
2. The engine will **automatically discover** it.
3. Add `"NN-slug"` to `LOGO_BRAND_COLORS` and `LOGO_LABELS` in `generator/config.py`.

### The one hard rule: the alpha mask *is* the logo

Particle targets are sampled from the SVG's rendered **alpha channel**, not from its
colors. Any logo drawn on an opaque background rectangle therefore rasterises to a
solid block, and the particles form a featureless square instead of the mark.

> This is exactly what happened to the original JavaScript badge: it is a full-bleed
> `<rect fill="#f7df1e">` with the letters knocked out in black, so it rendered at
> 100% alpha coverage. `assets/logos/01-javascript.svg` now draws the badge as a
> stroked frame plus filled letterforms, which leaves the background transparent.

Before adding a logo, check its coverage — anything at or near `1.000` is a solid block:

```bash
node generator/svg2png.js assets/logos/NN-slug.svg /tmp/check.png
python -c "import numpy as np;from PIL import Image;print((np.array(Image.open('/tmp/check.png').convert('RGBA'))[:,:,3]>127).mean())"
```

Healthy marks in this set land between `0.08` and `0.42`.

### Colors

`LOGO_BRAND_COLORS` in `generator/config.py` is the source of truth. Auto-extraction is
only a fallback, and it picks the most *frequent* hex in the file — which is often the
wrong one (MongoDB's grey wordmark outvotes its green leaf; AWS's near-black `#252F3E`
outvotes its orange). Every tone is finally lifted above `MIN_PARTICLE_LUMA` so no
bucket of particles disappears into the near-black background.

### Removing or Reordering logos:
- To remove: delete the SVG from `assets/logos/` and its two `config.py` entries.
- To reorder: renumber the filename prefixes.

### Changing Timings:
In `generator/config.py`, modify the `TIMING` dictionary:
```python
TIMING = {
    "clean_portrait_hold": 2.5,
    "photo_dissolve": 1.0,         
    "particle_portrait_hold": 1.5,
    "logo_transition": 1.5,        
    "logo_hold": 2.5,
    "reconstruct": 1.0             
}
```

## Generation

Ensure you have Node.js and Python installed.

```bash
cd generator
npm install
pip install -r requirements.txt
python build_hero.py
```

The output will be saved to `output/banner.svg`.
