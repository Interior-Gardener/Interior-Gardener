import re
import os
import random
from pathlib import Path
import base64
import numpy as np
import config
from animation_engine import compute_traveller_routes, generate_fluid_keyframes
from io import BytesIO
from PIL import Image
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
BANNER_OUT = OUTPUT_DIR / "banner.svg"

def get_base64_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((400, 490), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=75)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join(c + c for c in hex_color)
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))

def relative_luma(hex_color):
    r, g, b = hex_to_rgb(hex_color)
    return 0.299 * r + 0.587 * g + 0.114 * b

def saturation(hex_color):
    r, g, b = hex_to_rgb(hex_color)
    mx, mn = max(r, g, b), min(r, g, b)
    return 0.0 if mx == 0 else (mx - mn) / mx

def lift_to_min_luma(hex_color, min_luma):
    """Blend a color toward white until it clears the background."""
    luma = relative_luma(hex_color)
    if luma >= min_luma or luma >= 255:
        return hex_color
    t = (min_luma - luma) / (255.0 - luma)
    r, g, b = hex_to_rgb(hex_color)
    return rgb_to_hex(r + (255 - r) * t, g + (255 - g) * t, b + (255 - b) * t)

def pick_dominant_color(svg_path):
    """
    Pick the color a human would call 'the brand color' of this mark.

    Frequency alone is a poor signal: MongoDB's grey wordmark outnumbers its
    green leaf and AWS's near-black outnumbers its orange. So vivid colors are
    preferred over greys, and frequency only breaks ties among vivid ones.
    An explicit entry in config.LOGO_BRAND_COLORS always wins.
    """
    override = config.LOGO_BRAND_COLORS.get(svg_path.stem)
    if override:
        return override

    content = svg_path.read_text(encoding='utf-8')
    hex_colors = []
    for m in re.findall(r'#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b', content):
        if len(m) == 4:
            m = '#' + m[1] * 2 + m[2] * 2 + m[3] * 2
        hex_colors.append(m.upper())

    if not hex_colors:
        return None

    counter = Counter(hex_colors)
    vivid = [(c, n) for c, n in counter.items() if saturation(c) >= 0.25]
    pool = vivid if vivid else list(counter.items())
    # Most frequent first, then most saturated, then brightest.
    pool.sort(key=lambda cn: (cn[1], saturation(cn[0]), relative_luma(cn[0])), reverse=True)
    return pool[0][0]

def extract_logo_palette(svg_path, num_tones=5):
    """
    Build the 5-tone particle ramp for one logo.

    Tones run dark -> light around the brand color, and every tone is then
    lifted above config.MIN_PARTICLE_LUMA so no bucket of particles vanishes
    into the near-black banner background.
    """
    try:
        dominant = pick_dominant_color(svg_path)
        if not dominant:
            return config.PORTRAIT_COLORS

        r, g, b = hex_to_rgb(dominant)
        palette = []
        for i in range(num_tones):
            factor = 0.6 + (i / max(1, num_tones - 1)) * 0.8
            tone = rgb_to_hex(min(255, r * factor), min(255, g * factor), min(255, b * factor))
            palette.append(lift_to_min_luma(tone, config.MIN_PARTICLE_LUMA))
        return palette
    except Exception as e:
        print(f"Warning: color extraction failed for {svg_path}: {e}")
        return config.PORTRAIT_COLORS

def build_d_string(pts):
    parts = []
    for x, y in pts:
        parts.append(f"M{int(x)} {int(y)}h2v2h-2z")
    return "".join(parts)

def run():
    print("Parsing portrait_svg.svg to extract initial particles and colors...")
    with open(config.PORTRAIT_SVG, "r") as f:
        svg_content = f.read()

    paths = re.findall(r'<path fill=\"([^\"]+)\" d=\"([^\"]+)\"', svg_content)
    
    all_dots = []
    for fill, d in paths:
        for cmd in d.split('z'):
            if not cmd: continue
            m = re.search(r'M(\d+)\s*(\d+)', cmd)
            if m:
                all_dots.append({'x': int(m.group(1)), 'y': int(m.group(2)), 'fill': fill})
                
    num_hero = config.PARTICLES["count"]
    if len(all_dots) < num_hero:
        indices = np.random.choice(len(all_dots), num_hero, replace=True)
    else:
        indices = np.random.choice(len(all_dots), num_hero, replace=False)
        
    hero_indices_set = set(indices)
    hero_dots = [all_dots[i] for i in indices]
    static_dots = [all_dots[i] for i in range(len(all_dots)) if i not in hero_indices_set]
    
    scale = 490 / 283
    tx = 40 + (400 - (231 * scale)) / 2
    ty = 85
    
    static_paths_str = ""
    color_map = {}
    for d in static_dots:
        c = d['fill']
        if c not in color_map: color_map[c] = []
        color_map[c].append(d)
    for c, dots in color_map.items():
        d_str = "".join([f"M{int(d['x'] * scale + tx)} {int(d['y'] * scale + ty)}h1v1h-1z" for d in dots])
        static_paths_str += f'<path fill="{c}" d="{d_str}" />\n'
        
    hero_pts = []
    for d in hero_dots:
        hx = d['x'] * scale + tx
        hy = d['y'] * scale + ty
        hero_pts.append([hx, hy])
    hero_pts = np.array(hero_pts, dtype=np.float32)
    
    def color_brightness(hex_c):
        r, g, b = hex_to_rgb(hex_c)
        return 0.299*r + 0.587*g + 0.114*b
        
    for d in hero_dots:
        d['brightness'] = color_brightness(d['fill'])
        
    sorted_pairs = sorted(zip(hero_dots, hero_pts), key=lambda x: x[0]['brightness'])
    hero_dots = [p[0] for p in sorted_pairs]
    hero_pts = np.array([p[1] for p in sorted_pairs])
    
    buckets = 5
    bucket_size = num_hero // buckets
    
    print("Auto-discovering logos...")
    valid_logos = []
    palettes = [config.PORTRAIT_COLORS] # index 0 is portrait
    
    for ext in ['*.svg']:
        for logo_path in sorted(config.LOGOS_DIR.glob(ext)):
            print(f"Found {logo_path.name}")
            valid_logos.append(logo_path)
            palettes.append(extract_logo_palette(logo_path, buckets))
            
    if not valid_logos:
        print("No valid logos found in assets/logos/. Exiting.")
        return
        
    print(f"Discovered {len(valid_logos)} logos.")
    
    target_box = config.PARTICLES["target_width"]
    cx = 40 + 400 / 2
    cy = 85 + 490 / 2
    
    routes = compute_traveller_routes(hero_pts, valid_logos, target_box, target_box, num_hero)
    
    if not routes:
        print("Failed to compute routes. Exiting.")
        return
        
    for r in routes[1:-1]:
        r[:, 0] += cx - target_box/2
        r[:, 1] += cy - target_box/2
        
    print("Generating strict timeline...")
    
    dur = config.TIMING
    
    timeline_d = []          # (time, points, color_palette_index)
    timeline_photo = []      # (time, opacity)
    timeline_static = []     # (time, opacity)
    timeline_hero_op = []    # (time, opacity)
    label_windows = []       # (text, color, t_on, t_off)
    
    t = 0.0
    # A. CLEAN PORTRAIT HOLD
    timeline_d.append((t, routes[0], 0))
    timeline_photo.append((t, 1))
    timeline_static.append((t, 0))
    timeline_hero_op.append((t, 0))
    
    t += dur["clean_portrait_hold"]
    timeline_d.append((t, routes[0], 0))
    timeline_photo.append((t, 1))
    timeline_static.append((t, 0))
    timeline_hero_op.append((t, 0))
    
    # A -> B: PHOTO DISSOLVE
    t += dur["photo_dissolve"]
    timeline_d.append((t, routes[0], 0))
    timeline_photo.append((t, 0))
    timeline_static.append((t, 1))
    timeline_hero_op.append((t, 1))
    
    # B: PARTICLE PORTRAIT HOLD
    t += dur["particle_portrait_hold"]
    timeline_d.append((t, routes[0], 0))
    timeline_photo.append((t, 0))
    timeline_static.append((t, 1))
    timeline_hero_op.append((t, 1))
    label_windows.append((config.PORTRAIT_LABEL, config.PORTRAIT_COLORS[0],
                          t - dur["particle_portrait_hold"], t))
    
    # Logos Iteration
    for i in range(len(valid_logos)):
        # Transition to Logo i
        start_t = t
        end_t = t + dur["logo_transition"]
        frames = generate_fluid_keyframes(routes[i], routes[i+1], steps=6, turbulence=25.0)
        
        for j in range(1, len(frames)):
            frame_t = start_t + (end_t - start_t) * (j / (len(frames)-1))
            # The palette index smoothly transitions to i+1
            timeline_d.append((frame_t, frames[j], i+1))
            
        timeline_photo.append((end_t, 0))
        timeline_static.append((start_t + 0.1, 0)) # Disappear as soon as particles leave
        timeline_static.append((end_t, 0))
        timeline_hero_op.append((end_t, 1))
        
        t = end_t
        
        # Logo Hold
        t += dur["logo_hold"]
        timeline_d.append((t, routes[i+1], i+1))
        timeline_photo.append((t, 0))
        timeline_static.append((t, 0))
        timeline_hero_op.append((t, 1))

        # Caption fades in as the particles land and holds with the logo.
        stem = valid_logos[i].stem
        label = config.LOGO_LABELS.get(stem, stem.split("-")[-1].upper())
        label_windows.append((label, palettes[i + 1][2], end_t - 0.4, t))
        
    # N -> B: FINAL LOGO TO PORTRAIT
    start_t = t
    end_t = t + dur["logo_transition"]
    frames = generate_fluid_keyframes(routes[-2], routes[-1], steps=6, turbulence=25.0)
    for j in range(1, len(frames)):
        frame_t = start_t + (end_t - start_t) * (j / (len(frames)-1))
        timeline_d.append((frame_t, frames[j], 0)) # transition back to portrait colors
        
    timeline_photo.append((end_t, 0))
    timeline_static.append((start_t, 0))
    timeline_static.append((end_t, 1)) # Reappear when particles arrive
    timeline_hero_op.append((end_t, 1))
    t = end_t
    
    # B: PARTICLE PORTRAIT HOLD (Closing)
    t += dur["particle_portrait_hold"]
    timeline_d.append((t, routes[-1], 0))
    timeline_photo.append((t, 0))
    timeline_static.append((t, 1))
    timeline_hero_op.append((t, 1))
    label_windows.append((config.PORTRAIT_LABEL, config.PORTRAIT_COLORS[0],
                          t - dur["particle_portrait_hold"], t))
    
    # B -> A: RECONSTRUCT
    t += dur["reconstruct"]
    timeline_d.append((t, routes[-1], 0))
    timeline_photo.append((t, 1))
    timeline_static.append((t, 0))
    timeline_hero_op.append((t, 0))
    
    total_dur = t
    print(f"Total animation loop: {total_dur}s")
    
    path_elements = ""
    for b in range(buckets):
        start_idx = b * bucket_size
        end_idx = (b+1) * bucket_size if b < buckets-1 else num_hero
        
        d_values = []
        fill_values = []
        kt_d = []
        
        for frame_t, frame_pts, pal_idx in timeline_d:
            sub_pts = frame_pts[start_idx:end_idx]
            d_values.append(build_d_string(sub_pts))
            fill_values.append(palettes[pal_idx][b])
            kt_d.append(f"{frame_t / total_dur:.3f}")
            
        kt_str = ";".join(kt_d)
        
        path_elements += f'''
        <path fill="{palettes[0][b]}">
            <animate attributeName="d" values="{";".join(d_values)}" keyTimes="{kt_str}" dur="{total_dur}s" repeatCount="indefinite" />
            <animate attributeName="fill" values="{";".join(fill_values)}" keyTimes="{kt_str}" dur="{total_dur}s" repeatCount="indefinite" />
        </path>
        '''
        
    def format_op(timeline):
        kt = ";".join([f"{time/total_dur:.3f}" for time, op in timeline])
        val = ";".join([str(op) for time, op in timeline])
        return kt, val
        
    def label_track(t_on, t_off, fade=0.35):
        """Full-cycle opacity keyframes that show a caption only in its window."""
        raw = [(0.0, 0), (t_on - fade, 0), (t_on, 1),
               (t_off, 1), (t_off + fade, 0), (total_dur, 0)]
        pts = []
        for tt, op in raw:
            tt = min(max(tt, 0.0), total_dur)
            if pts and tt <= pts[-1][0]:
                pts[-1] = (pts[-1][0], op)
            else:
                pts.append((tt, op))
        return (";".join(f"{tt / total_dur:.4f}" for tt, _ in pts),
                ";".join(str(op) for _, op in pts))

    label_elements = ""
    for text, color, t_on, t_off in label_windows:
        kt, vals = label_track(t_on, t_off)
        label_elements += f'''
    <text x="440" y="72" text-anchor="end" fill="{color}" font-size="13" letter-spacing="3" opacity="0">{text}<animate attributeName="opacity" values="{vals}" keyTimes="{kt}" dur="{total_dur}s" repeatCount="indefinite" /></text>'''

    photo_kt, photo_val = format_op(timeline_photo)
    static_kt, static_val = format_op(timeline_static)
    hero_kt, hero_val = format_op(timeline_hero_op)
    
    print("Writing SVG...")
    photo_b64 = get_base64_image(config.PORTRAIT_PHOTO)
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 610" width="1180" height="610">
    <defs>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&amp;family=Inter:wght@400;500;700&amp;display=swap');
            text {{ font-family: 'JetBrains Mono', monospace; }}
            .label {{ fill: #94A3B8; font-size: 14px; }}
            .val {{ fill: #F8FAFC; font-weight: 700; font-size: 14px; }}
            .title {{ fill: #22D3EE; font-size: 12px; letter-spacing: 2px; }}
        </style>
        <filter id="photoBlur">
            <feGaussianBlur stdDeviation="0">
                <!-- We map opacity inversion to blur: when opacity is 0, blur is 8 -->
                <animate attributeName="stdDeviation" values="{';'.join(['0' if op==1 else '8' for _, op in timeline_photo])}" keyTimes="{photo_kt}" dur="{total_dur}s" repeatCount="indefinite" />
            </feGaussianBlur>
        </filter>
        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M40 0H0V40" fill="none" stroke="#22D3EE" stroke-width="1" opacity="0.045" />
        </pattern>
        <clipPath id="frameClip">
            <rect x="2" y="2" width="1176" height="606" rx="14" />
        </clipPath>
    </defs>

    <!-- Background -->
    <rect width="1180" height="610" fill="{config.BACKGROUND}" />
    <rect width="1180" height="610" fill="url(#grid)" clip-path="url(#frameClip)" />
    
    <!-- Outer Window Frame -->
    <rect x="2" y="2" width="1176" height="606" rx="14" fill="none" stroke="#22D3EE" stroke-width="2" opacity="0.4" filter="url(#glow)" />
    <rect x="2" y="2" width="1176" height="606" rx="14" fill="none" stroke="#22D3EE" stroke-width="1" />
    <rect x="2" y="2" width="1176" height="42" rx="14" fill="#070B16" />
    <line x1="2" y1="44" x2="1178" y2="44" stroke="rgba(255,255,255,0.1)" />

    <!-- Window controls -->
    <circle cx="24" cy="23" r="6" fill="#FF5F56" />
    <circle cx="44" cy="23" r="6" fill="#FFBD2E" />
    <circle cx="64" cy="23" r="6" fill="#27C93F" />
    <text x="590" y="27" fill="#64748B" font-size="12" text-anchor="middle">{config.EMAIL} - ./profile.sh --hero</text>
    <circle cx="1104" cy="23" r="4" fill="#22D3EE">
        <animate attributeName="opacity" values="1;0.2;1" dur="2.4s" repeatCount="indefinite" />
    </circle>
    <text x="1118" y="27" fill="#22D3EE" font-size="11" letter-spacing="2">LIVE</text>

    <!-- Left Box (Portrait area) -->
    <text x="40" y="72" class="title" fill="#64748B">VISUAL.MAP</text>{label_elements}
    <rect x="40" y="85" width="400" height="490" rx="8" fill="#070B16" stroke="rgba(34,211,238,0.2)" />

    <!-- Photo Layer -->
    <image x="40" y="85" width="400" height="490" href="data:image/jpeg;base64,{photo_b64}" preserveAspectRatio="xMidYMid slice" filter="url(#photoBlur)">
        <animate attributeName="opacity" values="{photo_val}" keyTimes="{photo_kt}" dur="{total_dur}s" repeatCount="indefinite" />
    </image>

    <!-- Static Background Dots -->
    <g opacity="0">
        <animate attributeName="opacity" values="{static_val}" keyTimes="{static_kt}" dur="{total_dur}s" repeatCount="indefinite" />
        {static_paths_str}
    </g>
    
    <!-- Dynamic Path Morphing Particles -->
    <g filter="url(#glow)" opacity="0">
        <animate attributeName="opacity" values="{hero_val}" keyTimes="{hero_kt}" dur="{total_dur}s" repeatCount="indefinite" />
        {path_elements}
    </g>

    <!-- Right Box (Info area) -->
    <text x="500" y="96" class="title">SYSTEM.INFO</text>
    <line x1="612" y1="92" x2="1140" y2="92" stroke="rgba(255,255,255,0.1)" />

    <text x="500" y="132" class="label">Name ........... <tspan class="val">{config.NAME}</tspan></text>
    <text x="500" y="160" class="label">Role ........... <tspan class="val">{config.ROLE}</tspan></text>
    <text x="500" y="188" class="label">Location ....... <tspan class="val">{config.LOCATION}</tspan></text>
    <text x="500" y="216" class="label">Education ...... <tspan class="val">{config.EDUCATION}</tspan></text>

    <text x="500" y="262" class="title">STACK.INFO</text>
    <line x1="600" y1="258" x2="1140" y2="258" stroke="rgba(255,255,255,0.1)" />

    <text x="500" y="298" class="label">Core.Lang ...... <tspan class="val">{config.LANGUAGES}</tspan></text>
    <text x="500" y="326" class="label">Core.Front ..... <tspan class="val">{config.FRONTEND}</tspan></text>
    <text x="500" y="354" class="label">Core.Back ...... <tspan class="val">{config.BACKEND}</tspan></text>
    <text x="500" y="382" class="label">Core.Data ...... <tspan class="val">{config.DATABASE}</tspan></text>
    <text x="500" y="410" class="label">Core.Infra ..... <tspan class="val">{config.INFRA}</tspan></text>

    <text x="500" y="456" class="title">NETWORK.INFO</text>
    <line x1="624" y1="452" x2="1140" y2="452" stroke="rgba(255,255,255,0.1)" />

    <text x="500" y="492" class="label">Email .......... <tspan class="val">{config.EMAIL}</tspan></text>
    <text x="500" y="520" class="label">Portfolio ...... <tspan class="val">{config.PORTFOLIO.replace("https://", "").rstrip("/")}</tspan></text>
    <text x="500" y="548" class="label">GitHub ......... <tspan class="val">{config.GITHUB}</tspan></text>
    <text x="500" y="576" class="label">LinkedIn ....... <tspan class="val">linkedin.com/in/{config.LINKEDIN.split("in/")[1].strip("/")}</tspan></text>
</svg>
'''

    BANNER_OUT.write_text(svg, encoding="utf-8")
    print(f"Saved optimized banner to {BANNER_OUT}")

if __name__ == "__main__":
    run()
