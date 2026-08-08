import re
import os
import random
from pathlib import Path
import base64
import numpy as np
import cv2
import config
from animation_engine import compute_traveller_routes

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
BANNER_OUT = OUTPUT_DIR / "banner.svg"

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def run():
    print("Parsing portrait_svg.svg...")
    with open(ROOT / "assets" / "portrait_svg.svg", "r") as f:
        svg_content = f.read()

    # Extract all paths
    paths = re.findall(r'<path fill=\"([^\"]+)\" d=\"([^\"]+)\"', svg_content)
    
    # We will pick 1500 dots randomly to be the "hero" particles.
    # The SVG contains commands like 'M x y h1 v1 H x Z'.
    
    all_dots = []
    # all_dots will be a list of dicts: {'x': x, 'y': y, 'fill': color}
    for fill, d in paths:
        commands = d.split('z')
        for cmd in commands:
            if not cmd: continue
            m = re.search(r'M(\d+)\s*(\d+)', cmd)
            if m:
                all_dots.append({
                    'x': int(m.group(1)),
                    'y': int(m.group(2)),
                    'fill': fill
                })
                
    print(f"Found {len(all_dots)} total dots.")
    
    num_hero = 1500
    indices = np.random.choice(len(all_dots), num_hero, replace=False)
    hero_indices_set = set(indices)
    
    hero_dots = [all_dots[i] for i in indices]
    static_dots = [all_dots[i] for i in range(len(all_dots)) if i not in hero_indices_set]
    
    print(f"Separated into {len(hero_dots)} hero dots and {len(static_dots)} static dots.")
    
    # Bounding box of original SVG is 231x283 (from <svg viewBox="0 0 231 283">).
    # We want to place it in our banner, bounding box approx 400x490, at x=40, y=85.
    scale = 490 / 283
    tx = 40 + (400 - (231 * scale)) / 2
    ty = 85
    
    # Generate the static paths
    # To save space, we group them by color
    color_map = {}
    for d in static_dots:
        c = d['fill']
        if c not in color_map:
            color_map[c] = []
        color_map[c].append(d)
        
    static_paths_str = ""
    for c, dots in color_map.items():
        d_str = ""
        for d in dots:
            d_str += f"M{d['x']} {d['y']}h1v1H{d['x']}z"
        static_paths_str += f'<path fill="{c}" d="{d_str}" />\n'
        
    # Scale and translate the hero dots for the transport algorithm
    hero_pts = []
    for d in hero_dots:
        hx = d['x'] * scale + tx
        hy = d['y'] * scale + ty
        hero_pts.append([hx, hy])
    hero_pts = np.array(hero_pts, dtype=np.float32)
    
    print("Computing transport routes...")
    logo_paths = [
        ROOT / "assets" / config.REACT_LOGO_FILE,
        ROOT / "assets" / config.NODE_LOGO_FILE,
        ROOT / "assets" / config.MONGODB_LOGO_FILE
    ]
    # Logo target bounding box should be centered at the portrait box
    routes = compute_traveller_routes(hero_pts, logo_paths, 300, 300, num_hero)
    
    # routes = [P0, L1, L2, L3]
    # We need to offset the logos to be centered over the portrait
    cx = 40 + 400 / 2
    cy = 85 + 490 / 2
    
    # We must ensure L1, L2, L3 are centered
    for r in routes[1:]:
        # compute_traveller_routes already centers them at (target_w/2, target_h/2)
        # So we just shift by (cx - 150), (cy - 150)
        r[:, 0] += cx - 150
        r[:, 1] += cy - 150

    print("Generating SMIL animations...")
    # Timeline (24s loop):
    # 0s - 3s: Photo static
    # 3s - 5s: Photo fades out, Particle fades in
    # 5s - 7s: Hero morphs to React (L1)
    # 7s - 9s: Hold React
    # 9s - 11s: Morph Node (L2)
    # 11s - 13s: Hold Node
    # 13s - 15s: Morph Mongo (L3)
    # 15s - 17s: Hold Mongo
    # 17s - 19s: Morph back to P0
    # 19s - 21s: Hold P0
    # 21s - 23s: Particle fades out, Photo fades in
    # 23s - 24s: Hold Photo
    
    # Keytimes for the movement
    kt = [
        "0",           # 0s: P0
        "0.208",       # 5s: start move
        "0.291",       # 7s: L1
        "0.375",       # 9s: hold L1
        "0.458",       # 11s: L2
        "0.541",       # 13s: hold L2
        "0.625",       # 15s: L3
        "0.708",       # 17s: hold L3
        "0.791",       # 19s: P0
        "1"            # 24s: hold P0
    ]
    kt_str = ";".join(kt)
    
    hero_circles = ""
    react_color = "#22D3EE"
    node_color = "#27C93F"
    mongo_color = "#10B981"

    for i in range(num_hero):
        p0 = routes[0][i]
        l1 = routes[1][i]
        l2 = routes[2][i]
        l3 = routes[3][i]
        color = hero_dots[i]['fill']
        
        sx, sy = hero_pts[i]
        dx1 = l1[0] - sx; dy1 = l1[1] - sy
        dx2 = l2[0] - sx; dy2 = l2[1] - sy
        dx3 = l3[0] - sx; dy3 = l3[1] - sy
        
        vals = [
            f"0 0",                 # 0-5s: Stay at P0
            f"0 0",                 # 5s: Start moving
            f"{dx1:.1f} {dy1:.1f}", # 7s: Arrive React
            f"{dx1:.1f} {dy1:.1f}", # 9s: Hold React
            f"{dx2:.1f} {dy2:.1f}", # 11s: Arrive Node
            f"{dx2:.1f} {dy2:.1f}", # 13s: Hold Node
            f"{dx3:.1f} {dy3:.1f}", # 15s: Arrive Mongo
            f"{dx3:.1f} {dy3:.1f}", # 17s: Hold Mongo
            f"0 0",                 # 19s: Arrive P0
            f"0 0"                  # 24s: Hold P0
        ]
        val_str = ";".join(vals)
        
        colors = [
            color, color, react_color, react_color, node_color, node_color, mongo_color, mongo_color, color, color
        ]
        col_str = ";".join(colors)
        
        hero_circles += f'''
        <circle cx="{sx:.1f}" cy="{sy:.1f}" r="{1.5*scale:.1f}" fill="{color}">
            <animateTransform attributeName="transform" type="translate" values="{val_str}" keyTimes="{kt_str}" dur="24s" repeatCount="indefinite" />
            <animate attributeName="fill" values="{col_str}" keyTimes="{kt_str}" dur="24s" repeatCount="indefinite" />
        </circle>
        '''
        
    print("Assembling final SVG...")
    
    # Raster photo base64
    photo_b64 = get_base64_image(ROOT / "assets" / "portrait_final.png")
    
    # We construct the SVG
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 610" width="1180" height="610">
    <defs>
        <!-- Fonts & Text Styles -->
        <style>
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&amp;family=Inter:wght@400;500;700&amp;display=swap');
            text {{ font-family: 'JetBrains Mono', monospace; }}
            .label {{ fill: #94A3B8; font-size: 14px; }}
            .val {{ fill: #F8FAFC; font-weight: 700; font-size: 14px; }}
            .title {{ fill: #22D3EE; font-size: 12px; letter-spacing: 2px; }}
        </style>
        <!-- Photo Blur Filter -->
        <filter id="photoBlur">
            <feGaussianBlur stdDeviation="0">
                <animate attributeName="stdDeviation" values="0;0;5;5;0;0" keyTimes="0;0.125;0.208;0.875;0.958;1" dur="24s" repeatCount="indefinite" />
            </feGaussianBlur>
        </filter>
        <!-- Glow for borders -->
        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="6" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
    </defs>

    <!-- Background -->
    <rect width="1180" height="610" fill="#0A101F" />
    
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

    <!-- Left Box (Portrait area) -->
    <text x="40" y="72" class="title" fill="#64748B">VISUAL.MAP</text>
    <rect x="40" y="85" width="400" height="490" rx="8" fill="#070B16" stroke="rgba(34,211,238,0.2)" />
    
    <!-- The Scanning Progress Bar (Slider) -->
    <rect x="40" y="575" width="400" height="4" fill="rgba(34,211,238,0.1)" rx="2" />
    <rect x="40" y="575" width="0" height="4" fill="#22D3EE" filter="url(#glow)" rx="2">
        <animate attributeName="width" values="0;400;0" keyTimes="0;0.958;1" dur="24s" repeatCount="indefinite" />
    </rect>

    <!-- Photo Layer -->
    <!-- Opacity fades out from 3s to 5s (0.125 to 0.208) -->
    <image x="40" y="85" width="400" height="490" href="data:image/png;base64,{photo_b64}" preserveAspectRatio="xMidYMid slice" filter="url(#photoBlur)">
        <animate attributeName="opacity" values="1;1;0;0;1;1" keyTimes="0;0.125;0.208;0.875;0.958;1" dur="24s" repeatCount="indefinite" />
    </image>

    <!-- SVG Particle Background Layer -->
    <!-- Opacity fades in from 3-5s, fades out from 5-7s, stays out until 17s, fades in 17-19s, holds until 21s, fades out 21-23s -->
    <g opacity="0">
        <animate attributeName="opacity" values="0;0;1;0;0;1;1;0;0" keyTimes="0;0.125;0.208;0.291;0.708;0.791;0.875;0.958;1" dur="24s" repeatCount="indefinite" />
        <g transform="translate({tx:.2f}, {ty:.2f}) scale({scale:.4f})">
            {static_paths_str}
        </g>
    </g>
    
    <!-- Flying Hero Particles -->
    <g opacity="0">
        <animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;0.125;0.208;0.875;0.958;1" dur="24s" repeatCount="indefinite" />
        {hero_circles}
    </g>

    <!-- Right Box (Info area) -->
    <text x="500" y="100" class="title">SYSTEM.INFO</text>
    <line x1="590" y1="96" x2="1140" y2="96" stroke="rgba(255,255,255,0.1)" />
    
    <text x="500" y="140" class="label">Name ........... <tspan class="val">{config.NAME}</tspan></text>
    <text x="500" y="170" class="label">Role ........... <tspan class="val">{config.ROLE}</tspan></text>
    <text x="500" y="200" class="label">Location ....... <tspan class="val">{config.LOCATION}</tspan></text>
    <text x="500" y="230" class="label">Education ...... <tspan class="val">{config.EDUCATION}</tspan></text>
    
    <text x="500" y="280" class="title">STACK.INFO</text>
    <line x1="590" y1="276" x2="1140" y2="276" stroke="rgba(255,255,255,0.1)" />
    
    <text x="500" y="320" class="label">Core.Lang ...... <tspan class="val">{config.LANGUAGES}</tspan></text>
    <text x="500" y="350" class="label">Core.Front ..... <tspan class="val">{config.FRONTEND}</tspan></text>
    <text x="500" y="380" class="label">Core.Back ...... <tspan class="val">{config.BACKEND}</tspan></text>
    <text x="500" y="410" class="label">Core.Data ...... <tspan class="val">{config.DATABASE}</tspan></text>
    <text x="500" y="440" class="label">Core.Infra ..... <tspan class="val">{config.INFRA}</tspan></text>
    
    <text x="500" y="490" class="title">NETWORK.INFO</text>
    <line x1="600" y1="486" x2="1140" y2="486" stroke="rgba(255,255,255,0.1)" />
    
    <text x="500" y="530" class="label">Email .......... <tspan class="val">{config.EMAIL}</tspan></text>
    <text x="500" y="560" class="label">Portfolio ...... <tspan class="val">{config.PORTFOLIO.replace("https://", "")}</tspan></text>
    
</svg>
'''

    BANNER_OUT.write_text(svg, encoding="utf-8")
    print(f"Saved banner to {BANNER_OUT}")

if __name__ == "__main__":
    run()
