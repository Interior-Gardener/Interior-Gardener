# ============================================
# GitHub Profile Banner Configuration
# ============================================
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------- Personal ----------
NAME = "Kartik Verma"
GITHUB_USERNAME = "Interior-Gardener"

BACKGROUND = "#0A101F"


# ---------- Assets ----------
PORTRAIT_PHOTO = ROOT / "assets" / "portrait_final.png"
PORTRAIT_SVG = ROOT / "assets" / "portrait_svg.svg"
LOGOS_DIR = ROOT / "assets" / "logos"

ROLE = "Full-Stack Developer"
LOCATION = "Mumbai, India"
EDUCATION = "B.Tech Computer Engineering (Honors – AIML)"

# ---------- Tech Stack ----------
LANGUAGES = "C · Java · Python · JavaScript"
FRONTEND = "HTML5 · CSS3 · Bootstrap · Tailwind · React.js"
BACKEND = "Node.js · Express.js · Flask · Django"
DATABASE = "MySQL · MongoDB · Oracle Database · Supabase"
INFRA = "AWS · Docker · Render · Vercel"

# ---------- Social ----------
LINKEDIN = "https://www.linkedin.com/in/kartikverma2204/"
EMAIL = "kartikverma2204@gmail.com"
PORTFOLIO = "https://kartik-verma.onrender.com/"
GITHUB = "github.com/Interior-Gardener"

# ---------- Animation Timeline (in seconds) ----------
TIMING = {
    "clean_portrait_hold": 2.5,
    "photo_dissolve": 1.0,         # photo -> particle portrait
    "particle_portrait_hold": 1.5,
    "logo_transition": 1.5,        # particles forming logo
    "logo_hold": 2.5,
    "reconstruct": 1.0             # particle portrait -> photo
}

# ---------- Particle Settings ----------
PARTICLES = {
    "count": 1500,
    "target_width": 300,  # Size of logo bounding box
    "target_height": 300
}

# ---------- Default Palette ----------
# The 5 base portrait colors (from light to dark)
PORTRAIT_COLORS = ["#22D3EE", "#38BDF8", "#A78BFA", "#8B5CF6", "#7C3AED"]

# ---------- Logo Brand Colors ----------
# Auto-extraction picks the most *frequent* hex in an SVG, which is often the
# wrong one: MongoDB's grey wordmark beats its green leaf, AWS's near-black
# #252F3E beats its orange, and stroke-only marks (Express) carry no hex at all.
# Anything listed here wins; everything else falls back to auto-extraction,
# which now prefers vivid colors and lifts them clear of the dark background.
LOGO_BRAND_COLORS = {
    "01-javascript": "#F7DF1E",
    "02-java":       "#E76F00",
    "03-react":      "#61DAFB",
    "04-nodejs":     "#68C15A",
    "05-express":    "#E2E8F0",
    "06-mongodb":    "#00ED64",
    "07-mysql":      "#00A0C6",
    "08-oracle":     "#F80000",
    "09-docker":     "#2496ED",
    "10-aws":        "#FF9900",
}

# Caption shown under the portrait while each logo is held.
LOGO_LABELS = {
    "01-javascript": "JAVASCRIPT",
    "02-java":       "JAVA",
    "03-react":      "REACT.JS",
    "04-nodejs":     "NODE.JS",
    "05-express":    "EXPRESS.JS",
    "06-mongodb":    "MONGODB",
    "07-mysql":      "MYSQL",
    "08-oracle":     "ORACLE DB",
    "09-docker":     "DOCKER",
    "10-aws":        "AWS",
}

# Caption shown while the particles hold the portrait.
PORTRAIT_LABEL = "KARTIK VERMA"

# Minimum relative luminance (0-255) a particle color may have against
# BACKGROUND. Brand colors darker than this get lifted until they read.
MIN_PARTICLE_LUMA = 96
