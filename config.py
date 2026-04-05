"""
Configuration for Herman Miller Chair Finder.
CLI args, environment loading, constants, timing, search terms, pricing.
"""

import os
import sys
import json
import random
import argparse
import pytz
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Find Herman Miller chairs on Facebook Marketplace"
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Run in pure test mode (skip Facebook, use test images)",
)
parser.add_argument(
    "--benchmark",
    action="store_true",
    help="Benchmark mode: test model accuracy on FB + HM images",
)
parser.add_argument(
    "--list-benchmarks", action="store_true", help="List all previous benchmark runs"
)
parser.add_argument(
    "--compare",
    nargs=2,
    metavar=("RUN1", "RUN2"),
    help='Compare two benchmark runs (use timestamp or "latest")',
)
parser.add_argument(
    "--prod",
    action="store_true",
    help="Production mode (slower, more human-like delays)",
)
parser.add_argument(
    "--dev", action="store_true", help="Dev mode (faster delays, default)"
)
parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
parser.add_argument("--quiet", "-q", action="store_true", help="Minimal logging")
parser.add_argument(
    "--count",
    "-n",
    type=int,
    default=None,
    help="Number of listings to check (default: from env or 20)",
)
parser.add_argument(
    "--scheduler",
    action="store_true",
    help="Run as scheduler (12x/day, 9am-2am local time)",
)
parser.add_argument(
    "--backend",
    choices=["openrouter", "anthropic"],
    default=None,
    help="AI backend: openrouter (default) or anthropic (uses ANTHROPIC_API_KEY)",
)
args = parser.parse_args()

# Mode config (can be overridden by CLI args)
DEV_MODE = not args.prod  # Default to dev mode unless --prod specified
VERBOSE_LOGGING = args.verbose or (not args.quiet)  # Verbose by default unless --quiet
PURE_TEST_MODE = args.test
BENCHMARK_MODE = args.benchmark
# LISTING_COUNT set later after env is loaded
TEST_MODE_CHANCE = 15  # 1 in X chance to trigger test mode during normal run

# HEADLESS_MODE set after env loading

# Timing config (in milliseconds)
if DEV_MODE:
    SCROLL_DELAY_MIN = 2000
    SCROLL_DELAY_MAX = 4000
    LISTING_DELAY_MIN = 1500
    LISTING_DELAY_MAX = 3000
    SCROLL_COUNT = 3
else:
    # Production: slower, more random, more human-like
    SCROLL_DELAY_MIN = 2000
    SCROLL_DELAY_MAX = 6000
    LISTING_DELAY_MIN = 3000
    LISTING_DELAY_MAX = 12000
    SCROLL_COUNT = random.randint(15, 30)  # Way more scrolling

# Search terms - varied queries that might surface HM chairs
# Not too generic (office chair) nor too specific (aeron) - somewhere in between
CHAIR_SEARCHES = [
    # General furniture/office terms - cast a wide net
    {"query": "ergonomic chair", "analyze": True},
    {"query": "mesh chair", "analyze": True},
    {"query": "desk chair", "analyze": True},
    {"query": "computer chair", "analyze": True},
    {"query": "work from home chair", "analyze": True},
    {"query": "home office chair", "analyze": True},
    {"query": "task chair", "analyze": True},
    {"query": "adjustable chair", "analyze": True},
    {"query": "lumbar support chair", "analyze": True},
    {"query": "executive chair", "analyze": True},
    {"query": "gaming chair", "analyze": True},  # Sometimes HM mislabeled
    {"query": "swivel chair", "analyze": True},
    {"query": "office furniture", "analyze": True},
    {"query": "study chair", "analyze": True},
    {"query": "black chair", "analyze": True},
    {"query": "grey chair", "analyze": True},
]

# Decoy searches - just browse, don't analyze (look human)
DECOY_SEARCHES = [
    "desk lamp",
    "monitor stand",
    "keyboard",
    "laptop stand",
    "bookshelf",
    "plant pot",
    "coffee table",
    "standing desk",
    "filing cabinet",
    "desk organizer",
    "monitor arm",
    "webcam",
    "usb hub",
    "mouse pad",
    "desk mat",
    "cable management",
    "printer",
    "scanner",
    "headphones",
    "speakers",
    "microphone",
    "ring light",
    "whiteboard",
    "corkboard",
    "storage box",
    "drawer unit",
    "coat rack",
]

# Retail prices for premium chairs (used for deal scoring)
# Format: brand/model -> retail price in USD
CHAIR_RETAIL_PRICES = {
    # Herman Miller
    "Aeron": 1395,
    "Embody": 1795,
    "Mirra": 1045,
    "Sayl": 695,
    "Cosm": 1295,
    # Steelcase
    "Steelcase Leap": 1400,
    "Steelcase Gesture": 2000,
    "Steelcase Karman": 1200,
    # Humanscale
    "Humanscale Freedom": 1200,
    "Humanscale Liberty": 900,
    # Haworth
    "Haworth Fern": 1500,
    "Haworth Zody": 1000,
}

# Deal thresholds (percentage of retail price)
DEAL_THRESHOLDS = {
    "fumble": 0.15,  # <15% of retail = seller fumbled hard (10/10 deal)
    "steal": 0.25,  # <25% of retail = absolute steal (8-9/10)
    "great": 0.40,  # <40% of retail = great deal (6-7/10)
    "good": 0.60,  # <60% of retail = good deal (4-5/10)
    "fair": 0.80,  # <80% of retail = fair price (2-3/10)
    "retail": 1.0,  # ~retail price (0-1/10)
}


def parse_price(price_str):
    """Extract numeric price from string like '$150' or '$1,200'."""
    if not price_str:
        return None
    import re

    match = re.search(r"\$?([\d,]+(?:\.\d{2})?)", str(price_str).replace(",", ""))
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def calculate_deal_score(listing_price, chair_model):
    """
    Calculate deal score 0-10 based on listing price vs retail.
    Returns (score, label, retail_price) or (None, None, None) if unknown.
    """
    price = parse_price(listing_price)
    if price is None:
        return None, None, None

    # Find retail price - check exact match first, then partial
    retail = None
    matched_model = None
    for model, retail_price in CHAIR_RETAIL_PRICES.items():
        if model.lower() in chair_model.lower() or chair_model.lower() in model.lower():
            retail = retail_price
            matched_model = model
            break

    if retail is None:
        return None, None, None

    ratio = price / retail

    # Calculate score and label
    if ratio <= DEAL_THRESHOLDS["fumble"]:
        score = 10
        label = "\U0001f525 FUMBLE"
    elif ratio <= DEAL_THRESHOLDS["steal"]:
        score = 9 if ratio <= 0.20 else 8
        label = "\U0001f48e STEAL"
    elif ratio <= DEAL_THRESHOLDS["great"]:
        score = 7 if ratio <= 0.32 else 6
        label = "\U0001f3af GREAT"
    elif ratio <= DEAL_THRESHOLDS["good"]:
        score = 5 if ratio <= 0.50 else 4
        label = "\U0001f44d GOOD"
    elif ratio <= DEAL_THRESHOLDS["fair"]:
        score = 3 if ratio <= 0.70 else 2
        label = "\U0001f610 FAIR"
    else:
        score = 1 if ratio <= 1.1 else 0
        label = "\U0001f4b8 RETAIL" if ratio <= 1.1 else "\u274c OVERPRICED"

    return score, label, retail


# Configuration
# Load from environment variables (use .env file or export them)
def _load_env_file():
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(
                        key.strip(), value.strip().strip('"').strip("'")
                    )


_load_env_file()

# Multiple API keys for parallel batch processing (comma-separated in env)
_api_keys_str = os.environ.get("OPENROUTER_API_KEYS", "")
OPENROUTER_API_KEYS = [k.strip() for k in _api_keys_str.split(",") if k.strip()]
OPENROUTER_API_KEY = OPENROUTER_API_KEYS[0] if OPENROUTER_API_KEYS else ""

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", "")
TO_EMAIL = os.environ.get("TO_EMAIL", "")

# Facebook cookies loaded from environment (JSON string)
_fb_cookies_str = os.environ.get("FB_COOKIES", "[]")
try:
    FB_COOKIES = json.loads(_fb_cookies_str)
except json.JSONDecodeError:
    FB_COOKIES = []

# Output directory (relative to script location)
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "found_chairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SQLite database for tracking found listings
DB_PATH = OUTPUT_DIR / "found_listings.db"

# Lock file to ensure only one browser instance runs at a time
LOCK_FILE = OUTPUT_DIR / ".hm_finder.lock"

# Timezone and location for scheduler and browser (configurable via environment)
TIMEZONE = os.environ.get("TIMEZONE", "UTC")
LOCAL_TZ = pytz.timezone(TIMEZONE)
LATITUDE = float(os.environ.get("LATITUDE", "0"))
LONGITUDE = float(os.environ.get("LONGITUDE", "0"))
LOCALE = os.environ.get("LOCALE", "en-US")
# Multiple cities supported - comma separated (e.g., "perth,melbourne,sydney")
_locations_str = os.environ.get(
    "MARKETPLACE_LOCATIONS", os.environ.get("MARKETPLACE_LOCATION", "melbourne")
)
MARKETPLACE_LOCATIONS = [
    loc.strip() for loc in _locations_str.split(",") if loc.strip()
]
CURRENT_CITY = MARKETPLACE_LOCATIONS[0]  # Will be updated when looping through cities

# Scheduler config (all configurable via environment)
SCHEDULER_RUNS_PER_DAY = int(os.environ.get("RUNS_PER_DAY", "12"))
SCHEDULER_START_HOUR = int(os.environ.get("START_HOUR", "9"))  # 9am local default
_end_hour = int(os.environ.get("END_HOUR", "2"))  # 2am local default
SCHEDULER_END_HOUR = (
    _end_hour if _end_hour > 12 else _end_hour + 24
)  # Handle next-day hours

# Alert settings
MIN_DEAL_SCORE = float(
    os.environ.get("MIN_DEAL_SCORE", "0")
)  # Minimum score to send alerts (0 = all)
MIN_CONFIDENCE = float(
    os.environ.get("MIN_CONFIDENCE", "70")
)  # Minimum AI confidence % to consider

# Scraping settings
DEFAULT_LISTING_COUNT = int(
    os.environ.get("LISTING_COUNT", "20")
)  # Default listings per run
LISTING_COUNT = args.count if args.count is not None else DEFAULT_LISTING_COUNT

# Headless mode: configurable via env, defaults to auto-detect (headless if no display)
_headless_env = os.environ.get("HEADLESS", "").lower()
if _headless_env in ("true", "1", "yes"):
    HEADLESS_MODE = True
elif _headless_env in ("false", "0", "no"):
    HEADLESS_MODE = False
else:
    HEADLESS_MODE = not os.environ.get("DISPLAY")  # Auto-detect

# Default model for analysis (configurable via env)
DEFAULT_MODEL = os.environ.get("AI_MODEL", "anthropic/claude-opus-4")

# Anthropic direct API key (for --backend anthropic)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# AI backend selection: CLI flag > env var > default (openrouter)
_backend_env = os.environ.get("AI_BACKEND", "openrouter").lower()
AI_BACKEND = args.backend or _backend_env  # CLI flag takes priority
