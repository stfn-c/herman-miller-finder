#!/usr/bin/env python3
"""
Herman Miller Chair Finder
Uses Playwright to scrape Facebook Marketplace for office chairs,
uses Claude Opus 4.5 via OpenRouter to identify Herman Miller chairs,
and sends email alerts for matches.

Usage:
  python find_herman_miller.py              # Normal mode (scrapes Facebook)
  python find_herman_miller.py --test       # Pure test mode (uses test images only)
  python find_herman_miller.py --prod       # Production mode (slower, more human-like)
  python find_herman_miller.py --dev        # Dev mode (faster, default)
  python find_herman_miller.py --verbose    # Verbose logging
  python find_herman_miller.py --quiet      # Minimal logging
  python find_herman_miller.py --test --verbose  # Combine flags
"""

import os
import sys
import json
import time
import base64
import asyncio
import random
import argparse
import requests
import sqlite3
import fcntl
import pytz
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright

# Parse command line arguments
parser = argparse.ArgumentParser(description='Find Herman Miller chairs on Facebook Marketplace')
parser.add_argument('--test', action='store_true', help='Run in pure test mode (skip Facebook, use test images)')
parser.add_argument('--benchmark', action='store_true', help='Benchmark mode: test model accuracy on FB + HM images')
parser.add_argument('--list-benchmarks', action='store_true', help='List all previous benchmark runs')
parser.add_argument('--compare', nargs=2, metavar=('RUN1', 'RUN2'), help='Compare two benchmark runs (use timestamp or "latest")')
parser.add_argument('--prod', action='store_true', help='Production mode (slower, more human-like delays)')
parser.add_argument('--dev', action='store_true', help='Dev mode (faster delays, default)')
parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
parser.add_argument('--quiet', '-q', action='store_true', help='Minimal logging')
parser.add_argument('--count', '-n', type=int, default=20, help='Number of listings to check (default: 20)')
parser.add_argument('--scheduler', action='store_true', help='Run as scheduler (12x/day, 9am-2am local time)')
args = parser.parse_args()

# Mode config (can be overridden by CLI args)
DEV_MODE = not args.prod  # Default to dev mode unless --prod specified
VERBOSE_LOGGING = args.verbose or (not args.quiet)  # Verbose by default unless --quiet
PURE_TEST_MODE = args.test
BENCHMARK_MODE = args.benchmark
LISTING_COUNT = args.count
TEST_MODE_CHANCE = 15   # 1 in X chance to trigger test mode during normal run

# Headless mode: auto-detect if no display available (server mode)
HEADLESS_MODE = not os.environ.get('DISPLAY')

# Available models for benchmarking (vision-capable)
# Format: (model_id, name, approx_cost_per_1M_tokens input/output)
# Model IDs verified against OpenRouter API on 2025-12-07 - VISION MODELS ONLY
BENCHMARK_MODELS = [
    # Premium - one of each (most recent)
    ("anthropic/claude-opus-4.5", "Claude Opus 4.5", "$15/$75"),
    ("google/gemini-2.5-pro", "Gemini 2.5 Pro", "$1.25/$10"),

    # High-tier - one of each
    ("anthropic/claude-sonnet-4", "Claude Sonnet 4", "$3/$15"),
    ("openai/gpt-4o", "GPT-4o", "$2.50/$10"),
    ("google/gemini-2.5-flash", "Gemini 2.5 Flash", "$0.15/$0.60"),

    # Mid-tier
    ("openai/gpt-4o-mini", "GPT-4o Mini", "$0.15/$0.60"),
    ("google/gemini-2.0-flash-001", "Gemini 2.0 Flash", "$0.10/$0.40"),
    ("anthropic/claude-3-haiku", "Claude 3 Haiku", "$0.25/$1.25"),
    ("meta-llama/llama-3.2-90b-vision-instruct", "Llama 3.2 90B Vision", "$0.20/$0.60"),
    ("meta-llama/llama-3.2-11b-vision-instruct", "Llama 3.2 11B Vision", "$0.10/$0.10"),

    # Budget / Specialized
    ("google/gemini-2.0-flash-lite-001", "Gemini 2.0 Flash Lite", "$0.075/$0.30"),
    ("qwen/qwen-vl-max", "Qwen VL Max", "$0.40/$0.40"),
    ("mistralai/pixtral-large-2411", "Pixtral Large", "$2/$6"),
    ("mistralai/pixtral-12b", "Pixtral 12B", "$0.10/$0.10"),
]

# Real Herman Miller chair images for testing
# Mix of Reddit/FB Marketplace finds + official product photos
TEST_HERMAN_MILLER_IMAGES = [
    # From Reddit - real marketplace-style photos (what we're hunting for)
    {"url": "https://preview.redd.it/purchasing-aeron-off-fb-marketplace-is-this-herman-chair-v0-uh9ahc8krgc81.jpg?width=640&crop=smart&auto=webp&s=47e5f8feb4a844973539eb12052a72287ecd3952", "title": "Office Chair", "price": "$80"},
    {"url": "https://preview.redd.it/purchasing-aeron-off-fb-marketplace-is-this-herman-chair-v0-s7if7g8krgc81.jpg?width=640&crop=smart&auto=webp&s=8373066fe13bb848a40fcb57f6e751724a848b65", "title": "Computer Chair", "price": "$50"},
    {"url": "https://preview.redd.it/purchasing-aeron-off-fb-marketplace-is-this-herman-chair-v0-e5a4sc8krgc81.jpg?width=640&crop=smart&auto=webp&s=99c73ea787480e95dbbf336ef283cb080c5ffad5", "title": "Mesh Office Chair", "price": "$120"},
    {"url": "https://preview.redd.it/purchasing-aeron-off-fb-marketplace-is-this-herman-chair-v0-l1aiqc8krgc81.jpg?width=640&crop=smart&auto=webp&s=affe0b9ce7613a731bae3fa849aea689c441d9a3", "title": "Desk Chair - Good Condition", "price": "$75"},
    {"url": "https://preview.redd.it/purchasing-aeron-off-fb-marketplace-is-this-herman-chair-v0-ek6ntd8krgc81.jpg?width=640&crop=smart&auto=webp&s=4b8cef62e545fd4ef390fe4e8f99b4360ce241d4", "title": "Black Office Chair", "price": "$100"},
    {"url": "https://i.redd.it/fb-marketplace-is-crazy-v0-os3h00el8awc1.jpg?width=4032&format=pjpg&auto=webp&s=e7bd2013b1b675cdb60777f5d742598969f0a820", "title": "Office Chair Moving Sale", "price": "$45"},
    {"url": "https://preview.redd.it/facebook-marketplace-for-300-v0-5wgf6fxbovhf1.jpg?width=640&crop=smart&auto=webp&s=87049a79533b8846edeebc56fb6ea6f2c12a5fa0", "title": "Ergonomic Chair", "price": "$300"},
    {"url": "https://preview.redd.it/won-the-fb-marketplace-lottery-today-v0-hx8qrh61r23g1.jpg?width=640&crop=smart&auto=webp&s=5c0e617d2a61924a1211a9700ebbe2f24c711778", "title": "Chair - Must Go Today", "price": "$60"},

    # Official Herman Miller product photos (cleaner, easier to recognize)
    {"url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/aeron_chair/202106/mh_prd_ovw_aeron_chair.jpg", "title": "Work Chair", "price": "$150"},
    {"url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/aeron_chair/202106/ig_prd_ovw_aeron_chair_01.jpg", "title": "Office Furniture", "price": "$200"},
    {"url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/embody_chairs/mh_prd_ovw_embody_chairs.jpg", "title": "Desk Chair Blue", "price": "$175"},
    {"url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/mirra_2_chair/mh_prd_ovw_mirra_2_chair.jpg", "title": "Mesh Chair", "price": "$90"},
    {"url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/sayl_chairs/mh_prd_ovw_sayl_chairs.jpg", "title": "Modern Office Chair", "price": "$125"},
    {"url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/cosm_chairs/northamerica/mh_prd_ovw_cosm_chairs_na.jpg", "title": "Ergonomic Seat", "price": "$250"},

    # More lifestyle/room shots from Herman Miller
    {"url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/aeron_chair/202106/ig_prd_ovw_aeron_chair_02.jpg", "title": "Home Office Chair", "price": "$180"},
    {"url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/aeron_chair/202106/ig_prd_ovw_aeron_chair_03.jpg", "title": "Conference Chair", "price": "$95"},
    {"url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/aeron_chair/202106/ig_prd_ovw_aeron_chair_04.jpg", "title": "Task Chair", "price": "$160"},
    {"url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/aeron_chair/202106/ig_prd_ovw_aeron_chair_05.jpg", "title": "Office Seating", "price": "$110"},
]

# Premium NON-Herman Miller chairs - should be identified as NOT HM
# These are expensive quality chairs that models might confuse with Herman Miller
TEST_OTHER_PREMIUM_CHAIRS = [
    # Steelcase Leap - $1,400+ retail, LiveBack technology
    {"url": "https://steelcase-res.cloudinary.com/image/upload/v1610026604/20-0149894.jpg", "title": "Office Chair", "price": "$200", "actual_brand": "Steelcase Leap"},
    # Steelcase Gesture - $2,000+ retail, 360-degree arms
    {"url": "https://images.steelcase.com/image/upload/v1676059815/21-0166043-1.jpg", "title": "Ergonomic Desk Chair", "price": "$150", "actual_brand": "Steelcase Gesture"},
    # Humanscale Freedom - $1,200+ retail, self-adjusting recline
    {"url": "https://www.ergodirect.com/images/Humanscale/13611/large/Humanscale-Freedom-Task-Chair_lg_1745860590.jpg", "title": "Task Chair", "price": "$175", "actual_brand": "Humanscale Freedom"},
    # Humanscale Liberty - $900+ retail, tri-panel mesh back
    {"url": "https://cdn11.bigcommerce.com/s-492apnl0xy/images/stencil/1280x1280/products/744/3282/humanscale-liberty-chair-hus088__49475.1490806767.jpg?c=2", "title": "Mesh Office Chair", "price": "$120", "actual_brand": "Humanscale Liberty"},
    # Haworth Fern - $1,500+ retail, Wave Suspension system
    {"url": "https://store.haworth.com/cdn/shop/files/Fern-Mesh_53ffb43c-2638-4ce3-a324-ae702c3fc1ef.jpg?v=1720535915", "title": "Executive Chair", "price": "$250", "actual_brand": "Haworth Fern"},
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
    "fumble": 0.15,      # <15% of retail = seller fumbled hard (10/10 deal)
    "steal": 0.25,       # <25% of retail = absolute steal (8-9/10)
    "great": 0.40,       # <40% of retail = great deal (6-7/10)
    "good": 0.60,        # <60% of retail = good deal (4-5/10)
    "fair": 0.80,        # <80% of retail = fair price (2-3/10)
    "retail": 1.0,       # ~retail price (0-1/10)
}


def parse_price(price_str):
    """Extract numeric price from string like '$150' or '$1,200'."""
    if not price_str:
        return None
    import re
    match = re.search(r'\$?([\d,]+(?:\.\d{2})?)', str(price_str).replace(',', ''))
    if match:
        return float(match.group(1).replace(',', ''))
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
        label = "🔥 FUMBLE"
    elif ratio <= DEAL_THRESHOLDS["steal"]:
        score = 9 if ratio <= 0.20 else 8
        label = "💎 STEAL"
    elif ratio <= DEAL_THRESHOLDS["great"]:
        score = 7 if ratio <= 0.32 else 6
        label = "🎯 GREAT"
    elif ratio <= DEAL_THRESHOLDS["good"]:
        score = 5 if ratio <= 0.50 else 4
        label = "👍 GOOD"
    elif ratio <= DEAL_THRESHOLDS["fair"]:
        score = 3 if ratio <= 0.70 else 2
        label = "😐 FAIR"
    else:
        score = 1 if ratio <= 1.1 else 0
        label = "💸 RETAIL" if ratio <= 1.1 else "❌ OVERPRICED"

    return score, label, retail


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
    "desk lamp", "monitor stand", "keyboard", "laptop stand", "bookshelf",
    "plant pot", "coffee table", "standing desk", "filing cabinet", "desk organizer",
    "monitor arm", "webcam", "usb hub", "mouse pad", "desk mat", "cable management",
    "printer", "scanner", "headphones", "speakers", "microphone", "ring light",
    "whiteboard", "corkboard", "storage box", "drawer unit", "coat rack"
]

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
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

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
MARKETPLACE_LOCATION = os.environ.get("MARKETPLACE_LOCATION", "melbourne")  # Facebook marketplace city slug

# Scheduler config: 12 runs per day, only during waking hours (9am-2am local time)
SCHEDULER_RUNS_PER_DAY = 12
SCHEDULER_START_HOUR = 9   # 9am local
SCHEDULER_END_HOUR = 26    # 2am next day (26 = 24 + 2)


def init_database():
    """Initialize SQLite database for tracking found listings."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS listings (
            listing_id TEXT PRIMARY KEY,
            url TEXT,
            title TEXT,
            price TEXT,
            model TEXT,
            confidence TEXT,
            reasoning TEXT,
            deal_score INTEGER,
            deal_label TEXT,
            found_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            alerted_at TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def is_listing_known(listing_id):
    """Check if we've already processed this listing."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM listings WHERE listing_id = ?', (listing_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def save_listing_to_db(listing_id, url, title, price, model, confidence, reasoning, deal_score=None, deal_label=None):
    """Save a found Herman Miller listing to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO listings
        (listing_id, url, title, price, model, confidence, reasoning, deal_score, deal_label, found_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (listing_id, url, title, price, model, confidence, reasoning, deal_score, deal_label, datetime.now()))
    conn.commit()
    conn.close()


def mark_listing_alerted(listing_id):
    """Mark a listing as having been included in an email alert."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE listings SET alerted_at = ? WHERE listing_id = ?', (datetime.now(), listing_id))
    conn.commit()
    conn.close()


def get_unalerted_listings():
    """Get listings that haven't been included in an alert yet."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM listings WHERE alerted_at IS NULL ORDER BY found_at DESC')
    columns = [description[0] for description in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results


def get_listing_stats():
    """Get stats about found listings."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM listings')
    total = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM listings WHERE alerted_at IS NOT NULL')
    alerted = cursor.fetchone()[0]
    cursor.execute('SELECT model, COUNT(*) as cnt FROM listings GROUP BY model ORDER BY cnt DESC')
    by_model = cursor.fetchall()
    conn.close()
    return {"total": total, "alerted": alerted, "by_model": by_model}


# Initialize database on import
init_database()


def list_benchmark_runs():
    """List all previous benchmark runs with summary stats."""
    benchmark_files = sorted(OUTPUT_DIR.glob("benchmark_*.json"), reverse=True)

    if not benchmark_files:
        print("No benchmark runs found.")
        return

    print("\n" + "=" * 80)
    print("📊 Previous Benchmark Runs")
    print("=" * 80)
    print(f"{'Timestamp':<20} {'Models':<8} {'Images':<8} {'Best Model':<25} {'Accuracy':<10}")
    print("-" * 80)

    for f in benchmark_files:
        try:
            with open(f) as fp:
                data = json.load(fp)

            timestamp = f.stem.replace("benchmark_", "")
            num_models = len(data.get("models_tested", []))
            num_images = len(data.get("images", []))

            # Calculate best model
            model_scores = {}
            for img in data.get("images", []):
                actual = img.get("is_actually_herman_miller", False)
                for r in img.get("results", []):
                    name = r.get("model_name", "Unknown")
                    if name not in model_scores:
                        model_scores[name] = {"correct": 0, "total": 0}
                    if r.get("analysis"):
                        analysis = r["analysis"]
                        # Check is_premium, is_herman_miller, brand, AND model name
                        predicted = analysis.get("is_premium", False) or analysis.get("is_herman_miller", False)
                        brand = analysis.get("brand", "").lower()
                        model = analysis.get("model", "").lower()
                        if brand in ["herman miller", "steelcase", "humanscale", "haworth"]:
                            predicted = True
                        hm_models = ["aeron", "embody", "mirra", "sayl", "cosm"]
                        other_premium = ["leap", "gesture", "freedom", "liberty", "fern", "zody", "karman"]
                        if any(m in model for m in hm_models + other_premium):
                            predicted = True
                        model_scores[name]["total"] += 1
                        if predicted == actual:
                            model_scores[name]["correct"] += 1

            best_model = "N/A"
            best_acc = 0
            for name, scores in model_scores.items():
                if scores["total"] > 0:
                    acc = scores["correct"] / scores["total"]
                    if acc > best_acc:
                        best_acc = acc
                        best_model = name

            print(f"{timestamp:<20} {num_models:<8} {num_images:<8} {best_model:<25} {best_acc*100:.1f}%")
        except Exception as e:
            print(f"{f.stem:<20} Error reading: {e}")

    print("-" * 80)
    print(f"Total runs: {len(benchmark_files)}")
    print(f"Latest: {benchmark_files[0].stem if benchmark_files else 'N/A'}")
    print("\nTo view a report: open found_chairs/benchmark_<timestamp>.html")
    print("To compare runs: python find_herman_miller.py --compare <timestamp1> <timestamp2>")


def compare_benchmark_runs(run1_id, run2_id):
    """Compare two benchmark runs side by side."""
    def find_run(run_id):
        if run_id == "latest":
            files = sorted(OUTPUT_DIR.glob("benchmark_*.json"), reverse=True)
            return files[0] if files else None
        else:
            matches = list(OUTPUT_DIR.glob(f"benchmark_{run_id}*.json"))
            return matches[0] if matches else None

    file1 = find_run(run1_id)
    file2 = find_run(run2_id)

    if not file1:
        print(f"Could not find benchmark run: {run1_id}")
        return
    if not file2:
        print(f"Could not find benchmark run: {run2_id}")
        return

    with open(file1) as f:
        data1 = json.load(f)
    with open(file2) as f:
        data2 = json.load(f)

    print("\n" + "=" * 90)
    print("📊 Benchmark Comparison")
    print("=" * 90)
    print(f"Run 1: {file1.stem}")
    print(f"Run 2: {file2.stem}")
    print("-" * 90)

    # Calculate stats for each run
    def calc_model_stats(data):
        stats = {}
        for img in data.get("images", []):
            actual = img.get("is_actually_herman_miller", False)
            for r in img.get("results", []):
                name = r.get("model_name", "Unknown")
                if name not in stats:
                    stats[name] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
                if r.get("analysis"):
                    analysis = r["analysis"]
                    # Check is_premium, is_herman_miller, brand, AND model name
                    predicted = analysis.get("is_premium", False) or analysis.get("is_herman_miller", False)
                    brand = analysis.get("brand", "").lower()
                    model = analysis.get("model", "").lower()
                    if brand in ["herman miller", "steelcase", "humanscale", "haworth"]:
                        predicted = True
                    hm_models = ["aeron", "embody", "mirra", "sayl", "cosm"]
                    other_premium = ["leap", "gesture", "freedom", "liberty", "fern", "zody", "karman"]
                    if any(m in model for m in hm_models + other_premium):
                        predicted = True
                    if actual and predicted:
                        stats[name]["tp"] += 1
                    elif actual and not predicted:
                        stats[name]["fn"] += 1
                    elif not actual and predicted:
                        stats[name]["fp"] += 1
                    else:
                        stats[name]["tn"] += 1
        return stats

    stats1 = calc_model_stats(data1)
    stats2 = calc_model_stats(data2)

    all_models = set(stats1.keys()) | set(stats2.keys())

    print(f"{'Model':<25} {'Run1 Acc':<12} {'Run2 Acc':<12} {'Δ':<8} {'Run1 FP':<10} {'Run2 FP':<10}")
    print("-" * 90)

    for model in sorted(all_models):
        s1 = stats1.get(model, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        s2 = stats2.get(model, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})

        total1 = s1["tp"] + s1["fp"] + s1["tn"] + s1["fn"]
        total2 = s2["tp"] + s2["fp"] + s2["tn"] + s2["fn"]

        acc1 = (s1["tp"] + s1["tn"]) / total1 * 100 if total1 > 0 else 0
        acc2 = (s2["tp"] + s2["tn"]) / total2 * 100 if total2 > 0 else 0
        delta = acc2 - acc1

        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        delta_color = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"

        print(f"{model:<25} {acc1:>6.1f}%      {acc2:>6.1f}%      {delta_color}{delta_str:<6} {s1['fp']:<10} {s2['fp']:<10}")

    print("-" * 90)


def analyze_image_with_model(image_base64, model_id="anthropic/claude-opus-4", listing_title="", listing_price="", api_key=None):
    """Use specified model via OpenRouter to analyze if chair is Herman Miller."""

    url = "https://openrouter.ai/api/v1/chat/completions"
    key = api_key or OPENROUTER_API_KEY

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://yaupicks.com",
        "X-Title": "Herman Miller Finder"
    }

    # Build context from listing info (price excluded - irrelevant to identification)
    listing_context = ""
    if listing_title:
        listing_context = f"\n\nListing title: \"{listing_title}\"\n(Ignore the title for identification - sellers often mislabel chairs. Judge ONLY by visual features.)"

    # Simple HM-only prompt
    prompt_text = f"""Is this a Herman Miller chair?

Herman Miller models:
- AERON: Mesh with horizontal bands, curved figure-8 frame, PostureFit lumbar
- EMBODY: Pixelated spine-like back
- MIRRA: Butterfly-shaped frame
- SAYL: Y-shaped suspension back
- COSM: Continuous flowing frame

Answer in JSON:
{{"reasoning":"why or why not","model":"Aeron|Embody|Mirra|Sayl|Cosm|None","confidence":"high|medium|low","is_herman_miller":true|false}}"""

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ],
        "max_tokens": 4000
    }

    max_retries = 4
    base_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                original_content = content  # Keep for logging

                # Clean up the response - handle markdown, control chars, etc.
                content = content.strip()

                # Remove markdown code blocks
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                # Remove control characters (except newlines/tabs)
                import re
                content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)

                # Try to extract JSON object from response (in case there's extra text)
                json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)

                # Fix common JSON issues
                content = content.replace('\n', ' ').replace('\r', '')
                # Fix unescaped quotes in reasoning field
                content = re.sub(r'("reasoning"\s*:\s*")(.*?)(",\s*"(?:brand|model|confidence|is_premium|is_herman_miller))',
                                lambda m: m.group(1) + m.group(2).replace('"', "'") + m.group(3),
                                content, flags=re.DOTALL)

                # Try to parse JSON, handle truncated/malformed responses
                try:
                    parsed = json.loads(content)
                    if VERBOSE_LOGGING:
                        brand = parsed.get('brand', parsed.get('model', 'Unknown'))
                        is_p = parsed.get('is_premium', parsed.get('is_herman_miller', False))
                        print(f"      ✓ {model_id.split('/')[-1]}: {brand} (premium={is_p})")
                    return parsed
                except json.JSONDecodeError as e:
                    # Log the parse failure
                    print(f"      ⚠️ {model_id.split('/')[-1]} JSON parse failed: {str(e)[:50]}")
                    if VERBOSE_LOGGING:
                        print(f"         Raw response: {original_content[:150]}...")

                    # Try to extract key fields from malformed response
                    is_premium_match = re.search(r'"is_premium"\s*:\s*(true|false)', content, re.IGNORECASE)
                    is_hm_match = re.search(r'"is_herman_miller"\s*:\s*(true|false)', content, re.IGNORECASE)
                    brand_match = re.search(r'"brand"\s*:\s*"([^"]*)"', content)
                    model_match = re.search(r'"model"\s*:\s*"([^"]*)"', content)
                    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)', content)
                    conf_match = re.search(r'"confidence"\s*:\s*"([^"]*)"', content)

                    # Determine is_premium from either field
                    is_premium = False
                    if is_premium_match:
                        is_premium = is_premium_match.group(1).lower() == 'true'
                    elif is_hm_match:
                        is_premium = is_hm_match.group(1).lower() == 'true'

                    if is_premium_match or is_hm_match or brand_match:
                        salvaged = {
                            "reasoning": f"[SALVAGED] {reason_match.group(1) if reason_match else 'parse error'}",
                            "brand": brand_match.group(1) if brand_match else "Unknown",
                            "model": model_match.group(1) if model_match else "Unknown",
                            "confidence": conf_match.group(1) if conf_match else "low",
                            "is_premium": is_premium,
                            "is_herman_miller": is_hm_match.group(1).lower() == 'true' if is_hm_match else False,
                            "salvaged": True
                        }
                        print(f"      🔧 {model_id.split('/')[-1]} SALVAGED: {salvaged.get('brand')} {salvaged.get('model')}")
                        return salvaged

                    # Can't salvage - return minimal response
                    print(f"      ❌ {model_id.split('/')[-1]} FAILED: couldn't parse or salvage")
                    return {
                        "reasoning": f"[PARSE ERROR] {content[:100]}...",
                        "brand": "Unknown",
                        "model": "None",
                        "confidence": "low",
                        "is_premium": False,
                        "error": True
                    }
            elif response.status_code == 429:
                # Rate limited - retry with exponential backoff
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                if attempt < max_retries - 1:
                    print(f"      ⏳ {model_id.split('/')[-1]} rate limited, retry {attempt+1}/{max_retries} in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"      ❌ {model_id.split('/')[-1]} rate limited after {max_retries} retries")
                    return {"error": True, "reasoning": "Rate limited", "brand": "Unknown", "model": "None", "is_premium": False}
            elif response.status_code >= 500:
                # Server error - retry
                if attempt < max_retries - 1:
                    print(f"      ⏳ {model_id.split('/')[-1]} server error {response.status_code}, retrying...")
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                else:
                    print(f"      ❌ {model_id.split('/')[-1]} server error: {response.status_code}")
                    return {"error": True, "reasoning": f"Server error {response.status_code}", "brand": "Unknown", "model": "None", "is_premium": False}
            else:
                print(f"      ❌ {model_id.split('/')[-1]} API error: {response.status_code} - {response.text[:100]}")
                return {"error": True, "reasoning": f"API error {response.status_code}", "brand": "Unknown", "model": "None", "is_premium": False}
        except requests.exceptions.Timeout:
            print(f"      ⏳ {model_id.split('/')[-1]} timeout, retry {attempt+1}/{max_retries}...")
            if attempt < max_retries - 1:
                time.sleep(base_delay)
                continue
            print(f"      ❌ {model_id.split('/')[-1]} timed out after {max_retries} retries")
            return {"error": True, "reasoning": "Timeout", "brand": "Unknown", "model": "None", "is_premium": False}
        except Exception as e:
            print(f"      ❌ {model_id.split('/')[-1]} error: {type(e).__name__}: {str(e)[:50]}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            return {"error": True, "reasoning": f"{type(e).__name__}", "brand": "Unknown", "model": "None", "is_premium": False}

    return {"error": True, "reasoning": "Unknown error", "brand": "Unknown", "model": "None", "is_premium": False}


def analyze_image_with_claude(image_base64):
    """Wrapper for backwards compatibility - uses Claude Opus 4."""
    return analyze_image_with_model(image_base64, "anthropic/claude-opus-4")


def save_herman_miller_listing(listing, analysis, image_data):
    """Save a Herman Miller listing to the output folder."""

    listing_dir = OUTPUT_DIR / f"{listing['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    listing_dir.mkdir(parents=True, exist_ok=True)

    # Save listing info
    info = {
        "id": listing["id"],
        "url": listing["url"],
        "title": listing.get("title", "Unknown"),
        "price": listing.get("price", "Unknown"),
        "analysis": analysis,
        "found_at": datetime.now().isoformat()
    }

    with open(listing_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Save image
    if image_data:
        with open(listing_dir / "image.jpg", "wb") as f:
            f.write(base64.b64decode(image_data))

    print(f"  💾 Saved to: {listing_dir}")
    return listing_dir


def send_email_alert(listings):
    """Send email alert with found premium chairs and deal scores."""

    if not listings:
        print("No premium chairs found to report")
        return

    # Sort by deal score (best deals first)
    sorted_listings = sorted(listings, key=lambda x: x.get('deal_score', 0), reverse=True)

    # Build email HTML
    html_items = []
    fumble_count = 0
    for listing in sorted_listings:
        # Get deal info
        deal_score = listing.get('deal_score')
        deal_label = listing.get('deal_label', '')
        retail_price = listing.get('retail_price')

        # Style based on deal quality
        if deal_score and deal_score >= 8:
            border_color = "#22c55e"  # Green for steals/fumbles
            bg_color = "#f0fdf4"
            fumble_count += 1
        elif deal_score and deal_score >= 6:
            border_color = "#3b82f6"  # Blue for great deals
            bg_color = "#eff6ff"
        elif deal_score and deal_score >= 4:
            border_color = "#eab308"  # Yellow for good deals
            bg_color = "#fefce8"
        else:
            border_color = "#ddd"
            bg_color = "#fff"

        # Build deal info string
        deal_info = ""
        if deal_score is not None:
            deal_info = f"""
            <div style="background: {bg_color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <span style="font-size: 24px; font-weight: bold;">{deal_label}</span>
                <span style="font-size: 18px; margin-left: 10px;">Deal Score: {deal_score}/10</span>
                <br><span style="color: #666;">Retail: ${retail_price} → Listed: {listing.get('price', 'Unknown')}</span>
            </div>
            """

        brand = listing.get('brand', 'Unknown')
        model = listing.get('model', 'Unknown')
        chair_name = f"{brand} {model}" if brand != "Unknown" else model

        html_items.append(f"""
        <div style="border: 2px solid {border_color}; padding: 15px; margin: 10px 0; border-radius: 8px; background: {bg_color};">
            <h3 style="margin: 0 0 10px 0;">{chair_name}</h3>
            {deal_info}
            <p><strong>Price:</strong> {listing.get('price', 'Unknown')}</p>
            <p><strong>Title:</strong> {listing.get('title', 'Unknown')}</p>
            <p><strong>Confidence:</strong> {listing.get('confidence', 'Unknown')}</p>
            <p><strong>Analysis:</strong> {listing.get('reasoning', 'N/A')}</p>
            <p><a href="{listing.get('url', '#')}" style="background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">View Listing →</a></p>
        </div>
        """)

    # Subject line reflects deal quality
    if fumble_count > 0:
        subject = f"🔥 {fumble_count} FUMBLE(S)! {len(listings)} Premium Chair(s) Found!"
    else:
        subject = f"🪑 {len(listings)} Premium Chair(s) Found on Marketplace!"

    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <h1 style="color: #333;">🪑 Premium Chairs Found!</h1>
        <p>Found {len(listings)} premium chair(s) on Facebook Marketplace:</p>
        {''.join(html_items)}
        <hr style="margin: 30px 0; border: none; border-top: 1px solid #ddd;">
        <p style="color: #666; font-size: 12px;">Deal Score Guide: 🔥 FUMBLE (10) = &lt;15% retail | 💎 STEAL (8-9) = &lt;25% | 🎯 GREAT (6-7) = &lt;40% | 👍 GOOD (4-5) = &lt;60%</p>
    </body>
    </html>
    """

    # Send via Resend
    url = "https://api.resend.com/emails"
    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "from": FROM_EMAIL,
        "to": TO_EMAIL,
        "subject": subject,
        "html": html_content
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code in [200, 201]:
            print(f"📧 Email sent successfully to {TO_EMAIL}")
        else:
            print(f"Failed to send email: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending email: {e}")


async def run_pure_test_mode():
    """Run through test images without touching Facebook - for testing the AI + email flow."""
    print("=" * 60)
    print("🧪 PURE TEST MODE - Herman Miller Chair Finder")
    print(f"Started at: {datetime.now()}")
    print("Testing with pre-loaded Herman Miller images...")
    print("=" * 60)

    herman_millers = []

    for i, test_listing in enumerate(TEST_HERMAN_MILLER_IMAGES):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(TEST_HERMAN_MILLER_IMAGES)}] Test Listing")
        print(f"{'='*60}")

        listing = {
            "id": f"test_{i+1}",
            "url": f"https://www.facebook.com/marketplace/item/test{i+1}/",
            "title": test_listing["title"],
            "price": test_listing["price"],
        }

        print(f"\n  📋 FAKE LISTING DATA:")
        print(f"     Title: {listing['title']}")
        print(f"     Price: {listing['price']}")
        print(f"     (This is what a seller might list a Herman Miller as)")

        print(f"\n  📷 Downloading image...")
        try:
            img_response = requests.get(test_listing["url"], timeout=15)
            if img_response.status_code == 200:
                image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                print(f"     Image size: {len(img_response.content) / 1024:.1f} KB")

                print(f"\n  🤖 Sending to Claude Opus 4.5...")
                analysis = analyze_image_with_claude(image_base64)

                if analysis:
                    print(f"\n  📊 AI ANALYSIS:")
                    print(f"     Reasoning: {analysis.get('reasoning', 'N/A')}")
                    print(f"     Model: {analysis.get('model', 'Unknown')}")
                    print(f"     Confidence: {analysis.get('confidence', 'Unknown')}")
                    print(f"     Is Herman Miller: {analysis.get('is_herman_miller', 'Unknown')}")

                    if analysis.get("is_herman_miller"):
                        print(f"\n  🎉🎉🎉 HERMAN MILLER FOUND! 🎉🎉🎉")
                        save_herman_miller_listing(listing, analysis, image_base64)
                        herman_millers.append({
                            **listing,
                            **analysis,
                            "is_test": True
                        })
                    else:
                        print(f"\n  ❌ Not recognized as Herman Miller")
                else:
                    print(f"\n  ❌ Analysis failed - no response from AI")
            else:
                print(f"     Failed to download: {img_response.status_code}")
        except Exception as e:
            print(f"\n  ❌ Error: {e}")

        # Small delay between tests
        print(f"\n  ⏳ Waiting 2s before next test...")
        await asyncio.sleep(2)

    # Send email alert
    if herman_millers:
        print(f"\n📧 Sending email alert with {len(herman_millers)} finds...")
        send_email_alert(herman_millers)
    else:
        print("\n📧 No Herman Miller chairs recognized, skipping email")

    print("\n" + "=" * 60)
    print(f"Test complete! AI recognized {len(herman_millers)}/{len(TEST_HERMAN_MILLER_IMAGES)} as Herman Miller")
    print("=" * 60)


import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


def analyze_single_model(args):
    """Worker function for parallel model testing."""
    model_id, model_name, cost, image_base64, title, price, api_key = args
    start_time = time.time()
    try:
        analysis = analyze_image_with_model(image_base64, model_id, title, price, api_key)
        elapsed = time.time() - start_time
        return {
            "model_id": model_id,
            "model_name": model_name,
            "cost": cost,
            "analysis": analysis,
            "elapsed_seconds": round(elapsed, 2),
            "error": None
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "model_id": model_id,
            "model_name": model_name,
            "cost": cost,
            "analysis": None,
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e)
        }


def generate_html_report(benchmark_data, output_path):
    """Generate a beautiful HTML report from benchmark data."""

    # Calculate summary stats per model
    model_stats = {}
    for model_id, model_name, cost in BENCHMARK_MODELS:
        model_stats[model_name] = {
            "model_id": model_id,
            "cost": cost,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0, "errors": 0,
            "total_time": 0
        }

    for img in benchmark_data["images"]:
        for result in img["results"]:
            name = result["model_name"]
            if name not in model_stats:
                continue

            if result["error"]:
                model_stats[name]["errors"] += 1
            elif result["analysis"]:
                # Check is_premium, is_herman_miller, brand, AND model name
                analysis = result["analysis"]
                predicted = analysis.get("is_premium", False) or analysis.get("is_herman_miller", False)

                # Also check brand and model name (in case boolean is wrong but identification is right)
                brand = analysis.get("brand", "").lower()
                model = analysis.get("model", "").lower()
                if brand in ["herman miller", "steelcase", "humanscale", "haworth"]:
                    predicted = True
                hm_models = ["aeron", "embody", "mirra", "sayl", "cosm"]
                other_premium = ["leap", "gesture", "freedom", "liberty", "fern", "zody", "karman"]
                if any(m in model for m in hm_models + other_premium):
                    predicted = True

                actual = img["is_actually_herman_miller"]

                if actual and predicted:
                    model_stats[name]["tp"] += 1
                elif actual and not predicted:
                    model_stats[name]["fn"] += 1
                elif not actual and predicted:
                    model_stats[name]["fp"] += 1
                else:
                    model_stats[name]["tn"] += 1

                model_stats[name]["total_time"] += result.get("elapsed_seconds", 0)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Herman Miller AI Benchmark</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #111; color: #fff; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 40px 20px; }}

        /* Header */
        header {{ text-align: center; margin-bottom: 50px; }}
        h1 {{ font-size: 2rem; font-weight: 600; margin-bottom: 8px; }}
        .subtitle {{ color: #666; font-size: 0.9rem; }}

        /* Stats Row */
        .stats {{ display: flex; justify-content: center; gap: 40px; margin-bottom: 50px; }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 2.5rem; font-weight: 700; }}
        .stat-value.green {{ color: #22c55e; }}
        .stat-value.blue {{ color: #3b82f6; }}
        .stat-label {{ color: #666; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }}

        /* Table */
        .table-wrapper {{ background: #1a1a1a; border-radius: 16px; overflow: hidden; margin-bottom: 50px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ padding: 16px 20px; text-align: left; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #666; border-bottom: 1px solid #333; }}
        td {{ padding: 16px 20px; border-bottom: 1px solid #222; }}
        tr:last-child td {{ border-bottom: none; }}
        tr:hover {{ background: #222; cursor: pointer; }}
        .model-name {{ font-weight: 500; }}
        .model-id {{ color: #666; font-size: 0.75rem; }}
        .cost {{ color: #888; font-size: 0.85rem; }}
        .acc-bar {{ width: 80px; height: 6px; background: #333; border-radius: 3px; overflow: hidden; display: inline-block; vertical-align: middle; margin-right: 8px; }}
        .acc-fill {{ height: 100%; border-radius: 3px; }}
        .acc-fill.high {{ background: #22c55e; }}
        .acc-fill.med {{ background: #eab308; }}
        .acc-fill.low {{ background: #ef4444; }}
        .num {{ color: #888; font-size: 0.85rem; }}
        .num.good {{ color: #22c55e; }}
        .num.bad {{ color: #ef4444; }}

        /* Section Headers */
        h2 {{ font-size: 1.2rem; font-weight: 600; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }}

        /* Filters */
        .filters {{ display: flex; gap: 8px; margin-bottom: 24px; }}
        .filter-btn {{ padding: 8px 16px; background: #222; border: none; color: #888; border-radius: 8px; cursor: pointer; font-size: 0.85rem; transition: all 0.2s; }}
        .filter-btn:hover, .filter-btn.active {{ background: #333; color: #fff; }}

        /* Image Grid */
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
        .image-card {{ background: #1a1a1a; border-radius: 12px; overflow: hidden; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }}
        .image-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); }}
        .image-card img {{ width: 100%; height: 200px; object-fit: cover; }}
        .image-card .info {{ padding: 16px; }}
        .image-card .title {{ font-weight: 500; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .image-card .meta {{ color: #666; font-size: 0.85rem; margin-bottom: 8px; }}
        .image-card .tag {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }}
        .image-card .tag.hm {{ background: rgba(34,197,94,0.15); color: #22c55e; }}
        .image-card .tag.fb {{ background: rgba(239,68,68,0.15); color: #ef4444; }}
        .image-card .score {{ float: right; font-size: 0.85rem; color: #888; }}
        .image-card .score span {{ color: #22c55e; font-weight: 600; }}

        /* Modal */
        .modal {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.95); z-index: 1000; overflow-y: auto; }}
        .modal.open {{ display: block; }}
        .modal-content {{ max-width: 900px; margin: 40px auto; padding: 20px; }}
        .modal-close {{ position: fixed; top: 20px; right: 30px; color: #666; font-size: 2rem; cursor: pointer; z-index: 1001; }}
        .modal-close:hover {{ color: #fff; }}
        .modal-header {{ display: flex; gap: 24px; margin-bottom: 30px; }}
        .modal-img {{ width: 300px; height: 300px; object-fit: cover; border-radius: 12px; cursor: pointer; }}
        .modal-info {{ flex: 1; }}
        .modal-info h3 {{ font-size: 1.5rem; margin-bottom: 8px; }}
        .modal-info .meta {{ color: #666; margin-bottom: 16px; }}
        .modal-info .tag {{ display: inline-block; padding: 6px 14px; border-radius: 8px; font-size: 0.85rem; font-weight: 600; }}
        .modal-info .link {{ color: #3b82f6; text-decoration: none; font-size: 0.9rem; margin-top: 16px; display: inline-block; }}
        .modal-info .link:hover {{ text-decoration: underline; }}

        /* Model Results in Modal */
        .results-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px; }}
        .result-card {{ background: #222; border-radius: 10px; padding: 14px; border-left: 3px solid #333; }}
        .result-card.correct {{ border-left-color: #22c55e; }}
        .result-card.wrong {{ border-left-color: #ef4444; }}
        .result-card .model {{ font-weight: 500; font-size: 0.9rem; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center; }}
        .result-card .verdict {{ padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
        .result-card .verdict.yes {{ background: rgba(34,197,94,0.2); color: #22c55e; }}
        .result-card .verdict.no {{ background: rgba(239,68,68,0.2); color: #ef4444; }}
        .result-card .reasoning {{ color: #888; font-size: 0.8rem; line-height: 1.5; margin-top: 8px; }}
        .result-card .time {{ color: #555; font-size: 0.75rem; margin-top: 6px; }}

        /* Fullscreen Image */
        .fullscreen {{ display: none; position: fixed; inset: 0; background: #000; z-index: 2000; justify-content: center; align-items: center; }}
        .fullscreen.open {{ display: flex; }}
        .fullscreen img {{ max-width: 95%; max-height: 95%; object-fit: contain; }}
        .fullscreen-close {{ position: absolute; top: 20px; right: 30px; color: #fff; font-size: 2rem; cursor: pointer; }}

        /* Model Detail View */
        .model-detail-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }}
        .model-detail-header h3 {{ font-size: 1.5rem; }}
        .model-detail-stats {{ display: flex; gap: 24px; }}
        .model-detail-stats .stat {{ text-align: center; }}
        .model-detail-stats .stat-val {{ font-size: 1.5rem; font-weight: 700; }}
        .model-detail-stats .stat-val.green {{ color: #22c55e; }}
        .model-detail-stats .stat-val.red {{ color: #ef4444; }}
        .model-detail-stats .stat-lbl {{ color: #666; font-size: 0.75rem; text-transform: uppercase; }}
        .model-images-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }}
        .model-img-card {{ background: #222; border-radius: 10px; overflow: hidden; border-left: 3px solid #333; }}
        .model-img-card.correct {{ border-left-color: #22c55e; }}
        .model-img-card.wrong {{ border-left-color: #ef4444; }}
        .model-img-card img {{ width: 100%; height: 140px; object-fit: cover; cursor: pointer; }}
        .model-img-card .details {{ padding: 12px; }}
        .model-img-card .img-title {{ font-size: 0.85rem; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .model-img-card .img-meta {{ font-size: 0.75rem; color: #666; margin: 4px 0; }}
        .model-img-card .verdict-row {{ display: flex; justify-content: space-between; align-items: center; margin-top: 8px; }}
        .model-img-card .mini-tag {{ padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }}
        .model-img-card .mini-tag.yes {{ background: rgba(34,197,94,0.2); color: #22c55e; }}
        .model-img-card .mini-tag.no {{ background: rgba(239,68,68,0.2); color: #ef4444; }}
        .model-img-card .correctness {{ font-size: 0.75rem; }}
        .model-img-card .correctness.right {{ color: #22c55e; }}
        .model-img-card .correctness.wrong {{ color: #ef4444; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Herman Miller AI Benchmark</h1>
            <p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · {len(benchmark_data["images"])} images · {len(benchmark_data["models_tested"])} models</p>
        </header>

        <div class="stats">
            <div class="stat"><div class="stat-value blue">{len(benchmark_data["images"])}</div><div class="stat-label">Images</div></div>
            <div class="stat"><div class="stat-value">{len(benchmark_data["models_tested"])}</div><div class="stat-label">Models</div></div>
            <div class="stat"><div class="stat-value green">{sum(1 for i in benchmark_data["images"] if i["is_actually_herman_miller"])}</div><div class="stat-label">Herman Miller</div></div>
            <div class="stat"><div class="stat-value">{sum(1 for i in benchmark_data["images"] if not i["is_actually_herman_miller"])}</div><div class="stat-label">Facebook</div></div>
        </div>

        <h2>Model Rankings</h2>
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Model</th>
                        <th>Cost</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>TP</th>
                        <th>FP</th>
                        <th>TN</th>
                        <th>FN</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody>'''

    # Sort models by accuracy
    sorted_models = []
    for name, stats in model_stats.items():
        total = stats["tp"] + stats["fp"] + stats["tn"] + stats["fn"]
        if total > 0:
            accuracy = (stats["tp"] + stats["tn"]) / total
            precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
            recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
            avg_time = stats["total_time"] / total if total > 0 else 0
            sorted_models.append((name, stats, accuracy, precision, recall, avg_time))

    sorted_models.sort(key=lambda x: x[2], reverse=True)

    for rank, (name, stats, accuracy, precision, recall, avg_time) in enumerate(sorted_models, 1):
        acc_class = "high" if accuracy >= 0.8 else "med" if accuracy >= 0.6 else "low"
        # Escape name for JS
        name_escaped = name.replace("'", "\\'")
        html += f'''
                    <tr onclick="openModelDetail('{name_escaped}')">
                        <td>{rank}</td>
                        <td><div class="model-name">{name}</div><div class="model-id">{stats["model_id"]}</div></td>
                        <td class="cost">{stats["cost"]}</td>
                        <td><div class="acc-bar"><div class="acc-fill {acc_class}" style="width:{accuracy*100}%"></div></div>{accuracy*100:.1f}%</td>
                        <td class="num">{precision*100:.0f}%</td>
                        <td class="num">{recall*100:.0f}%</td>
                        <td class="num good">{stats["tp"]}</td>
                        <td class="num bad">{stats["fp"]}</td>
                        <td class="num">{stats["tn"]}</td>
                        <td class="num bad">{stats["fn"]}</td>
                        <td class="num">{avg_time:.1f}s</td>
                    </tr>'''

    html += '''
                </tbody>
            </table>
        </div>

        <h2>Images</h2>
        <div class="filters">
            <button class="filter-btn active" onclick="filterImages('all')">All</button>
            <button class="filter-btn" onclick="filterImages('hm')">Herman Miller</button>
            <button class="filter-btn" onclick="filterImages('fb')">Facebook</button>
        </div>
        <div class="image-grid">
'''

    # Build image data for JavaScript
    images_js_data = []

    for idx, img in enumerate(benchmark_data["images"]):
        is_hm = img["is_actually_herman_miller"]
        tag_class = "hm" if is_hm else "fb"
        tag_text = "Herman Miller" if is_hm else "Facebook"
        data_type = "hm" if is_hm else "fb"
        img_url = img.get("image_url", "")
        listing_url = img.get("listing_url", "")

        # Count correct predictions (check is_premium, is_herman_miller, brand, AND model name)
        def get_predicted(analysis):
            if not analysis:
                return False
            predicted = analysis.get("is_premium", False) or analysis.get("is_herman_miller", False)
            brand = analysis.get("brand", "").lower()
            model = analysis.get("model", "").lower()
            # Check brand
            if brand in ["herman miller", "steelcase", "humanscale", "haworth"]:
                predicted = True
            # Check model names (in case brand is wrong but model is specific)
            hm_models = ["aeron", "embody", "mirra", "sayl", "cosm"]
            other_premium = ["leap", "gesture", "freedom", "liberty", "fern", "zody", "karman"]
            if any(m in model for m in hm_models + other_premium):
                predicted = True
            return predicted

        correct_count = sum(1 for r in img["results"] if r.get("analysis") and get_predicted(r["analysis"]) == is_hm)
        total_count = sum(1 for r in img["results"] if r.get("analysis"))

        html += f'''
            <div class="image-card" data-type="{data_type}" onclick="openImageModal({idx})">
                <img src="{img_url}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22280%22 height=%22200%22><rect fill=%22%23222%22 width=%22280%22 height=%22200%22/><text fill=%22%23666%22 x=%22140%22 y=%22100%22 text-anchor=%22middle%22>#{idx+1}</text></svg>'" />
                <div class="info">
                    <div class="title">{img["title"]}</div>
                    <div class="meta">{img.get("price", "N/A")}</div>
                    <span class="tag {tag_class}">{tag_text}</span>
                    <span class="score"><span>{correct_count}</span>/{total_count} correct</span>
                </div>
            </div>'''

        # Prepare data for modal
        results_data = []
        for r in img["results"]:
            if r.get("analysis"):
                predicted = get_predicted(r["analysis"])
                brand = r["analysis"].get("brand", "")
                results_data.append({
                    "model": r["model_name"],
                    "predicted": predicted,
                    "correct": predicted == is_hm,
                    "reasoning": r["analysis"].get("reasoning", "")[:300],
                    "confidence": r["analysis"].get("confidence", "?"),
                    "chair_model": r["analysis"].get("model", "Unknown"),
                    "brand": brand,
                    "time": r.get("elapsed_seconds", 0)
                })

        images_js_data.append({
            "title": img["title"],
            "price": img.get("price", "N/A"),
            "source": img.get("source", ""),
            "img_url": img_url,
            "listing_url": listing_url,
            "is_hm": is_hm,
            "results": results_data
        })

    import json as json_module
    html += f'''
        </div>
    </div>

    <!-- Image Detail Modal -->
    <div id="detailModal" class="modal">
        <span class="modal-close" onclick="closeDetailModal()">&times;</span>
        <div class="modal-content">
            <div class="modal-header">
                <img id="modalMainImg" class="modal-img" onclick="openFullscreen()" />
                <div class="modal-info">
                    <h3 id="modalTitle"></h3>
                    <div class="meta" id="modalMeta"></div>
                    <div id="modalTag"></div>
                    <a id="modalLink" class="link" target="_blank" style="display:none">View Original Listing →</a>
                </div>
            </div>
            <h2 style="margin-bottom:16px">Model Results</h2>
            <div id="modalResults" class="results-grid"></div>
        </div>
    </div>

    <!-- Fullscreen Image -->
    <div id="fullscreenModal" class="fullscreen" onclick="closeFullscreen()">
        <span class="fullscreen-close">&times;</span>
        <img id="fullscreenImg" />
    </div>

    <!-- Model Detail Modal -->
    <div id="modelModal" class="modal">
        <span class="modal-close" onclick="closeModelModal()">&times;</span>
        <div class="modal-content">
            <div class="model-detail-header">
                <h3 id="modelModalTitle"></h3>
                <div class="model-detail-stats">
                    <div class="stat"><div class="stat-val green" id="modelTP">0</div><div class="stat-lbl">True Pos</div></div>
                    <div class="stat"><div class="stat-val red" id="modelFP">0</div><div class="stat-lbl">False Pos</div></div>
                    <div class="stat"><div class="stat-val" id="modelTN">0</div><div class="stat-lbl">True Neg</div></div>
                    <div class="stat"><div class="stat-val red" id="modelFN">0</div><div class="stat-lbl">False Neg</div></div>
                </div>
            </div>
            <div class="filters" style="margin-bottom:16px">
                <button class="filter-btn active" onclick="filterModelResults('all')">All</button>
                <button class="filter-btn" onclick="filterModelResults('correct')">Correct Only</button>
                <button class="filter-btn" onclick="filterModelResults('wrong')">Wrong Only</button>
            </div>
            <div id="modelImagesGrid" class="model-images-grid"></div>
        </div>
    </div>

    <script>
        const imageData = {json_module.dumps(images_js_data)};

        function filterImages(type) {{
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            document.querySelectorAll('.image-card').forEach(card => {{
                card.style.display = (type === 'all' || card.dataset.type === type) ? 'block' : 'none';
            }});
        }}

        function openImageModal(idx) {{
            const img = imageData[idx];
            document.getElementById('modalMainImg').src = img.img_url;
            document.getElementById('modalTitle').textContent = img.title;
            document.getElementById('modalMeta').textContent = img.price + ' · ' + img.source;

            const tagDiv = document.getElementById('modalTag');
            tagDiv.innerHTML = img.is_hm
                ? '<span class="tag" style="background:rgba(34,197,94,0.15);color:#22c55e">Ground Truth: Herman Miller</span>'
                : '<span class="tag" style="background:rgba(239,68,68,0.15);color:#ef4444">Ground Truth: Not Herman Miller</span>';

            const linkEl = document.getElementById('modalLink');
            if (img.listing_url) {{
                linkEl.href = img.listing_url;
                linkEl.style.display = 'inline-block';
            }} else {{
                linkEl.style.display = 'none';
            }}

            let resultsHtml = '';
            for (const r of img.results) {{
                const cardClass = r.correct ? 'correct' : 'wrong';
                const verdictClass = r.predicted ? 'yes' : 'no';
                const verdictText = r.predicted ? 'HM' : 'Not HM';
                resultsHtml += `
                    <div class="result-card ${{cardClass}}">
                        <div class="model">${{r.model}} <span class="verdict ${{verdictClass}}">${{verdictText}}</span></div>
                        <div class="reasoning">${{r.reasoning}}</div>
                        <div class="time">${{r.chair_model}} · ${{r.confidence}} · ${{r.time.toFixed(1)}}s</div>
                    </div>`;
            }}
            document.getElementById('modalResults').innerHTML = resultsHtml;

            document.getElementById('detailModal').classList.add('open');
            document.body.style.overflow = 'hidden';
        }}

        function closeDetailModal() {{
            document.getElementById('detailModal').classList.remove('open');
            document.body.style.overflow = 'auto';
        }}

        function openFullscreen() {{
            document.getElementById('fullscreenImg').src = document.getElementById('modalMainImg').src;
            document.getElementById('fullscreenModal').classList.add('open');
        }}

        function closeFullscreen() {{
            document.getElementById('fullscreenModal').classList.remove('open');
        }}

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') {{
                closeFullscreen();
                closeDetailModal();
                closeModelModal();
            }}
        }});

        let currentModelResults = [];

        function openModelDetail(modelName) {{
            document.getElementById('modelModalTitle').textContent = modelName;

            let tp = 0, fp = 0, tn = 0, fn = 0;
            currentModelResults = [];

            for (let i = 0; i < imageData.length; i++) {{
                const img = imageData[i];
                const result = img.results.find(r => r.model === modelName);
                if (result) {{
                    const predicted = result.predicted;
                    const actual = img.is_hm;
                    const correct = predicted === actual;

                    if (actual && predicted) tp++;
                    else if (!actual && predicted) fp++;
                    else if (!actual && !predicted) tn++;
                    else if (actual && !predicted) fn++;

                    currentModelResults.push({{
                        idx: i,
                        img_url: img.img_url,
                        title: img.title,
                        price: img.price,
                        is_hm: actual,
                        predicted: predicted,
                        correct: correct,
                        reasoning: result.reasoning,
                        confidence: result.confidence
                    }});
                }}
            }}

            document.getElementById('modelTP').textContent = tp;
            document.getElementById('modelFP').textContent = fp;
            document.getElementById('modelTN').textContent = tn;
            document.getElementById('modelFN').textContent = fn;

            renderModelResults('all');
            document.getElementById('modelModal').classList.add('open');
            document.body.style.overflow = 'hidden';
        }}

        function renderModelResults(filter) {{
            let html = '';
            for (const r of currentModelResults) {{
                if (filter === 'correct' && !r.correct) continue;
                if (filter === 'wrong' && r.correct) continue;

                const cardClass = r.correct ? 'correct' : 'wrong';
                const verdictClass = r.predicted ? 'yes' : 'no';
                const verdictText = r.predicted ? 'Said HM' : 'Said Not HM';
                const correctText = r.correct ? '✓ Correct' : '✗ Wrong';
                const correctClass = r.correct ? 'right' : 'wrong';
                const truthText = r.is_hm ? 'Actually HM' : 'Actually Not HM';

                html += `
                    <div class="model-img-card ${{cardClass}}" data-correct="${{r.correct}}">
                        <img src="${{r.img_url}}" onclick="openFullscreenDirect('${{r.img_url}}')" onerror="this.style.background='#333'" />
                        <div class="details">
                            <div class="img-title">${{r.title}}</div>
                            <div class="img-meta">${{r.price}} · ${{truthText}}</div>
                            <div class="verdict-row">
                                <span class="mini-tag ${{verdictClass}}">${{verdictText}}</span>
                                <span class="correctness ${{correctClass}}">${{correctText}}</span>
                            </div>
                            <div style="margin-top:8px;font-size:0.75rem;color:#888;line-height:1.4">${{r.reasoning || 'No reasoning'}}</div>
                        </div>
                    </div>`;
            }}
            document.getElementById('modelImagesGrid').innerHTML = html;
        }}

        function filterModelResults(filter) {{
            document.querySelectorAll('#modelModal .filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            renderModelResults(filter);
        }}

        function closeModelModal() {{
            document.getElementById('modelModal').classList.remove('open');
            document.body.style.overflow = 'auto';
        }}

        function openFullscreenDirect(url) {{
            document.getElementById('fullscreenImg').src = url;
            document.getElementById('fullscreenModal').classList.add('open');
        }}
    </script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)


async def run_benchmark_mode():
    """Benchmark different AI models for accuracy on Herman Miller detection.

    Runs ALL models in PARALLEL for each image, then generates HTML report.
    """
    print("=" * 60)
    print("🏁 BENCHMARK MODE - Model Accuracy Testing")
    print(f"Started at: {datetime.now()}")
    print("=" * 60)

    # Show available models and let user choose
    print("\n📋 Available models to benchmark:\n")
    print(f"   {'#':<3} {'Model':<25} {'Cost (in/out)':<15}")
    print(f"   {'-'*3} {'-'*25} {'-'*15}")
    for i, (model_id, name, cost) in enumerate(BENCHMARK_MODELS):
        print(f"   {i+1:<3} {name:<25} {cost:<15}")

    print(f"\n   0   Run ALL models")
    print()

    # Get user input
    try:
        choice = input("Enter model number(s) to test (comma-separated, or 0 for all): ").strip()
        if choice == "0" or choice.lower() == "all":
            models_to_test = list(BENCHMARK_MODELS)
        else:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            models_to_test = [BENCHMARK_MODELS[i] for i in indices if 0 <= i < len(BENCHMARK_MODELS)]
    except (ValueError, IndexError):
        print("Invalid input, testing all models...")
        models_to_test = list(BENCHMARK_MODELS)

    print(f"\n🎯 Testing {len(models_to_test)} model(s) IN PARALLEL")

    # First, collect test images - FB chairs + Herman Millers
    print("\n📷 Collecting test images...")

    test_images = []

    # Download ALL Herman Miller images (these are KNOWN positives)
    print("   Downloading known Herman Miller images...")
    hm_count = 0
    for i, hm in enumerate(TEST_HERMAN_MILLER_IMAGES):
        try:
            resp = requests.get(hm["url"], timeout=15)
            if resp.status_code == 200:
                test_images.append({
                    "image_base64": base64.b64encode(resp.content).decode('utf-8'),
                    "image_url": hm["url"],
                    "is_actually_herman_miller": True,
                    "source": f"HM: {hm['title']}",
                    "title": hm["title"],
                    "price": hm.get("price", "Unknown"),
                })
                hm_count += 1
                print(f"      ✓ HM image {i+1}/{len(TEST_HERMAN_MILLER_IMAGES)}: {hm['title']}")
        except Exception as e:
            print(f"      ✗ Failed to download HM image {i+1}: {e}")

    # Download premium NON-Herman Miller chairs (known negatives - but quality chairs)
    print("\n   Downloading premium non-HM chairs (Steelcase, Humanscale, Haworth)...")
    premium_count = 0
    for i, chair in enumerate(TEST_OTHER_PREMIUM_CHAIRS):
        try:
            resp = requests.get(chair["url"], timeout=15)
            if resp.status_code == 200:
                test_images.append({
                    "image_base64": base64.b64encode(resp.content).decode('utf-8'),
                    "image_url": chair["url"],
                    "is_actually_herman_miller": False,
                    "source": f"Premium: {chair['actual_brand']}",
                    "title": chair["title"],
                    "price": chair.get("price", "Unknown"),
                    "actual_brand": chair["actual_brand"],
                })
                premium_count += 1
                print(f"      ✓ Premium {i+1}/{len(TEST_OTHER_PREMIUM_CHAIRS)}: {chair['actual_brand']}")
        except Exception as e:
            print(f"      ✗ Failed to download premium chair {i+1}: {e}")

    # Target 2x Facebook images compared to HM images
    fb_target = hm_count * 2
    print(f"\n   Targeting {fb_target} Facebook images (2x the {hm_count} HM images)")

    # Scrape some Facebook images (these are ASSUMED negatives - but we'll print for verification)
    print("   Scraping Facebook Marketplace for regular chairs...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=HEADLESS_MODE,
            args=["--disable-blink-features=AutomationControlled", "--disable-infobars", "--no-sandbox"]
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            locale=LOCALE,
            timezone_id=TIMEZONE,
        )

        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
        """)

        await context.add_cookies(FB_COOKIES)
        page = await context.new_page()

        await page.goto(f"https://www.facebook.com/marketplace/{MARKETPLACE_LOCATION}/search?query=office%20chair",
                       wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(3000)

        # Scroll to load more
        for _ in range(3):
            await page.keyboard.press("End")
            await page.wait_for_timeout(2000)

        # Get listing URLs
        listing_elements = await page.query_selector_all('a[href*="/marketplace/item/"]')
        listing_urls = []
        seen = set()
        for elem in listing_elements:
            href = await elem.get_attribute("href")
            if href and "/marketplace/item/" in href:
                listing_id = href.split("/marketplace/item/")[1].split("/")[0].split("?")[0]
                if listing_id not in seen:
                    seen.add(listing_id)
                    listing_urls.append(f"https://www.facebook.com/marketplace/item/{listing_id}/")

        print(f"      Found {len(listing_urls)} listings")

        # Fetch images to match 2x ratio
        fb_count = 0
        for url in listing_urls[:fb_target + 10]:
            if fb_count >= fb_target:
                break
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                await page.wait_for_timeout(1500)

                title_elem = await page.query_selector('h1')
                title = await title_elem.inner_text() if title_elem else "Unknown Chair"

                price_elem = await page.query_selector('span:has-text("$")')
                price = await price_elem.inner_text() if price_elem else "Unknown"

                img_elem = await page.query_selector('img[data-visualcompletion="media-vc-image"]')
                if not img_elem:
                    img_elem = await page.query_selector('div[role="main"] img')

                if img_elem:
                    img_src = await img_elem.get_attribute("src")
                    if img_src:
                        img_resp = requests.get(img_src, timeout=10)
                        if img_resp.status_code == 200:
                            test_images.append({
                                "image_base64": base64.b64encode(img_resp.content).decode('utf-8'),
                                "image_url": img_src,
                                "is_actually_herman_miller": False,
                                "source": f"FB: {title[:30]}",
                                "title": title,
                                "price": price,
                            })
                            fb_count += 1
                            print(f"      ✓ FB image {fb_count}/{fb_target}: {title[:40]} - {price}")

                await page.wait_for_timeout(1000)
            except Exception as e:
                continue

        await browser.close()

    print(f"\n   Total test images: {len(test_images)}")
    print(f"   - Known Herman Millers: {sum(1 for t in test_images if t['is_actually_herman_miller'])}")
    print(f"   - Premium non-HM (Steelcase/Humanscale/Haworth): {sum(1 for t in test_images if t.get('actual_brand'))}")
    print(f"   - Facebook random chairs: {sum(1 for t in test_images if not t['is_actually_herman_miller'] and not t.get('actual_brand'))}")

    # Shuffle images for fairness
    random.shuffle(test_images)

    # Prepare benchmark data structure
    benchmark_data = {
        "timestamp": datetime.now().isoformat(),
        "models_tested": [m[1] for m in models_to_test],
        "images": []
    }

    # Process images in parallel batches, each batch uses a different API key
    NUM_PARALLEL_BATCHES = len(OPENROUTER_API_KEYS)  # 3 batches in parallel
    print(f"\n{'='*60}")
    print(f"🚀 Running {len(models_to_test)} models per image")
    print(f"   {NUM_PARALLEL_BATCHES} parallel streams (one per API key)")
    print(f"   {len(test_images)} total images")
    print(f"{'='*60}")

    def process_single_image_with_key(args):
        """Process one image against all models using specified API key."""
        i, img_data, api_key = args

        # Prepare args for all models (include API key)
        model_args = [
            (model_id, model_name, cost, img_data["image_base64"], img_data["title"], img_data["price"], api_key)
            for model_id, model_name, cost in models_to_test
        ]

        # Run all models in parallel for this image
        with ThreadPoolExecutor(max_workers=len(models_to_test)) as model_executor:
            results = list(model_executor.map(analyze_single_model, model_args))

        # Count results
        correct = 0
        incorrect = 0
        errors = 0
        actual_hm = img_data["is_actually_herman_miller"]

        for r in results:
            if r["error"]:
                errors += 1
            elif r["analysis"]:
                # Check is_premium, is_herman_miller, brand, AND model name
                analysis = r["analysis"]
                predicted = analysis.get("is_premium", False) or analysis.get("is_herman_miller", False)
                brand = analysis.get("brand", "").lower()
                model = analysis.get("model", "").lower()
                if brand in ["herman miller", "steelcase", "humanscale", "haworth"]:
                    predicted = True
                hm_models = ["aeron", "embody", "mirra", "sayl", "cosm"]
                other_premium = ["leap", "gesture", "freedom", "liberty", "fern", "zody", "karman"]
                if any(m in model for m in hm_models + other_premium):
                    predicted = True
                if predicted == actual_hm:
                    correct += 1
                else:
                    incorrect += 1

        return {
            "index": i,
            "source": img_data["source"],
            "price": img_data["price"],
            "correct": correct,
            "incorrect": incorrect,
            "errors": errors,
            "img_result": {
                "image_url": img_data.get("image_url", ""),
                "image_base64": img_data["image_base64"][:100] + "...",
                "is_actually_herman_miller": actual_hm,
                "source": img_data["source"],
                "title": img_data["title"],
                "price": img_data["price"],
                "results": results
            }
        }

    # Distribute images across API keys (round-robin)
    all_tasks = []
    for i, img_data in enumerate(test_images):
        api_key = OPENROUTER_API_KEYS[i % NUM_PARALLEL_BATCHES]
        all_tasks.append((i, img_data, api_key))

    print(f"\n   Distributing {len(all_tasks)} images across {NUM_PARALLEL_BATCHES} API keys...")
    for key_idx in range(NUM_PARALLEL_BATCHES):
        count = sum(1 for t in all_tasks if t[2] == OPENROUTER_API_KEYS[key_idx])
        print(f"   Key {key_idx + 1}: {count} images")

    start_time = time.time()

    # Process ALL images in parallel (limited by thread pool size)
    # Each image uses its assigned API key
    MAX_CONCURRENT = NUM_PARALLEL_BATCHES * 3  # 3 images per key at a time = 9 concurrent
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(process_single_image_with_key, task): task for task in all_tasks}

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            completed += 1
            print(f"   [{completed}/{len(all_tasks)}] {res['source']}: ✓{res['correct']} ✗{res['incorrect']} err:{res['errors']}")
            benchmark_data["images"].append(res["img_result"])

    total_time = time.time() - start_time
    print(f"\n✅ All {len(test_images)} images processed in {total_time:.1f}s")

    # Generate outputs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save JSON
    json_path = OUTPUT_DIR / f"benchmark_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2, default=str)
    print(f"\n📁 JSON saved to: {json_path}")

    # Generate HTML report
    html_path = OUTPUT_DIR / f"benchmark_{timestamp}.html"
    generate_html_report(benchmark_data, html_path)
    print(f"🌐 HTML report saved to: {html_path}")

    # Create/update "latest" symlinks for easy access
    latest_json = OUTPUT_DIR / "benchmark_latest.json"
    latest_html = OUTPUT_DIR / "benchmark_latest.html"
    if latest_json.is_symlink() or latest_json.exists():
        latest_json.unlink()
    if latest_html.is_symlink() or latest_html.exists():
        latest_html.unlink()
    latest_json.symlink_to(json_path.name)
    latest_html.symlink_to(html_path.name)
    print(f"🔗 Updated benchmark_latest.html symlink")

    # Open in browser
    import webbrowser
    webbrowser.open(f"file://{html_path}")
    print(f"\n✅ Opening report in browser...")


async def main():
    """Main function to find Herman Miller chairs."""

    # Check for list benchmarks mode
    if args.list_benchmarks:
        list_benchmark_runs()
        return

    # Check for compare mode
    if args.compare:
        compare_benchmark_runs(args.compare[0], args.compare[1])
        return

    # Check for pure test mode
    if PURE_TEST_MODE:
        await run_pure_test_mode()
        return

    # Check for benchmark mode
    if BENCHMARK_MODE:
        await run_benchmark_mode()
        return

    print("=" * 60)
    print("Herman Miller Chair Finder")
    print(f"Started at: {datetime.now()}")
    print("=" * 60)

    herman_millers = []

    async with async_playwright() as p:
        # Launch browser with anti-detection measures
        print("\n🌐 Launching browser...")

        # Use a real Chrome installation path if available, otherwise use Playwright's
        browser = await p.chromium.launch(
            headless=HEADLESS_MODE,
            args=[
                "--disable-blink-features=AutomationControlled",  # Hide automation
                "--disable-infobars",  # No "Chrome is being controlled" bar
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-popup-blocking",
                "--no-sandbox",  # Required for running as root/in containers
            ]
        )

        # Create context with realistic settings
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            locale=LOCALE,
            timezone_id=TIMEZONE,
            geolocation={"latitude": LATITUDE, "longitude": LONGITUDE},
            permissions=["geolocation"],
            color_scheme="light",
            device_scale_factor=2,
        )

        # Remove webdriver property and other automation markers
        await context.add_init_script("""
            // Remove webdriver flag
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });

            // Remove automation-related properties
            delete navigator.__proto__.webdriver;

            // Fix plugins (headless Chrome has empty plugins)
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
                    { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
                    { name: 'Native Client', filename: 'internal-nacl-plugin' },
                ]
            });

            // Fix languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-AU', 'en-GB', 'en-US', 'en']
            });

            // Fix platform
            Object.defineProperty(navigator, 'platform', {
                get: () => 'MacIntel'
            });

            // Fix hardware concurrency (common value)
            Object.defineProperty(navigator, 'hardwareConcurrency', {
                get: () => 8
            });

            // Fix device memory
            Object.defineProperty(navigator, 'deviceMemory', {
                get: () => 8
            });

            // Remove Playwright/Puppeteer markers from window
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );

            // Make chrome object look real
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };
        """)

        # Add cookies
        await context.add_cookies(FB_COOKIES)

        page = await context.new_page()

        # Helper: human-like scrolling with random pauses, mouse movements, etc
        async def human_scroll(num_scrolls, description=""):
            for i in range(num_scrolls):
                # Random scroll method
                scroll_method = random.choices(
                    ["PageDown", "PageDown", "wheel", "arrow", "End"],
                    weights=[40, 30, 15, 10, 5]
                )[0]

                if scroll_method == "wheel":
                    # Smooth mouse wheel scroll
                    await page.mouse.wheel(0, random.randint(300, 800))
                elif scroll_method == "arrow":
                    # Arrow key scrolling (multiple presses)
                    for _ in range(random.randint(3, 8)):
                        await page.keyboard.press("ArrowDown")
                        await page.wait_for_timeout(random.uniform(50, 150))
                else:
                    await page.keyboard.press(scroll_method)

                # Random delay
                base_delay = random.uniform(800, 2500)

                # Sometimes pause longer like reading something
                if random.random() < 0.15:
                    base_delay += random.uniform(2000, 6000)

                # Sometimes move mouse randomly (like looking at something)
                if random.random() < 0.25:
                    viewport = page.viewport_size
                    if viewport:
                        x = random.randint(100, viewport['width'] - 100)
                        y = random.randint(100, viewport['height'] - 100)
                        await page.mouse.move(x, y, steps=random.randint(5, 15))
                        await page.wait_for_timeout(random.uniform(200, 800))

                # Sometimes hover over a listing (but don't click)
                if random.random() < 0.1:
                    try:
                        items = await page.query_selector_all('a[href*="/marketplace/item/"]')
                        if items:
                            item = random.choice(items[:10])  # Pick from visible ones
                            await item.hover()
                            await page.wait_for_timeout(random.uniform(500, 1500))
                    except:
                        pass

                await page.wait_for_timeout(base_delay)

                if description and (i + 1) % 5 == 0:
                    print(f"   {description}: {i+1}/{num_scrolls} scrolls...")

        # Helper: browse a search without analyzing (decoy behavior)
        async def browse_decoy(query):
            print(f"\n🎭 Browsing '{query}' (decoy)...")
            await page.goto(
                f"https://www.facebook.com/marketplace/{MARKETPLACE_LOCATION}/search?query={query.replace(' ', '%20')}",
                wait_until="domcontentloaded", timeout=60000
            )
            await page.wait_for_timeout(random.uniform(2000, 4000))

            # Scroll around
            await human_scroll(random.randint(3, 10), f"Browsing {query}")

            # Maybe click on a random listing
            if random.random() < 0.4:
                try:
                    items = await page.query_selector_all('a[href*="/marketplace/item/"]')
                    if items:
                        item = random.choice(items[:15])
                        print(f"   Clicking random listing...")
                        await item.click()
                        await page.wait_for_timeout(random.uniform(3000, 8000))

                        # Scroll a bit on the listing page
                        for _ in range(random.randint(1, 3)):
                            await page.keyboard.press("PageDown")
                            await page.wait_for_timeout(random.uniform(500, 1500))

                        # Go back
                        await page.go_back()
                        await page.wait_for_timeout(random.uniform(1000, 2000))
                except:
                    pass

            print(f"   Done browsing {query}")

        # Build search queue - mix of real searches and decoys
        search_queue = []

        # Add chair searches (the ones we actually analyze)
        chair_search = random.choice(CHAIR_SEARCHES)
        search_queue.append(chair_search)

        # In prod, add decoy searches before and after
        if not DEV_MODE:
            # Maybe start with a decoy
            if random.random() < 0.5:
                search_queue.insert(0, {"query": random.choice(DECOY_SEARCHES), "analyze": False})

            # Maybe add another chair search
            if random.random() < 0.3:
                other_chair = random.choice([s for s in CHAIR_SEARCHES if s != chair_search])
                search_queue.append(other_chair)

            # Maybe end with a decoy
            if random.random() < 0.4:
                search_queue.append({"query": random.choice(DECOY_SEARCHES), "analyze": False})

        print(f"\n📋 Search plan: {[s['query'] for s in search_queue]}")

        # Process each search
        listings = []
        seen_ids = set()

        for search in search_queue:
            query = search["query"]
            should_analyze = search["analyze"]

            if not should_analyze:
                # Just browse, don't collect listings
                await browse_decoy(query)
                continue

            # This is a real chair search - collect listings
            print(f"\n🔍 Searching for '{query}'...")
            await page.goto(
                f"https://www.facebook.com/marketplace/{MARKETPLACE_LOCATION}/search?query={query.replace(' ', '%20')}",
                wait_until="domcontentloaded", timeout=60000
            )
            await page.wait_for_timeout(random.uniform(2500, 5000))

            # Scroll to load lots of listings
            scroll_count = SCROLL_COUNT if not DEV_MODE else 3
            print(f"📜 Scrolling ({scroll_count} times)...")
            await human_scroll(scroll_count, "Loading chairs")

            # Collect listings from this search
            listing_elements = await page.query_selector_all('a[href*="/marketplace/item/"]')

            for elem in listing_elements:
                try:
                    href = await elem.get_attribute("href")
                    if href and "/marketplace/item/" in href:
                        parts = href.split("/marketplace/item/")
                        if len(parts) > 1:
                            listing_id = parts[1].split("/")[0].split("?")[0]
                            if listing_id and listing_id not in seen_ids:
                                seen_ids.add(listing_id)
                                listings.append({
                                    "id": listing_id,
                                    "url": f"https://www.facebook.com/marketplace/item/{listing_id}/",
                                    "element": elem,
                                    "source_query": query
                                })
                except:
                    continue

            print(f"   Found {len(listings)} total unique listings so far")

            # Brief pause between searches
            if search != search_queue[-1]:
                await page.wait_for_timeout(random.uniform(2000, 5000))

        print(f"\n✅ Total: {len(listings)} unique chair listings to analyze")

        # Filter out already-known listings
        new_listings = []
        skipped = 0
        for listing in listings:
            if is_listing_known(listing["id"]):
                skipped += 1
            else:
                new_listings.append(listing)

        if skipped > 0:
            print(f"⏭️  Skipped {skipped} already-processed listings")
        print(f"📋 {len(new_listings)} new listings to check")

        # Process each NEW listing
        for i, listing in enumerate(new_listings[:LISTING_COUNT]):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{min(len(new_listings), LISTING_COUNT)}] Checking listing {listing['id']}")
            print(f"{'='*60}")

            # Determine if this is a test run (1 in TEST_MODE_CHANCE)
            # In test mode, we swap in a REAL Herman Miller image so the AI genuinely
            # recognizes it - this simulates finding an actual Herman Miller listed as
            # a generic "office chair" (which is exactly what we're hunting for!)
            is_test_mode = random.randint(1, TEST_MODE_CHANCE) == 1
            test_chair = None
            if is_test_mode:
                test_chair = random.choice(TEST_HERMAN_MILLER_IMAGES)
                print(f"  🧪 TEST MODE ACTIVATED (1/{TEST_MODE_CHANCE} chance)")
                print(f"     Swapping image with real {test_chair['model']} photo")
                print(f"     (Simulating: someone listed a {test_chair['model']} as 'office chair')")

            try:
                # Navigate to listing page
                await page.goto(listing["url"], wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(2000)

                # Get the title
                title_elem = await page.query_selector('h1')
                title = await title_elem.inner_text() if title_elem else "Unknown"
                listing["title"] = title

                # Get the price
                price_elem = await page.query_selector('span:has-text("$")')
                if price_elem:
                    price_text = await price_elem.inner_text()
                    listing["price"] = price_text
                else:
                    listing["price"] = "Unknown"

                if VERBOSE_LOGGING:
                    print(f"\n  📋 SCRAPED DATA:")
                    print(f"     Title: {title}")
                    print(f"     Price: {listing['price']}")
                    print(f"     URL: {listing['url']}")

                # Find the main image
                img_elem = await page.query_selector('img[data-visualcompletion="media-vc-image"]')
                if not img_elem:
                    img_elem = await page.query_selector('div[role="main"] img')

                if img_elem:
                    img_src = await img_elem.get_attribute("src")

                    if img_src:
                        if VERBOSE_LOGGING:
                            print(f"     Image URL: {img_src[:80]}...")

                        # In test mode, swap the image URL with a real Herman Miller
                        if is_test_mode and test_chair:
                            img_src = test_chair["url"]
                            print(f"\n  🧪 SWAPPED IMAGE URL to: {img_src[:60]}...")

                        print(f"\n  📷 Downloading image...")
                        try:
                            img_response = requests.get(img_src, timeout=10)
                            if img_response.status_code == 200:
                                image_base64 = base64.b64encode(img_response.content).decode('utf-8')

                                if VERBOSE_LOGGING:
                                    print(f"     Image size: {len(img_response.content) / 1024:.1f} KB")

                                print(f"\n  🤖 Sending to Claude Opus 4.5...")
                                analysis = analyze_image_with_claude(image_base64)

                                if analysis:
                                    model = analysis.get('model', 'Unknown')
                                    is_hm = analysis.get('is_herman_miller', False)
                                    confidence = analysis.get('confidence', 'Unknown')
                                    reasoning = analysis.get('reasoning', 'N/A')

                                    print(f"\n  📊 AI ANALYSIS:")
                                    print(f"     Model: {model}")
                                    print(f"     Reasoning: {reasoning}")
                                    print(f"     Confidence: {confidence}")
                                    print(f"     Is Herman Miller: {is_hm}")

                                    if is_hm:
                                        # Calculate deal score
                                        deal_score, deal_label, retail_price = calculate_deal_score(listing['price'], model)

                                        print(f"\n  🎉🎉🎉 HERMAN MILLER FOUND: {model} 🎉🎉🎉")
                                        if deal_score is not None:
                                            print(f"  💰 Deal Score: {deal_score}/10 {deal_label}")
                                            print(f"     Retail: ${retail_price} → Listed: {listing['price']}")
                                        if is_test_mode:
                                            print(f"  (This is a TEST - not a real find)")

                                        # Save to database (skip if test mode)
                                        if not is_test_mode:
                                            save_listing_to_db(
                                                listing_id=listing['id'],
                                                url=listing['url'],
                                                title=listing.get('title', 'Unknown'),
                                                price=listing.get('price', 'Unknown'),
                                                model=model,
                                                confidence=confidence,
                                                reasoning=reasoning,
                                                deal_score=deal_score,
                                                deal_label=deal_label
                                            )
                                            print(f"  💾 Saved to database")

                                        save_herman_miller_listing(listing, analysis, image_base64)
                                        herman_millers.append({
                                            **listing,
                                            **analysis,
                                            "deal_score": deal_score,
                                            "deal_label": deal_label,
                                            "retail_price": retail_price,
                                            "is_test": is_test_mode
                                        })
                                    else:
                                        print(f"\n  ❌ Not a Herman Miller")
                                else:
                                    print(f"\n  ❌ Analysis failed - no response from AI")
                        except Exception as e:
                            print(f"\n  ❌ Failed to download image: {e}")
                else:
                    print(f"\n  ⚠️ No image found on this listing")

            except Exception as e:
                print(f"\n  ❌ Error processing listing: {e}")
                continue

            # Rate limiting - random delay to seem more human
            wait_time = random.uniform(LISTING_DELAY_MIN, LISTING_DELAY_MAX)

            # In prod, occasionally take a longer break
            if not DEV_MODE and random.random() < 0.15:
                wait_time += random.uniform(5000, 15000)
                print(f"\n  ⏳ Taking a break... {wait_time/1000:.1f}s")
            else:
                print(f"\n  ⏳ Waiting {wait_time/1000:.1f}s before next listing...")

            await page.wait_for_timeout(wait_time)

        await browser.close()

    # Send email alert (only for non-test finds)
    real_finds = [h for h in herman_millers if not h.get('is_test')]
    if real_finds:
        print(f"\n📧 Sending email alert for {len(real_finds)} Herman Miller(s)...")
        send_email_alert(real_finds)

        # Mark as alerted in database
        for h in real_finds:
            mark_listing_alerted(h['id'])

        # Print summary with deal scores
        fumbles = [h for h in real_finds if h.get('deal_score', 0) >= 8]
        if fumbles:
            print(f"\n🔥 FUMBLE ALERT: {len(fumbles)} incredible deal(s) found!")
            for f in fumbles:
                print(f"   - {f.get('model', '')} @ {f.get('price', '?')} ({f.get('deal_label', '')})")
    else:
        print("\n📧 No Herman Miller chairs found, skipping email")

    # Show database stats
    stats = get_listing_stats()
    print("\n" + "=" * 60)
    print(f"Scan complete! Found {len(herman_millers)} Herman Miller(s) this run")
    print(f"📊 Database: {stats['total']} total finds, {stats['alerted']} alerted")
    if stats['by_model']:
        print(f"   By model: {', '.join(f'{m}: {c}' for m, c in stats['by_model'])}")
    print("=" * 60)


def acquire_lock():
    """Acquire exclusive lock to ensure only one browser instance runs."""
    lock_fd = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
        return lock_fd
    except BlockingIOError:
        lock_fd.close()
        return None


def release_lock(lock_fd):
    """Release the lock file."""
    if lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        try:
            LOCK_FILE.unlink()
        except:
            pass


def is_within_active_hours():
    """Check if current local time is within active hours (9am-2am)."""
    local_now = datetime.now(LOCAL_TZ)
    hour = local_now.hour
    # Active hours: 9am (9) to 2am (2 next day)
    # This means: 9-23 is OK, 0-1 is OK (early morning), 2-8 is NOT OK
    return hour >= SCHEDULER_START_HOUR or hour < (SCHEDULER_END_HOUR - 24)


def get_next_run_delay():
    """Calculate delay until next run. Runs 12x/day during active hours."""
    # Active window: 9am to 2am = 17 hours
    # 12 runs = roughly every 85 minutes (with some randomness)
    active_hours = 17
    runs_per_day = SCHEDULER_RUNS_PER_DAY
    avg_interval_minutes = (active_hours * 60) / runs_per_day  # ~85 min

    # Add randomness: +/- 20 minutes
    jitter = random.uniform(-20, 20)
    interval = avg_interval_minutes + jitter

    return max(30, interval) * 60  # Return seconds, minimum 30 minutes


def run_scheduler():
    """Run the scheduler loop - 12x/day during waking hours."""
    print("=" * 60)
    print("🕐 Herman Miller Finder - Scheduler Mode")
    print(f"   Runs: {SCHEDULER_RUNS_PER_DAY}x per day")
    print(f"   Active hours: {SCHEDULER_START_HOUR}:00 - {SCHEDULER_END_HOUR - 24}:00 ({TIMEZONE})")
    print("=" * 60)

    run_count = 0

    while True:
        local_now = datetime.now(LOCAL_TZ)

        if not is_within_active_hours():
            # Calculate time until 9am local
            if local_now.hour >= (SCHEDULER_END_HOUR - 24) and local_now.hour < SCHEDULER_START_HOUR:
                hours_until_active = SCHEDULER_START_HOUR - local_now.hour
                sleep_seconds = hours_until_active * 3600 - local_now.minute * 60
                print(f"\n😴 Outside active hours ({local_now.strftime('%H:%M')} {TIMEZONE})")
                print(f"   Sleeping until 9:00 AM ({hours_until_active:.1f}h)...")
                time.sleep(max(60, sleep_seconds))
                continue

        # Try to acquire lock
        lock_fd = acquire_lock()
        if not lock_fd:
            print(f"\n⚠️  Another instance is running, waiting 5 minutes...")
            time.sleep(300)
            continue

        try:
            run_count += 1
            print(f"\n{'=' * 60}")
            print(f"🔍 Starting scan #{run_count} at {local_now.strftime('%Y-%m-%d %H:%M:%S')} {TIMEZONE}")
            print(f"{'=' * 60}")

            # Run the main scan
            asyncio.run(main())

        except Exception as e:
            print(f"\n❌ Error during scan: {e}")
        finally:
            release_lock(lock_fd)

        # Calculate next run time
        delay = get_next_run_delay()
        next_run = datetime.now(LOCAL_TZ).timestamp() + delay
        next_run_time = datetime.fromtimestamp(next_run, LOCAL_TZ)

        print(f"\n⏰ Next scan at {next_run_time.strftime('%H:%M:%S')} {TIMEZONE}")
        print(f"   (sleeping {delay/60:.0f} minutes)")
        time.sleep(delay)


if __name__ == "__main__":
    if args.scheduler:
        run_scheduler()
    else:
        # Single run mode - still use lock to prevent concurrent runs
        lock_fd = acquire_lock()
        if not lock_fd:
            print("❌ Another instance is already running. Exiting.")
            sys.exit(1)
        try:
            asyncio.run(main())
        finally:
            release_lock(lock_fd)
