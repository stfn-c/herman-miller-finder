"""Benchmark mode: model accuracy testing, HTML report generation, run comparison."""

import json
import time
import base64
import random
import requests
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from playwright.async_api import async_playwright
import src.config as config
from src.config import (
    HEADLESS_MODE,
    LOCALE,
    TIMEZONE,
    FB_COOKIES,
    OPENROUTER_API_KEYS,
    OUTPUT_DIR,
    VERBOSE_LOGGING,
)
from src.analyzer import analyze_image_with_model, analyze_single_model
from src.test_mode import TEST_HERMAN_MILLER_IMAGES, TEST_OTHER_PREMIUM_CHAIRS

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


def list_benchmark_runs():
    """List all previous benchmark runs with summary stats."""
    benchmark_files = sorted(OUTPUT_DIR.glob("benchmark_*.json"), reverse=True)

    if not benchmark_files:
        print("No benchmark runs found.")
        return

    print("\n" + "=" * 80)
    print("\U0001f4ca Previous Benchmark Runs")
    print("=" * 80)
    print(
        f"{'Timestamp':<20} {'Models':<8} {'Images':<8} {'Best Model':<25} {'Accuracy':<10}"
    )
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
                        predicted = analysis.get("is_premium", False) or analysis.get(
                            "is_herman_miller", False
                        )
                        brand = analysis.get("brand", "").lower()
                        model = analysis.get("model", "").lower()
                        if brand in [
                            "herman miller",
                            "steelcase",
                            "humanscale",
                            "haworth",
                        ]:
                            predicted = True
                        hm_models = ["aeron", "embody", "mirra", "sayl", "cosm"]
                        other_premium = [
                            "leap",
                            "gesture",
                            "freedom",
                            "liberty",
                            "fern",
                            "zody",
                            "karman",
                        ]
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

            print(
                f"{timestamp:<20} {num_models:<8} {num_images:<8} {best_model:<25} {best_acc * 100:.1f}%"
            )
        except Exception as e:
            print(f"{f.stem:<20} Error reading: {e}")

    print("-" * 80)
    print(f"Total runs: {len(benchmark_files)}")
    print(f"Latest: {benchmark_files[0].stem if benchmark_files else 'N/A'}")
    print("\nTo view a report: open found_chairs/benchmark_<timestamp>.html")
    print(
        "To compare runs: python find_herman_miller.py --compare <timestamp1> <timestamp2>"
    )


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
    print("\U0001f4ca Benchmark Comparison")
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
                    predicted = analysis.get("is_premium", False) or analysis.get(
                        "is_herman_miller", False
                    )
                    brand = analysis.get("brand", "").lower()
                    model = analysis.get("model", "").lower()
                    if brand in ["herman miller", "steelcase", "humanscale", "haworth"]:
                        predicted = True
                    hm_models = ["aeron", "embody", "mirra", "sayl", "cosm"]
                    other_premium = [
                        "leap",
                        "gesture",
                        "freedom",
                        "liberty",
                        "fern",
                        "zody",
                        "karman",
                    ]
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

    print(
        f"{'Model':<25} {'Run1 Acc':<12} {'Run2 Acc':<12} {'\u0394':<8} {'Run1 FP':<10} {'Run2 FP':<10}"
    )
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
        delta_color = (
            "\U0001f4c8" if delta > 0 else "\U0001f4c9" if delta < 0 else "\u27a1\ufe0f"
        )

        print(
            f"{model:<25} {acc1:>6.1f}%      {acc2:>6.1f}%      {delta_color}{delta_str:<6} {s1['fp']:<10} {s2['fp']:<10}"
        )

    print("-" * 90)


def generate_html_report(benchmark_data, output_path):
    """Generate a beautiful HTML report from benchmark data."""

    # Calculate summary stats per model
    model_stats = {}
    for model_id, model_name, cost in BENCHMARK_MODELS:
        model_stats[model_name] = {
            "model_id": model_id,
            "cost": cost,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "errors": 0,
            "total_time": 0,
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
                predicted = analysis.get("is_premium", False) or analysis.get(
                    "is_herman_miller", False
                )

                # Also check brand and model name (in case boolean is wrong but identification is right)
                brand = analysis.get("brand", "").lower()
                model = analysis.get("model", "").lower()
                if brand in ["herman miller", "steelcase", "humanscale", "haworth"]:
                    predicted = True
                hm_models = ["aeron", "embody", "mirra", "sayl", "cosm"]
                other_premium = [
                    "leap",
                    "gesture",
                    "freedom",
                    "liberty",
                    "fern",
                    "zody",
                    "karman",
                ]
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

    html = f"""<!DOCTYPE html>
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
            <p class="subtitle">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} \u00b7 {len(benchmark_data["images"])} images \u00b7 {len(benchmark_data["models_tested"])} models</p>
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
                <tbody>"""

    # Sort models by accuracy
    sorted_models = []
    for name, stats in model_stats.items():
        total = stats["tp"] + stats["fp"] + stats["tn"] + stats["fn"]
        if total > 0:
            accuracy = (stats["tp"] + stats["tn"]) / total
            precision = (
                stats["tp"] / (stats["tp"] + stats["fp"])
                if (stats["tp"] + stats["fp"]) > 0
                else 0
            )
            recall = (
                stats["tp"] / (stats["tp"] + stats["fn"])
                if (stats["tp"] + stats["fn"]) > 0
                else 0
            )
            avg_time = stats["total_time"] / total if total > 0 else 0
            sorted_models.append((name, stats, accuracy, precision, recall, avg_time))

    sorted_models.sort(key=lambda x: x[2], reverse=True)

    for rank, (name, stats, accuracy, precision, recall, avg_time) in enumerate(
        sorted_models, 1
    ):
        acc_class = "high" if accuracy >= 0.8 else "med" if accuracy >= 0.6 else "low"
        # Escape name for JS
        name_escaped = name.replace("'", "\\'")
        html += f"""
                    <tr onclick="openModelDetail('{name_escaped}')">
                        <td>{rank}</td>
                        <td><div class="model-name">{name}</div><div class="model-id">{stats["model_id"]}</div></td>
                        <td class="cost">{stats["cost"]}</td>
                        <td><div class="acc-bar"><div class="acc-fill {acc_class}" style="width:{accuracy * 100}%"></div></div>{accuracy * 100:.1f}%</td>
                        <td class="num">{precision * 100:.0f}%</td>
                        <td class="num">{recall * 100:.0f}%</td>
                        <td class="num good">{stats["tp"]}</td>
                        <td class="num bad">{stats["fp"]}</td>
                        <td class="num">{stats["tn"]}</td>
                        <td class="num bad">{stats["fn"]}</td>
                        <td class="num">{avg_time:.1f}s</td>
                    </tr>"""

    html += """
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
"""

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
            predicted = analysis.get("is_premium", False) or analysis.get(
                "is_herman_miller", False
            )
            brand = analysis.get("brand", "").lower()
            model = analysis.get("model", "").lower()
            # Check brand
            if brand in ["herman miller", "steelcase", "humanscale", "haworth"]:
                predicted = True
            # Check model names (in case brand is wrong but model is specific)
            hm_models = ["aeron", "embody", "mirra", "sayl", "cosm"]
            other_premium = [
                "leap",
                "gesture",
                "freedom",
                "liberty",
                "fern",
                "zody",
                "karman",
            ]
            if any(m in model for m in hm_models + other_premium):
                predicted = True
            return predicted

        correct_count = sum(
            1
            for r in img["results"]
            if r.get("analysis") and get_predicted(r["analysis"]) == is_hm
        )
        total_count = sum(1 for r in img["results"] if r.get("analysis"))

        html += f'''
            <div class="image-card" data-type="{data_type}" onclick="openImageModal({idx})">
                <img src="{img_url}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22280%22 height=%22200%22><rect fill=%22%23222%22 width=%22280%22 height=%22200%22/><text fill=%22%23666%22 x=%22140%22 y=%22100%22 text-anchor=%22middle%22>#{idx + 1}</text></svg>'" />
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
                results_data.append(
                    {
                        "model": r["model_name"],
                        "predicted": predicted,
                        "correct": predicted == is_hm,
                        "reasoning": r["analysis"].get("reasoning", "")[:300],
                        "confidence": r["analysis"].get("confidence", "?"),
                        "chair_model": r["analysis"].get("model", "Unknown"),
                        "brand": brand,
                        "time": r.get("elapsed_seconds", 0),
                    }
                )

        images_js_data.append(
            {
                "title": img["title"],
                "price": img.get("price", "N/A"),
                "source": img.get("source", ""),
                "img_url": img_url,
                "listing_url": listing_url,
                "is_hm": is_hm,
                "results": results_data,
            }
        )

    import json as json_module

    html += f"""
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
                    <a id="modalLink" class="link" target="_blank" style="display:none">View Original Listing \u2192</a>
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
            document.getElementById('modalMeta').textContent = img.price + ' \u00b7 ' + img.source;

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
                        <div class="time">${{r.chair_model}} \u00b7 ${{r.confidence}} \u00b7 ${{r.time.toFixed(1)}}s</div>
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
                const correctText = r.correct ? '\u2713 Correct' : '\u2717 Wrong';
                const correctClass = r.correct ? 'right' : 'wrong';
                const truthText = r.is_hm ? 'Actually HM' : 'Actually Not HM';

                html += `
                    <div class="model-img-card ${{cardClass}}" data-correct="${{r.correct}}">
                        <img src="${{r.img_url}}" onclick="openFullscreenDirect('${{r.img_url}}')" onerror="this.style.background='#333'" />
                        <div class="details">
                            <div class="img-title">${{r.title}}</div>
                            <div class="img-meta">${{r.price}} \u00b7 ${{truthText}}</div>
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
</html>"""

    with open(output_path, "w") as f:
        f.write(html)


async def run_benchmark_mode():
    """Benchmark different AI models for accuracy on Herman Miller detection.

    Runs ALL models in PARALLEL for each image, then generates HTML report.
    """
    print("=" * 60)
    print("\U0001f3c1 BENCHMARK MODE - Model Accuracy Testing")
    print(f"Started at: {datetime.now()}")
    print("=" * 60)

    # Show available models and let user choose
    print("\n\U0001f4cb Available models to benchmark:\n")
    print(f"   {'#':<3} {'Model':<25} {'Cost (in/out)':<15}")
    print(f"   {'-' * 3} {'-' * 25} {'-' * 15}")
    for i, (model_id, name, cost) in enumerate(BENCHMARK_MODELS):
        print(f"   {i + 1:<3} {name:<25} {cost:<15}")

    print(f"\n   0   Run ALL models")
    print()

    # Get user input
    try:
        choice = input(
            "Enter model number(s) to test (comma-separated, or 0 for all): "
        ).strip()
        if choice == "0" or choice.lower() == "all":
            models_to_test = list(BENCHMARK_MODELS)
        else:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            models_to_test = [
                BENCHMARK_MODELS[i] for i in indices if 0 <= i < len(BENCHMARK_MODELS)
            ]
    except (ValueError, IndexError):
        print("Invalid input, testing all models...")
        models_to_test = list(BENCHMARK_MODELS)

    print(f"\n\U0001f3af Testing {len(models_to_test)} model(s) IN PARALLEL")

    # First, collect test images - FB chairs + Herman Millers
    print("\n\U0001f4f7 Collecting test images...")

    test_images = []

    # Download ALL Herman Miller images (these are KNOWN positives)
    print("   Downloading known Herman Miller images...")
    hm_count = 0
    for i, hm in enumerate(TEST_HERMAN_MILLER_IMAGES):
        try:
            resp = requests.get(hm["url"], timeout=15)
            if resp.status_code == 200:
                test_images.append(
                    {
                        "image_base64": base64.b64encode(resp.content).decode("utf-8"),
                        "image_url": hm["url"],
                        "is_actually_herman_miller": True,
                        "source": f"HM: {hm['title']}",
                        "title": hm["title"],
                        "price": hm.get("price", "Unknown"),
                    }
                )
                hm_count += 1
                print(
                    f"      \u2713 HM image {i + 1}/{len(TEST_HERMAN_MILLER_IMAGES)}: {hm['title']}"
                )
        except Exception as e:
            print(f"      \u2717 Failed to download HM image {i + 1}: {e}")

    # Download premium NON-Herman Miller chairs (known negatives - but quality chairs)
    print("\n   Downloading premium non-HM chairs (Steelcase, Humanscale, Haworth)...")
    premium_count = 0
    for i, chair in enumerate(TEST_OTHER_PREMIUM_CHAIRS):
        try:
            resp = requests.get(chair["url"], timeout=15)
            if resp.status_code == 200:
                test_images.append(
                    {
                        "image_base64": base64.b64encode(resp.content).decode("utf-8"),
                        "image_url": chair["url"],
                        "is_actually_herman_miller": False,
                        "source": f"Premium: {chair['actual_brand']}",
                        "title": chair["title"],
                        "price": chair.get("price", "Unknown"),
                        "actual_brand": chair["actual_brand"],
                    }
                )
                premium_count += 1
                print(
                    f"      \u2713 Premium {i + 1}/{len(TEST_OTHER_PREMIUM_CHAIRS)}: {chair['actual_brand']}"
                )
        except Exception as e:
            print(f"      \u2717 Failed to download premium chair {i + 1}: {e}")

    # Target 2x Facebook images compared to HM images
    fb_target = hm_count * 2
    print(f"\n   Targeting {fb_target} Facebook images (2x the {hm_count} HM images)")

    # Scrape some Facebook images (these are ASSUMED negatives - but we'll print for verification)
    print("   Scraping Facebook Marketplace for regular chairs...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=HEADLESS_MODE,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--no-sandbox",
            ],
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

        await page.goto(
            f"https://www.facebook.com/marketplace/{config.CURRENT_CITY}/search?query=office%20chair",
            wait_until="domcontentloaded",
            timeout=60000,
        )
        await page.wait_for_timeout(3000)

        # Scroll to load more
        for _ in range(3):
            await page.keyboard.press("End")
            await page.wait_for_timeout(2000)

        # Get listing URLs
        listing_elements = await page.query_selector_all(
            'a[href*="/marketplace/item/"]'
        )
        listing_urls = []
        seen = set()
        for elem in listing_elements:
            href = await elem.get_attribute("href")
            if href and "/marketplace/item/" in href:
                listing_id = (
                    href.split("/marketplace/item/")[1].split("/")[0].split("?")[0]
                )
                if listing_id not in seen:
                    seen.add(listing_id)
                    listing_urls.append(
                        f"https://www.facebook.com/marketplace/item/{listing_id}/"
                    )

        print(f"      Found {len(listing_urls)} listings")

        # Fetch images to match 2x ratio
        fb_count = 0
        for url in listing_urls[: fb_target + 10]:
            if fb_count >= fb_target:
                break
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                await page.wait_for_timeout(1500)

                title_elem = await page.query_selector("h1")
                title = await title_elem.inner_text() if title_elem else "Unknown Chair"

                price_elem = await page.query_selector('span:has-text("$")')
                price = await price_elem.inner_text() if price_elem else "Unknown"

                img_elem = await page.query_selector(
                    'img[data-visualcompletion="media-vc-image"]'
                )
                if not img_elem:
                    img_elem = await page.query_selector('div[role="main"] img')

                if img_elem:
                    img_src = await img_elem.get_attribute("src")
                    if img_src:
                        img_resp = requests.get(img_src, timeout=10)
                        if img_resp.status_code == 200:
                            test_images.append(
                                {
                                    "image_base64": base64.b64encode(
                                        img_resp.content
                                    ).decode("utf-8"),
                                    "image_url": img_src,
                                    "is_actually_herman_miller": False,
                                    "source": f"FB: {title[:30]}",
                                    "title": title,
                                    "price": price,
                                }
                            )
                            fb_count += 1
                            print(
                                f"      \u2713 FB image {fb_count}/{fb_target}: {title[:40]} - {price}"
                            )

                await page.wait_for_timeout(1000)
            except Exception as e:
                continue

        await browser.close()

    print(f"\n   Total test images: {len(test_images)}")
    print(
        f"   - Known Herman Millers: {sum(1 for t in test_images if t['is_actually_herman_miller'])}"
    )
    print(
        f"   - Premium non-HM (Steelcase/Humanscale/Haworth): {sum(1 for t in test_images if t.get('actual_brand'))}"
    )
    print(
        f"   - Facebook random chairs: {sum(1 for t in test_images if not t['is_actually_herman_miller'] and not t.get('actual_brand'))}"
    )

    # Shuffle images for fairness
    random.shuffle(test_images)

    # Prepare benchmark data structure
    benchmark_data = {
        "timestamp": datetime.now().isoformat(),
        "models_tested": [m[1] for m in models_to_test],
        "images": [],
    }

    # Process images in parallel batches, each batch uses a different API key
    NUM_PARALLEL_BATCHES = len(OPENROUTER_API_KEYS)  # 3 batches in parallel
    print(f"\n{'=' * 60}")
    print(f"\U0001f680 Running {len(models_to_test)} models per image")
    print(f"   {NUM_PARALLEL_BATCHES} parallel streams (one per API key)")
    print(f"   {len(test_images)} total images")
    print(f"{'=' * 60}")

    def process_single_image_with_key(args):
        """Process one image against all models using specified API key."""
        i, img_data, api_key = args

        # Prepare args for all models (include API key)
        model_args = [
            (
                model_id,
                model_name,
                cost,
                img_data["image_base64"],
                img_data["title"],
                img_data["price"],
                api_key,
            )
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
                predicted = analysis.get("is_premium", False) or analysis.get(
                    "is_herman_miller", False
                )
                brand = analysis.get("brand", "").lower()
                model = analysis.get("model", "").lower()
                if brand in ["herman miller", "steelcase", "humanscale", "haworth"]:
                    predicted = True
                hm_models = ["aeron", "embody", "mirra", "sayl", "cosm"]
                other_premium = [
                    "leap",
                    "gesture",
                    "freedom",
                    "liberty",
                    "fern",
                    "zody",
                    "karman",
                ]
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
                "results": results,
            },
        }

    # Distribute images across API keys (round-robin)
    all_tasks = []
    for i, img_data in enumerate(test_images):
        api_key = OPENROUTER_API_KEYS[i % NUM_PARALLEL_BATCHES]
        all_tasks.append((i, img_data, api_key))

    print(
        f"\n   Distributing {len(all_tasks)} images across {NUM_PARALLEL_BATCHES} API keys..."
    )
    for key_idx in range(NUM_PARALLEL_BATCHES):
        count = sum(1 for t in all_tasks if t[2] == OPENROUTER_API_KEYS[key_idx])
        print(f"   Key {key_idx + 1}: {count} images")

    start_time = time.time()

    # Process ALL images in parallel (limited by thread pool size)
    # Each image uses its assigned API key
    MAX_CONCURRENT = (
        NUM_PARALLEL_BATCHES * 3
    )  # 3 images per key at a time = 9 concurrent
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {
            executor.submit(process_single_image_with_key, task): task
            for task in all_tasks
        }

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            completed += 1
            print(
                f"   [{completed}/{len(all_tasks)}] {res['source']}: \u2713{res['correct']} \u2717{res['incorrect']} err:{res['errors']}"
            )
            benchmark_data["images"].append(res["img_result"])

    total_time = time.time() - start_time
    print(f"\n\u2705 All {len(test_images)} images processed in {total_time:.1f}s")

    # Generate outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = OUTPUT_DIR / f"benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(benchmark_data, f, indent=2, default=str)
    print(f"\n\U0001f4c1 JSON saved to: {json_path}")

    # Generate HTML report
    html_path = OUTPUT_DIR / f"benchmark_{timestamp}.html"
    generate_html_report(benchmark_data, html_path)
    print(f"\U0001f310 HTML report saved to: {html_path}")

    # Create/update "latest" symlinks for easy access
    latest_json = OUTPUT_DIR / "benchmark_latest.json"
    latest_html = OUTPUT_DIR / "benchmark_latest.html"
    if latest_json.is_symlink() or latest_json.exists():
        latest_json.unlink()
    if latest_html.is_symlink() or latest_html.exists():
        latest_html.unlink()
    latest_json.symlink_to(json_path.name)
    latest_html.symlink_to(html_path.name)
    print(f"\U0001f517 Updated benchmark_latest.html symlink")

    # Open in browser
    import webbrowser

    webbrowser.open(f"file://{html_path}")
    print(f"\n\u2705 Opening report in browser...")
