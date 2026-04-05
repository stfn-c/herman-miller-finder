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

import sys
import asyncio
from src.config import args, PURE_TEST_MODE, BENCHMARK_MODE
from src.benchmark import (
    list_benchmark_runs,
    compare_benchmark_runs,
    run_benchmark_mode,
)
from src.test_mode import run_pure_test_mode
from src.scraper import run_facebook_scraper, run_all_cities
from src.scheduler import run_scheduler, acquire_lock, release_lock


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

    # Default: run Facebook scraper
    await run_facebook_scraper()


if __name__ == "__main__":
    if args.scheduler:
        run_scheduler(lambda: run_all_cities(main))
    else:
        # Single run mode - still use lock to prevent concurrent runs
        lock_fd = acquire_lock()
        if not lock_fd:
            print("\u274c Another instance is already running. Exiting.")
            sys.exit(1)
        try:
            asyncio.run(run_all_cities(main))
        finally:
            release_lock(lock_fd)
