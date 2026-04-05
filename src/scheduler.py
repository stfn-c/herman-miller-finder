"""Scheduler: run_scheduler, lock management, active hours."""

import os
import sys
import time
import fcntl
import random
import asyncio
from datetime import datetime
from src.config import (
    LOCAL_TZ,
    TIMEZONE,
    LOCK_FILE,
    SCHEDULER_RUNS_PER_DAY,
    SCHEDULER_START_HOUR,
    SCHEDULER_END_HOUR,
)


def acquire_lock():
    """Acquire exclusive lock to ensure only one browser instance runs."""
    lock_fd = open(LOCK_FILE, "w")
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


def run_scheduler(run_all_cities_fn):
    """Run the scheduler loop - 12x/day during waking hours."""
    print("=" * 60)
    print("\U0001f550 Herman Miller Finder - Scheduler Mode")
    print(f"   Runs: {SCHEDULER_RUNS_PER_DAY}x per day")
    print(
        f"   Active hours: {SCHEDULER_START_HOUR}:00 - {SCHEDULER_END_HOUR - 24}:00 ({TIMEZONE})"
    )
    print("=" * 60)

    run_count = 0

    while True:
        local_now = datetime.now(LOCAL_TZ)

        if not is_within_active_hours():
            # Calculate time until 9am local
            if (
                local_now.hour >= (SCHEDULER_END_HOUR - 24)
                and local_now.hour < SCHEDULER_START_HOUR
            ):
                hours_until_active = SCHEDULER_START_HOUR - local_now.hour
                sleep_seconds = hours_until_active * 3600 - local_now.minute * 60
                print(
                    f"\n\U0001f634 Outside active hours ({local_now.strftime('%H:%M')} {TIMEZONE})"
                )
                print(f"   Sleeping until 9:00 AM ({hours_until_active:.1f}h)...")
                time.sleep(max(60, sleep_seconds))
                continue

        # Try to acquire lock
        lock_fd = acquire_lock()
        if not lock_fd:
            print(f"\n\u26a0\ufe0f  Another instance is running, waiting 5 minutes...")
            time.sleep(300)
            continue

        try:
            run_count += 1
            print(f"\n{'=' * 60}")
            print(
                f"\U0001f50d Starting scan #{run_count} at {local_now.strftime('%Y-%m-%d %H:%M:%S')} {TIMEZONE}"
            )
            print(f"{'=' * 60}")

            # Run the main scan (all cities)
            asyncio.run(run_all_cities_fn())

        except Exception as e:
            print(f"\n\u274c Error during scan: {e}")
        finally:
            release_lock(lock_fd)

        # Calculate next run time
        delay = get_next_run_delay()
        next_run = datetime.now(LOCAL_TZ).timestamp() + delay
        next_run_time = datetime.fromtimestamp(next_run, LOCAL_TZ)

        print(f"\n\u23f0 Next scan at {next_run_time.strftime('%H:%M:%S')} {TIMEZONE}")
        print(f"   (sleeping {delay / 60:.0f} minutes)")
        time.sleep(delay)
