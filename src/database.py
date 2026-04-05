"""SQLite operations for tracking found listings."""

import sqlite3
from datetime import datetime
from src.config import DB_PATH


def init_database():
    """Initialize SQLite database for tracking found listings."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
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
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS seen (
            listing_id TEXT PRIMARY KEY,
            seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def is_listing_known(listing_id):
    """Check if we've already processed this listing."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM seen WHERE listing_id = ?", (listing_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def mark_listing_seen(listing_id):
    """Mark a listing as processed so we never re-analyze it."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO seen (listing_id, seen_at) VALUES (?, ?)",
        (listing_id, datetime.now()),
    )
    conn.commit()
    conn.close()


def save_listing_to_db(
    listing_id,
    url,
    title,
    price,
    model,
    confidence,
    reasoning,
    deal_score=None,
    deal_label=None,
):
    """Save a found Herman Miller listing to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO listings
        (listing_id, url, title, price, model, confidence, reasoning, deal_score, deal_label, found_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            listing_id,
            url,
            title,
            price,
            model,
            confidence,
            reasoning,
            deal_score,
            deal_label,
            datetime.now(),
        ),
    )
    conn.commit()
    conn.close()


def mark_listing_alerted(listing_id):
    """Mark a listing as having been included in an email alert."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE listings SET alerted_at = ? WHERE listing_id = ?",
        (datetime.now(), listing_id),
    )
    conn.commit()
    conn.close()


def get_unalerted_listings():
    """Get listings that haven't been included in an alert yet."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM listings WHERE alerted_at IS NULL ORDER BY found_at DESC"
    )
    columns = [description[0] for description in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results


def get_listing_stats():
    """Get stats about found listings."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM listings")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM listings WHERE alerted_at IS NOT NULL")
    alerted = cursor.fetchone()[0]
    cursor.execute(
        "SELECT model, COUNT(*) as cnt FROM listings GROUP BY model ORDER BY cnt DESC"
    )
    by_model = cursor.fetchall()
    conn.close()
    return {"total": total, "alerted": alerted, "by_model": by_model}


# Initialize database on import
init_database()
