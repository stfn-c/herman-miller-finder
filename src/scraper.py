"""Facebook Marketplace scraping: browser setup, scrolling, listing collection."""

import os
import random
import base64
import asyncio
import requests
from datetime import datetime
from cloakbrowser import launch_persistent_context_async
import src.config as config
from src.config import (
    HEADLESS_MODE,
    LOCALE,
    TIMEZONE,
    LATITUDE,
    LONGITUDE,
    FB_COOKIES,
    CHAIR_SEARCHES,
    DECOY_SEARCHES,
    DEV_MODE,
    SCROLL_COUNT,
    LISTING_COUNT,
    TEST_MODE_CHANCE,
    VERBOSE_LOGGING,
    MIN_DEAL_SCORE,
    DEFAULT_MODEL,
    LISTING_DELAY_MIN,
    LISTING_DELAY_MAX,
    MARKETPLACE_LOCATIONS,
    calculate_deal_score,
)
from src.database import (
    is_listing_known,
    mark_listing_seen,
    save_listing_to_db,
    mark_listing_alerted,
    get_listing_stats,
)
from src.analyzer import analyze_image_with_claude, save_herman_miller_listing
from src.email_alert import send_email_alert
from src.test_mode import TEST_HERMAN_MILLER_IMAGES


async def run_facebook_scraper():
    """Main Facebook scraping function to find Herman Miller chairs."""

    print("=" * 60)
    print("Herman Miller Chair Finder")
    print(f"Started at: {datetime.now()}")
    print("=" * 60)

    herman_millers = []

    profile_dir = str(config.SCRIPT_DIR / ".fb_profile")
    print("\n\U0001f310 Launching CloakBrowser...")

    context = await launch_persistent_context_async(
        profile_dir,
        headless=HEADLESS_MODE,
        humanize=True,
        locale=LOCALE,
        timezone_id=TIMEZONE,
        geolocation={"latitude": LATITUDE, "longitude": LONGITUDE},
        permissions=["geolocation"],
        color_scheme="light",
        viewport={"width": 1280, "height": 900},
    )

    try:
        await context.add_cookies(FB_COOKIES)

        page = await context.new_page()

        # Helper: human-like scrolling with random pauses, mouse movements, etc
        async def human_scroll(num_scrolls, description=""):
            for i in range(num_scrolls):
                # Random scroll method
                scroll_method = random.choices(
                    ["PageDown", "PageDown", "wheel", "arrow", "End"],
                    weights=[40, 30, 15, 10, 5],
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
                        x = random.randint(100, viewport["width"] - 100)
                        y = random.randint(100, viewport["height"] - 100)
                        await page.mouse.move(x, y, steps=random.randint(5, 15))
                        await page.wait_for_timeout(random.uniform(200, 800))

                # Sometimes hover over a listing (but don't click)
                if random.random() < 0.1:
                    try:
                        items = await page.query_selector_all(
                            'a[href*="/marketplace/item/"]'
                        )
                        if items:
                            item = random.choice(items[:10])  # Pick from visible ones
                            await item.hover()
                            await page.wait_for_timeout(random.uniform(500, 1500))
                    except:
                        pass

                await page.wait_for_timeout(base_delay)

                if description and (i + 1) % 5 == 0:
                    print(f"   {description}: {i + 1}/{num_scrolls} scrolls...")

        # Helper: browse a search without analyzing (decoy behavior)
        async def browse_decoy(query):
            print(f"\n\U0001f3ad Browsing '{query}' (decoy)...")
            await page.goto(
                f"https://www.facebook.com/marketplace/{config.CURRENT_CITY}/search?query={query.replace(' ', '%20')}",
                wait_until="domcontentloaded",
                timeout=60000,
            )
            await page.wait_for_timeout(random.uniform(2000, 4000))

            # Scroll around
            await human_scroll(random.randint(3, 10), f"Browsing {query}")

            # Maybe click on a random listing
            if random.random() < 0.4:
                try:
                    items = await page.query_selector_all(
                        'a[href*="/marketplace/item/"]'
                    )
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
                search_queue.insert(
                    0, {"query": random.choice(DECOY_SEARCHES), "analyze": False}
                )

            # Maybe add another chair search
            if random.random() < 0.3:
                other_chair = random.choice(
                    [s for s in CHAIR_SEARCHES if s != chair_search]
                )
                search_queue.append(other_chair)

            # Maybe end with a decoy
            if random.random() < 0.4:
                search_queue.append(
                    {"query": random.choice(DECOY_SEARCHES), "analyze": False}
                )

        print(f"\n\U0001f4cb Search plan: {[s['query'] for s in search_queue]}")

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
            print(f"\n\U0001f50d Searching for '{query}'...")
            await page.goto(
                f"https://www.facebook.com/marketplace/{config.CURRENT_CITY}/search?query={query.replace(' ', '%20')}",
                wait_until="domcontentloaded",
                timeout=60000,
            )
            await page.wait_for_timeout(random.uniform(2500, 5000))

            # Scroll to load lots of listings
            scroll_count = SCROLL_COUNT if not DEV_MODE else 3
            print(f"\U0001f4dc Scrolling ({scroll_count} times)...")
            await human_scroll(scroll_count, "Loading chairs")

            # Collect listings from this search
            listing_elements = await page.query_selector_all(
                'a[href*="/marketplace/item/"]'
            )

            for elem in listing_elements:
                try:
                    href = await elem.get_attribute("href")
                    if href and "/marketplace/item/" in href:
                        parts = href.split("/marketplace/item/")
                        if len(parts) > 1:
                            listing_id = parts[1].split("/")[0].split("?")[0]
                            if listing_id and listing_id not in seen_ids:
                                seen_ids.add(listing_id)
                                listings.append(
                                    {
                                        "id": listing_id,
                                        "url": f"https://www.facebook.com/marketplace/item/{listing_id}/",
                                        "element": elem,
                                        "source_query": query,
                                    }
                                )
                except:
                    continue

            print(f"   Found {len(listings)} total unique listings so far")

            # Brief pause between searches
            if search != search_queue[-1]:
                await page.wait_for_timeout(random.uniform(2000, 5000))

        print(f"\n\u2705 Total: {len(listings)} unique chair listings to analyze")

        # Filter out already-known listings
        new_listings = []
        skipped = 0
        for listing in listings:
            if is_listing_known(listing["id"]):
                skipped += 1
            else:
                new_listings.append(listing)

        if skipped > 0:
            print(f"\u23ed\ufe0f  Skipped {skipped} already-processed listings")
        print(f"\U0001f4cb {len(new_listings)} new listings to check")

        # Process each NEW listing
        for i, listing in enumerate(new_listings[:LISTING_COUNT]):
            print(f"\n{'=' * 60}")
            print(
                f"[{i + 1}/{min(len(new_listings), LISTING_COUNT)}] Checking listing {listing['id']}"
            )
            print(f"{'=' * 60}")

            # Determine if this is a test run (1 in TEST_MODE_CHANCE)
            # In test mode, we swap in a REAL Herman Miller image so the AI genuinely
            # recognizes it - this simulates finding an actual Herman Miller listed as
            # a generic "office chair" (which is exactly what we're hunting for!)
            is_test_mode = random.randint(1, TEST_MODE_CHANCE) == 1
            test_chair = None
            if is_test_mode:
                test_chair = random.choice(TEST_HERMAN_MILLER_IMAGES)
                print(f"  \U0001f9ea TEST MODE ACTIVATED (1/{TEST_MODE_CHANCE} chance)")
                print(f"     Swapping image with test Herman Miller photo")
                print(
                    f"     (Simulating: someone listed a Herman Miller as '{test_chair.get('title', 'office chair')}')"
                )

            try:
                # Navigate to listing page
                await page.goto(
                    listing["url"], wait_until="domcontentloaded", timeout=30000
                )
                await page.wait_for_timeout(2000)

                # Get the title
                title_elem = await page.query_selector("h1")
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
                    print(f"\n  \U0001f4cb SCRAPED DATA:")
                    print(f"     Title: {title}")
                    print(f"     Price: {listing['price']}")
                    print(f"     URL: {listing['url']}")

                # Find the main image
                img_elem = await page.query_selector(
                    'img[data-visualcompletion="media-vc-image"]'
                )
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
                            print(
                                f"\n  \U0001f9ea SWAPPED IMAGE URL to: {img_src[:60]}..."
                            )

                        print(f"\n  \U0001f4f7 Downloading image...")
                        try:
                            img_response = requests.get(img_src, timeout=10)
                            if img_response.status_code == 200:
                                image_base64 = base64.b64encode(
                                    img_response.content
                                ).decode("utf-8")

                                if VERBOSE_LOGGING:
                                    print(
                                        f"     Image size: {len(img_response.content) / 1024:.1f} KB"
                                    )

                                print(
                                    f"\n  \U0001f916 Sending to {DEFAULT_MODEL.split('/')[-1]}..."
                                )
                                analysis = analyze_image_with_claude(image_base64)

                                if analysis:
                                    model = analysis.get("model", "Unknown")
                                    is_hm = analysis.get("is_herman_miller", False)
                                    confidence = analysis.get("confidence", "Unknown")
                                    reasoning = analysis.get("reasoning", "N/A")

                                    print(f"\n  \U0001f4ca AI ANALYSIS:")
                                    print(f"     Model: {model}")
                                    print(f"     Reasoning: {reasoning}")
                                    print(f"     Confidence: {confidence}")
                                    print(f"     Is Herman Miller: {is_hm}")

                                    if is_hm:
                                        # Calculate deal score
                                        deal_score, deal_label, retail_price = (
                                            calculate_deal_score(
                                                listing["price"], model
                                            )
                                        )

                                        print(
                                            f"\n  \U0001f389\U0001f389\U0001f389 HERMAN MILLER FOUND: {model} \U0001f389\U0001f389\U0001f389"
                                        )
                                        if deal_score is not None:
                                            print(
                                                f"  \U0001f4b0 Deal Score: {deal_score}/10 {deal_label}"
                                            )
                                            print(
                                                f"     Retail: ${retail_price} \u2192 Listed: {listing['price']}"
                                            )
                                        if is_test_mode:
                                            print(
                                                f"  (This is a TEST - not a real find)"
                                            )

                                        # Save to database (skip if test mode)
                                        if not is_test_mode:
                                            save_listing_to_db(
                                                listing_id=listing["id"],
                                                url=listing["url"],
                                                title=listing.get("title", "Unknown"),
                                                price=listing.get("price", "Unknown"),
                                                model=model,
                                                confidence=confidence,
                                                reasoning=reasoning,
                                                deal_score=deal_score,
                                                deal_label=deal_label,
                                            )
                                            print(f"  \U0001f4be Saved to database")

                                        save_herman_miller_listing(
                                            listing, analysis, image_base64
                                        )
                                        herman_millers.append(
                                            {
                                                **listing,
                                                **analysis,
                                                "deal_score": deal_score,
                                                "deal_label": deal_label,
                                                "retail_price": retail_price,
                                                "is_test": is_test_mode,
                                            }
                                        )
                                    else:
                                        print(f"\n  \u274c Not a Herman Miller")
                                else:
                                    print(
                                        f"\n  \u274c Analysis failed - no response from AI"
                                    )
                        except Exception as e:
                            print(f"\n  \u274c Failed to download image: {e}")
                else:
                    print(f"\n  \u26a0\ufe0f No image found on this listing")

            except Exception as e:
                print(f"\n  ❌ Error processing listing: {e}")
                mark_listing_seen(listing["id"])
                continue

            mark_listing_seen(listing["id"])

            # Rate limiting - random delay to seem more human
            wait_time = random.uniform(LISTING_DELAY_MIN, LISTING_DELAY_MAX)

            # In prod, occasionally take a longer break
            if not DEV_MODE and random.random() < 0.15:
                wait_time += random.uniform(5000, 15000)
                print(f"\n  \u23f3 Taking a break... {wait_time / 1000:.1f}s")
            else:
                print(
                    f"\n  \u23f3 Waiting {wait_time / 1000:.1f}s before next listing..."
                )

            await page.wait_for_timeout(wait_time)

    finally:
        await context.close()

    # Send email alert (only for non-test finds unless EMAIL_TEST_FINDS is set)
    include_test = os.environ.get("EMAIL_TEST_FINDS", "").lower() in (
        "true",
        "1",
        "yes",
    )
    real_finds = [h for h in herman_millers if include_test or not h.get("is_test")]

    # Filter by minimum deal score if set
    if MIN_DEAL_SCORE > 0:
        real_finds = [
            h for h in real_finds if (h.get("deal_score") or 0) >= MIN_DEAL_SCORE
        ]

    if real_finds:
        print(
            f"\n\U0001f4e7 Sending email alert for {len(real_finds)} Herman Miller(s)..."
        )
        send_email_alert(real_finds)

        # Mark as alerted in database
        for h in real_finds:
            mark_listing_alerted(h["id"])

        # Print summary with deal scores
        fumbles = [h for h in real_finds if h.get("deal_score", 0) >= 8]
        if fumbles:
            print(
                f"\n\U0001f525 FUMBLE ALERT: {len(fumbles)} incredible deal(s) found!"
            )
            for f in fumbles:
                print(
                    f"   - {f.get('model', '')} @ {f.get('price', '?')} ({f.get('deal_label', '')})"
                )
    else:
        print("\n\U0001f4e7 No Herman Miller chairs found, skipping email")

    # Show database stats
    stats = get_listing_stats()
    print("\n" + "=" * 60)
    print(f"Scan complete! Found {len(herman_millers)} Herman Miller(s) this run")
    print(
        f"\U0001f4ca Database: {stats['total']} total finds, {stats['alerted']} alerted"
    )
    if stats["by_model"]:
        print(f"   By model: {', '.join(f'{m}: {c}' for m, c in stats['by_model'])}")
    print("=" * 60)


async def run_all_cities(main_fn):
    """Run main_fn for each configured city."""

    print(
        f"\U0001f30d Scanning {len(MARKETPLACE_LOCATIONS)} city/cities: {', '.join(MARKETPLACE_LOCATIONS)}"
    )

    for i, city in enumerate(MARKETPLACE_LOCATIONS):
        config.CURRENT_CITY = city

        if len(MARKETPLACE_LOCATIONS) > 1:
            print(f"\n{'=' * 60}")
            print(
                f"\U0001f4cd City {i + 1}/{len(MARKETPLACE_LOCATIONS)}: {city.upper()}"
            )
            print(f"{'=' * 60}")

            # Delay between cities in prod mode
            if i > 0 and not DEV_MODE:
                delay = random.uniform(30, 90)
                print(f"\u23f3 Waiting {delay:.0f}s before next city...")
                await asyncio.sleep(delay)

        await main_fn()
