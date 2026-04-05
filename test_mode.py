"""Test images data and pure test mode runner."""

import asyncio
import base64
import requests
from datetime import datetime
from config import DEFAULT_MODEL
from analyzer import analyze_image_with_claude, save_herman_miller_listing
from email_alert import send_email_alert

# Real Herman Miller chair images for testing
# Mix of Reddit/FB Marketplace finds + official product photos
TEST_HERMAN_MILLER_IMAGES = [
    # From Reddit - real marketplace-style photos (what we're hunting for)
    {
        "url": "https://preview.redd.it/purchasing-aeron-off-fb-marketplace-is-this-herman-chair-v0-uh9ahc8krgc81.jpg?width=640&crop=smart&auto=webp&s=47e5f8feb4a844973539eb12052a72287ecd3952",
        "title": "Office Chair",
        "price": "$80",
    },
    {
        "url": "https://preview.redd.it/purchasing-aeron-off-fb-marketplace-is-this-herman-chair-v0-s7if7g8krgc81.jpg?width=640&crop=smart&auto=webp&s=8373066fe13bb848a40fcb57f6e751724a848b65",
        "title": "Computer Chair",
        "price": "$50",
    },
    {
        "url": "https://preview.redd.it/purchasing-aeron-off-fb-marketplace-is-this-herman-chair-v0-e5a4sc8krgc81.jpg?width=640&crop=smart&auto=webp&s=99c73ea787480e95dbbf336ef283cb080c5ffad5",
        "title": "Mesh Office Chair",
        "price": "$120",
    },
    {
        "url": "https://preview.redd.it/purchasing-aeron-off-fb-marketplace-is-this-herman-chair-v0-l1aiqc8krgc81.jpg?width=640&crop=smart&auto=webp&s=affe0b9ce7613a731bae3fa849aea689c441d9a3",
        "title": "Desk Chair - Good Condition",
        "price": "$75",
    },
    {
        "url": "https://preview.redd.it/purchasing-aeron-off-fb-marketplace-is-this-herman-chair-v0-ek6ntd8krgc81.jpg?width=640&crop=smart&auto=webp&s=4b8cef62e545fd4ef390fe4e8f99b4360ce241d4",
        "title": "Black Office Chair",
        "price": "$100",
    },
    {
        "url": "https://i.redd.it/fb-marketplace-is-crazy-v0-os3h00el8awc1.jpg?width=4032&format=pjpg&auto=webp&s=e7bd2013b1b675cdb60777f5d742598969f0a820",
        "title": "Office Chair Moving Sale",
        "price": "$45",
    },
    {
        "url": "https://preview.redd.it/facebook-marketplace-for-300-v0-5wgf6fxbovhf1.jpg?width=640&crop=smart&auto=webp&s=87049a79533b8846edeebc56fb6ea6f2c12a5fa0",
        "title": "Ergonomic Chair",
        "price": "$300",
    },
    {
        "url": "https://preview.redd.it/won-the-fb-marketplace-lottery-today-v0-hx8qrh61r23g1.jpg?width=640&crop=smart&auto=webp&s=5c0e617d2a61924a1211a9700ebbe2f24c711778",
        "title": "Chair - Must Go Today",
        "price": "$60",
    },
    # Official Herman Miller product photos (cleaner, easier to recognize)
    {
        "url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/aeron_chair/202106/mh_prd_ovw_aeron_chair.jpg",
        "title": "Work Chair",
        "price": "$150",
    },
    {
        "url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/aeron_chair/202106/ig_prd_ovw_aeron_chair_01.jpg",
        "title": "Office Furniture",
        "price": "$200",
    },
    {
        "url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/embody_chairs/mh_prd_ovw_embody_chairs.jpg",
        "title": "Desk Chair Blue",
        "price": "$175",
    },
    {
        "url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/mirra_2_chair/mh_prd_ovw_mirra_2_chair.jpg",
        "title": "Mesh Chair",
        "price": "$90",
    },
    {
        "url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/sayl_chairs/mh_prd_ovw_sayl_chairs.jpg",
        "title": "Modern Office Chair",
        "price": "$125",
    },
    {
        "url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/cosm_chairs/northamerica/mh_prd_ovw_cosm_chairs_na.jpg",
        "title": "Ergonomic Seat",
        "price": "$250",
    },
    # More lifestyle/room shots from Herman Miller
    {
        "url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/aeron_chair/202106/ig_prd_ovw_aeron_chair_02.jpg",
        "title": "Home Office Chair",
        "price": "$180",
    },
    {
        "url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/aeron_chair/202106/ig_prd_ovw_aeron_chair_03.jpg",
        "title": "Conference Chair",
        "price": "$95",
    },
    {
        "url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/aeron_chair/202106/ig_prd_ovw_aeron_chair_04.jpg",
        "title": "Task Chair",
        "price": "$160",
    },
    {
        "url": "https://www.hermanmiller.com/content/dam/hmicom/page_assets/products/aeron_chair/202106/ig_prd_ovw_aeron_chair_05.jpg",
        "title": "Office Seating",
        "price": "$110",
    },
]

# Premium NON-Herman Miller chairs - should be identified as NOT HM
# These are expensive quality chairs that models might confuse with Herman Miller
TEST_OTHER_PREMIUM_CHAIRS = [
    # Steelcase Leap - $1,400+ retail, LiveBack technology
    {
        "url": "https://steelcase-res.cloudinary.com/image/upload/v1610026604/20-0149894.jpg",
        "title": "Office Chair",
        "price": "$200",
        "actual_brand": "Steelcase Leap",
    },
    # Steelcase Gesture - $2,000+ retail, 360-degree arms
    {
        "url": "https://images.steelcase.com/image/upload/v1676059815/21-0166043-1.jpg",
        "title": "Ergonomic Desk Chair",
        "price": "$150",
        "actual_brand": "Steelcase Gesture",
    },
    # Humanscale Freedom - $1,200+ retail, self-adjusting recline
    {
        "url": "https://www.ergodirect.com/images/Humanscale/13611/large/Humanscale-Freedom-Task-Chair_lg_1745860590.jpg",
        "title": "Task Chair",
        "price": "$175",
        "actual_brand": "Humanscale Freedom",
    },
    # Humanscale Liberty - $900+ retail, tri-panel mesh back
    {
        "url": "https://cdn11.bigcommerce.com/s-492apnl0xy/images/stencil/1280x1280/products/744/3282/humanscale-liberty-chair-hus088__49475.1490806767.jpg?c=2",
        "title": "Mesh Office Chair",
        "price": "$120",
        "actual_brand": "Humanscale Liberty",
    },
    # Haworth Fern - $1,500+ retail, Wave Suspension system
    {
        "url": "https://store.haworth.com/cdn/shop/files/Fern-Mesh_53ffb43c-2638-4ce3-a324-ae702c3fc1ef.jpg?v=1720535915",
        "title": "Executive Chair",
        "price": "$250",
        "actual_brand": "Haworth Fern",
    },
]


async def run_pure_test_mode():
    """Run through test images without touching Facebook - for testing the AI + email flow."""
    print("=" * 60)
    print("\U0001f9ea PURE TEST MODE - Herman Miller Chair Finder")
    print(f"Started at: {datetime.now()}")
    print("Testing with pre-loaded Herman Miller images...")
    print("=" * 60)

    herman_millers = []

    for i, test_listing in enumerate(TEST_HERMAN_MILLER_IMAGES):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(TEST_HERMAN_MILLER_IMAGES)}] Test Listing")
        print(f"{'=' * 60}")

        listing = {
            "id": f"test_{i + 1}",
            "url": f"https://www.facebook.com/marketplace/item/test{i + 1}/",
            "title": test_listing["title"],
            "price": test_listing["price"],
        }

        print(f"\n  \U0001f4cb FAKE LISTING DATA:")
        print(f"     Title: {listing['title']}")
        print(f"     Price: {listing['price']}")
        print(f"     (This is what a seller might list a Herman Miller as)")

        print(f"\n  \U0001f4f7 Downloading image...")
        try:
            img_response = requests.get(test_listing["url"], timeout=15)
            if img_response.status_code == 200:
                image_base64 = base64.b64encode(img_response.content).decode("utf-8")
                print(f"     Image size: {len(img_response.content) / 1024:.1f} KB")

                print(f"\n  \U0001f916 Sending to {DEFAULT_MODEL.split('/')[-1]}...")
                analysis = analyze_image_with_claude(image_base64)

                if analysis:
                    print(f"\n  \U0001f4ca AI ANALYSIS:")
                    print(f"     Reasoning: {analysis.get('reasoning', 'N/A')}")
                    print(f"     Model: {analysis.get('model', 'Unknown')}")
                    print(f"     Confidence: {analysis.get('confidence', 'Unknown')}")
                    print(
                        f"     Is Herman Miller: {analysis.get('is_herman_miller', 'Unknown')}"
                    )

                    if analysis.get("is_herman_miller"):
                        print(
                            f"\n  \U0001f389\U0001f389\U0001f389 HERMAN MILLER FOUND! \U0001f389\U0001f389\U0001f389"
                        )
                        save_herman_miller_listing(listing, analysis, image_base64)
                        herman_millers.append({**listing, **analysis, "is_test": True})
                    else:
                        print(f"\n  \u274c Not recognized as Herman Miller")
                else:
                    print(f"\n  \u274c Analysis failed - no response from AI")
            else:
                print(f"     Failed to download: {img_response.status_code}")
        except Exception as e:
            print(f"\n  \u274c Error: {e}")

        # Small delay between tests
        print(f"\n  \u23f3 Waiting 2s before next test...")
        await asyncio.sleep(2)

    # Send email alert
    if herman_millers:
        print(f"\n\U0001f4e7 Sending email alert with {len(herman_millers)} finds...")
        send_email_alert(herman_millers)
    else:
        print("\n\U0001f4e7 No Herman Miller chairs recognized, skipping email")

    print("\n" + "=" * 60)
    print(
        f"Test complete! AI recognized {len(herman_millers)}/{len(TEST_HERMAN_MILLER_IMAGES)} as Herman Miller"
    )
    print("=" * 60)
