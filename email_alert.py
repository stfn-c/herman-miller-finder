"""Email alerting: send_email_alert with HTML generation."""

import requests
from config import RESEND_API_KEY, FROM_EMAIL, TO_EMAIL


def send_email_alert(listings):
    """Send email alert with found premium chairs and deal scores."""

    if not listings:
        print("No premium chairs found to report")
        return

    # Sort by deal score (best deals first)
    sorted_listings = sorted(
        listings, key=lambda x: x.get("deal_score", 0), reverse=True
    )

    # Build email HTML
    html_items = []
    fumble_count = 0
    for listing in sorted_listings:
        # Get deal info
        deal_score = listing.get("deal_score")
        deal_label = listing.get("deal_label", "")
        retail_price = listing.get("retail_price")

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
                <br><span style="color: #666;">Retail: ${retail_price} \u2192 Listed: {listing.get("price", "Unknown")}</span>
            </div>
            """

        brand = listing.get("brand", "Unknown")
        model = listing.get("model", "Unknown")
        chair_name = f"{brand} {model}" if brand != "Unknown" else model

        html_items.append(f"""
        <div style="border: 2px solid {border_color}; padding: 15px; margin: 10px 0; border-radius: 8px; background: {bg_color};">
            <h3 style="margin: 0 0 10px 0;">{chair_name}</h3>
            {deal_info}
            <p><strong>Price:</strong> {listing.get("price", "Unknown")}</p>
            <p><strong>Title:</strong> {listing.get("title", "Unknown")}</p>
            <p><strong>Confidence:</strong> {listing.get("confidence", "Unknown")}</p>
            <p><strong>Analysis:</strong> {listing.get("reasoning", "N/A")}</p>
            <p><a href="{listing.get("url", "#")}" style="background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">View Listing \u2192</a></p>
        </div>
        """)

    # Subject line reflects deal quality
    if fumble_count > 0:
        subject = f"\U0001f525 {fumble_count} FUMBLE(S)! {len(listings)} Premium Chair(s) Found!"
    else:
        subject = f"\U0001fa91 {len(listings)} Premium Chair(s) Found on Marketplace!"

    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <h1 style="color: #333;">\U0001fa91 Premium Chairs Found!</h1>
        <p>Found {len(listings)} premium chair(s) on Facebook Marketplace:</p>
        {"".join(html_items)}
        <hr style="margin: 30px 0; border: none; border-top: 1px solid #ddd;">
        <p style="color: #666; font-size: 12px;">Deal Score Guide: \U0001f525 FUMBLE (10) = &lt;15% retail | \U0001f48e STEAL (8-9) = &lt;25% | \U0001f3af GREAT (6-7) = &lt;40% | \U0001f44d GOOD (4-5) = &lt;60%</p>
    </body>
    </html>
    """

    # Send via Resend
    url = "https://api.resend.com/emails"
    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "from": FROM_EMAIL,
        "to": TO_EMAIL,
        "subject": subject,
        "html": html_content,
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code in [200, 201]:
            print(f"\U0001f4e7 Email sent successfully to {TO_EMAIL}")
        else:
            print(f"Failed to send email: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending email: {e}")
