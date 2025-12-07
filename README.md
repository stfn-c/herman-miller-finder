# Herman Miller Chair Finder

**Stop overpaying for office chairs.** This tool automatically scans Facebook Marketplace for Herman Miller chairs and alerts you when someone lists one at a good price.

Herman Miller chairs (like the Aeron) retail for $1,500-$2,500 new, but people often sell them used without knowing their value. This tool uses AI to spot them and emails you before they're gone.

## What It Does

1. **Scans Facebook Marketplace** - Searches for office chairs in your area
2. **Identifies Herman Miller chairs** - AI looks at each listing photo and determines if it's a genuine Herman Miller
3. **Calculates the deal quality** - Compares the asking price to retail value
4. **Sends you an email** - When it finds a good deal, you get notified immediately

## How Good Is the Deal?

The tool rates every find on a 0-10 scale:

| Score | What It Means | Example |
|-------|---------------|---------|
| 10 - FUMBLE | Seller has no idea what they have | $200 Aeron (retails $2,000) |
| 8-9 - STEAL | Exceptional deal, act fast | $400 Aeron |
| 6-7 - GREAT | Well below market value | $600 Aeron |
| 4-5 - GOOD | Solid used price | $900 Aeron |
| 2-3 - FAIR | Reasonable but not exciting | $1,400 Aeron |
| 0-1 - PASS | At or above retail | $1,800+ Aeron |

## Chairs It Can Identify

Aeron, Embody, Sayl, Mirra, Mirra 2, Cosm, Setu, Eames Soft Pad, Eames Aluminum Group, Celle, Lino, Verus, Caper, and more.

---

## Setup Guide

### What You'll Need

- **A computer with Python installed** (version 3.8 or newer)
- **An OpenRouter account** - This gives the tool access to AI (free tier available at [openrouter.ai](https://openrouter.ai))
- **A Resend account** - For sending email alerts (free tier at [resend.com](https://resend.com))
- **A Facebook account** - The tool browses Marketplace as you

### Step 1: Download the Code

Open Terminal (Mac) or Command Prompt (Windows) and run:

```
git clone https://github.com/stfn-c/herman-miller-finder.git
cd herman-miller-finder
```

### Step 2: Install Requirements

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

On Windows, replace `source venv/bin/activate` with `venv\Scripts\activate`

### Step 3: Set Up Your Credentials

Create a file called `.env` (copy from the example):

```
cp .env.example .env
```

Open `.env` in any text editor and fill in:

| Setting | Where to Get It |
|---------|-----------------|
| `OPENROUTER_API_KEYS` | [openrouter.ai/keys](https://openrouter.ai/keys) - Create an API key |
| `RESEND_API_KEY` | [resend.com/api-keys](https://resend.com/api-keys) - Create an API key |
| `FROM_EMAIL` | Your sending email (must verify domain in Resend) |
| `TO_EMAIL` | Where you want alerts sent |
| `FB_COOKIES` | Your Facebook login session (see Step 4) |
| `TIMEZONE` | Your timezone (e.g., `America/New_York`, `Europe/London`) |
| `MARKETPLACE_LOCATION` | Your city's Facebook Marketplace slug (e.g., `nyc`, `london`, `sydney`) |
| `LATITUDE` / `LONGITUDE` | Your location coordinates (for realistic browsing) |

### Step 4: Get Your Facebook Cookies

The tool needs your Facebook login cookies to browse Marketplace. We've included a script to make this easy:

1. Open Chrome and go to [facebook.com](https://facebook.com)
2. Make sure you're logged in
3. Press `F12` to open Developer Tools
4. Click the **Console** tab
5. Copy the entire contents of `export_cookies.js` and paste it into the console
6. Press Enter
7. The script will output your `FB_COOKIES=...` line - copy it into your `.env` file

---

## Running the Tool

### Basic Usage

```
python find_herman_miller.py
```

This scans Marketplace once and emails you any Herman Miller chairs it finds.

### Run It Automatically

To have it check throughout the day (12 times, from 9am to 2am in your timezone):

```
python find_herman_miller.py --scheduler
```

### Other Options

| Command | What It Does |
|---------|--------------|
| `python find_herman_miller.py --test` | Test with sample images (no Facebook) |
| `python find_herman_miller.py -n 50` | Check 50 listings instead of default 20 |
| `python find_herman_miller.py --verbose` | Show detailed progress |
| `python find_herman_miller.py --benchmark` | Compare different AI models' accuracy |

---

## How It Works (Non-Technical)

1. The tool opens an invisible browser and goes to Facebook Marketplace
2. It searches for terms like "office chair", "ergonomic chair", "mesh chair"
3. For each listing, it downloads the photo and sends it to an AI
4. The AI analyzes the image and says "This is a Herman Miller Aeron" or "This is just a generic office chair"
5. If it's a Herman Miller, the tool calculates if the price is good compared to retail
6. Good deals get emailed to you with the listing link, photo, and deal score

---

## Finding Your Location Settings

### Timezone

Use a timezone from [this list](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones). Common examples:
- `America/New_York`
- `America/Los_Angeles`
- `Europe/London`
- `Australia/Sydney`

### Marketplace Location

This is the city slug Facebook uses in URLs. Go to Facebook Marketplace, search in your city, and look at the URL:

```
https://www.facebook.com/marketplace/CITY_SLUG/search?query=...
                                      ^^^^^^^^^^
```

Examples: `nyc`, `london`, `sydney`, `melbourne`, `toronto`, `losangeles`

### Latitude & Longitude

Search for your city on Google Maps and look at the URL, or use [latlong.net](https://www.latlong.net/).

---

## Important Notes

- **Facebook's Rules**: Automated browsing may violate Facebook's terms of service. Use at your own discretion.
- **Cookie Expiry**: Facebook cookies expire periodically. If the tool stops working, you'll need to get fresh cookies using the export script.

## License

MIT License - free to use and modify.
