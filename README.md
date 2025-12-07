# Herman Miller Chair Finder

People sell Herman Miller chairs on Facebook Marketplace all the time without knowing what they have. I got tired of missing out on $200 Aerons, so I built this.

It scans Marketplace, uses AI to identify Herman Miller chairs from photos, and emails you when it finds one.

## How It Works

1. Scrapes Facebook Marketplace for office chair listings
2. AI analyzes each photo to identify Herman Miller chairs
3. Calculates how good the deal is compared to retail price
4. Emails you the good ones

## Deal Scores

| Score | Meaning |
|-------|---------|
| 10 | FUMBLE - seller has no clue ($200 Aeron) |
| 8-9 | STEAL - way under market |
| 6-7 | GREAT - solid deal |
| 4-5 | GOOD - fair used price |
| 2-3 | FAIR - nothing special |
| 0-1 | PASS - retail or overpriced |

Recognizes: Aeron, Embody, Sayl, Mirra, Cosm, Setu, Eames, Celle, Lino, and more.

---

## Setup

### Requirements

- Python 3.8+
- [OpenRouter](https://openrouter.ai) account (for AI - has free tier)
- [Resend](https://resend.com) account (for emails - has free tier)
- Facebook account

### Install

```
git clone https://github.com/stfn-c/herman-miller-finder.git
cd herman-miller-finder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Windows: use `venv\Scripts\activate` instead.

### Configure

```
cp .env.example .env
```

Edit `.env` with your settings. See [Configuration](#configuration) below for all options.

### Get Facebook Cookies

The tool needs your Facebook session cookies. There's a script to grab them:

1. Go to [facebook.com](https://facebook.com) in Chrome
2. Make sure you're logged in
3. Press `F12` to open DevTools
4. Click the **Console** tab at the top
5. Paste this and hit Enter:

```javascript
(function() {
    const cookieNames = ['datr', 'sb', 'c_user', 'xs', 'fr', 'locale'];
    const cookies = [];
    document.cookie.split(';').forEach(cookie => {
        const [name, value] = cookie.trim().split('=');
        if (cookieNames.includes(name)) {
            cookies.push({name: name, value: value, domain: '.facebook.com', path: '/'});
        }
    });
    const output = 'FB_COOKIES=' + JSON.stringify(cookies);
    console.log('\n' + output + '\n');
    try { navigator.clipboard.writeText(output); console.log('Copied to clipboard!'); } catch(e) {}
})();
```

6. Copy the `FB_COOKIES=...` line into your `.env` file

Or copy the contents of `export_cookies.js` and paste that instead - same thing.

---

## Usage

### One-time scan

```
python find_herman_miller.py
```

### Run on schedule

Checks throughout the day (default: 12 times, 9am-2am):

```
python find_herman_miller.py --scheduler
```

### Options

| Flag | What it does |
|------|--------------|
| `-n 50` | Check 50 listings instead of default |
| `--test` | Test mode - uses sample images, no Facebook |
| `--verbose` | Show detailed output |
| `--benchmark` | Compare different AI models |

---

## Configuration

All settings go in your `.env` file. Only the API keys and cookies are required - everything else has defaults.

### Required

| Setting | Description |
|---------|-------------|
| `OPENROUTER_API_KEYS` | Your OpenRouter API key |
| `RESEND_API_KEY` | Your Resend API key |
| `FROM_EMAIL` | Sender email (verify domain in Resend) |
| `TO_EMAIL` | Where to send alerts |
| `FB_COOKIES` | Facebook session cookies (see above) |

### Location

| Setting | Default | Description |
|---------|---------|-------------|
| `MARKETPLACE_LOCATIONS` | `melbourne` | City slugs - comma separated for multiple (e.g., `perth,melbourne,sydney`) |
| `TIMEZONE` | `UTC` | Your timezone for scheduler |
| `LATITUDE` | `0` | Your latitude (for browser geolocation) |
| `LONGITUDE` | `0` | Your longitude |
| `LOCALE` | `en-US` | Browser locale |

To find your marketplace slug, go to Facebook Marketplace in your city and look at the URL:
```
facebook.com/marketplace/SLUG/search
                         ^^^^
```

Multiple cities are scanned one after another. In prod mode, there's a 30-90s delay between cities.

### Scheduler

| Setting | Default | Description |
|---------|---------|-------------|
| `RUNS_PER_DAY` | `12` | How many times to scan per day |
| `START_HOUR` | `9` | Start scanning at this hour (24h) |
| `END_HOUR` | `2` | Stop scanning at this hour (2 = 2am) |

### Alerts

| Setting | Default | Description |
|---------|---------|-------------|
| `MIN_DEAL_SCORE` | `0` | Only alert for deals >= this score (0-10) |
| `LISTING_COUNT` | `20` | How many listings to check per run |

Set `MIN_DEAL_SCORE=6` to only get notified about great deals or better.

### Browser

| Setting | Default | Description |
|---------|---------|-------------|
| `HEADLESS` | auto | `true` = invisible browser, `false` = show browser window |

By default it auto-detects: headless on servers, visible on desktop.

---

## Finding Your Location

### Timezone

Use a [tz database name](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones):
- `America/New_York`
- `America/Los_Angeles`
- `Europe/London`
- `Australia/Sydney`

### Coordinates

Google your city + "coordinates" or use [latlong.net](https://www.latlong.net/).

---

## Notes

- Facebook cookies expire. If it stops working, grab fresh cookies.
- Automated scraping probably violates Facebook's TOS. Your call.

## License

MIT
