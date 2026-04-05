# Herman Miller Chair Finder

People sell Herman Miller chairs on Facebook Marketplace all the time without knowing what they have. I got tired of missing out on $200 Aerons, so I built this.

It scans Marketplace, uses AI vision to identify Herman Miller chairs from photos, scores the deal against retail, and emails you.

## How It Works

1. [CloakBrowser](https://github.com/CloakHQ/CloakBrowser) (stealth Chromium) scrapes Facebook Marketplace
2. AI analyzes each listing photo for Herman Miller chairs
3. Calculates deal score against retail price
4. Emails you the finds via [Resend](https://resend.com)
5. SQLite cache ensures no listing is ever processed twice

## Deal Scores

| Score | Meaning |
|-------|---------|
| 10 | FUMBLE - seller has no clue ($200 Aeron) |
| 8-9 | STEAL - way under market |
| 6-7 | GREAT - solid deal |
| 4-5 | GOOD - fair used price |
| 2-3 | FAIR - nothing special |
| 0-1 | PASS - retail or overpriced |

Recognizes: Aeron, Embody, Sayl, Mirra, Cosm, and more.

## Project Structure

```
find_herman_miller.py           # Entry point + CLI
src/
  config.py                     # Env loading, constants, CLI args
  scraper.py                    # CloakBrowser + Facebook scraping
  analyzer.py                   # AI image analysis (Anthropic / OpenRouter)
  database.py                   # SQLite listing cache + stats
  email_alert.py                # Resend email alerts
  benchmark.py                  # Multi-model accuracy benchmarking
  test_mode.py                  # Test images + test runner
  scheduler.py                  # Cron-like scheduler + lock management
scripts/
  export_cookies.js             # Browser console script to grab FB cookies
```

---

## Setup

### Requirements

- Python 3.8+
- One of:
  - [Anthropic](https://console.anthropic.com) API key (direct Claude access)
  - [OpenRouter](https://openrouter.ai) API key (multi-model, has free tier)
- [Resend](https://resend.com) account (for email alerts - has free tier)
- Facebook account

### Install

```
git clone https://github.com/stfn-c/herman-miller-finder.git
cd herman-miller-finder
pip install -r requirements.txt
```

CloakBrowser downloads its Chromium binary automatically on first run.

### Configure

```
cp .env.example .env
```

Edit `.env` with your settings. See [Configuration](#configuration) below.

### Get Facebook Cookies

1. Go to [facebook.com](https://facebook.com) in Chrome
2. Make sure you're logged in
3. Press `F12` > **Console** tab
4. Paste the contents of `scripts/export_cookies.js` and hit Enter
5. Copy the `FB_COOKIES=...` line into your `.env` file

---

## Usage

```
python find_herman_miller.py                     # dev mode (fast)
python find_herman_miller.py --prod              # prod mode (human-like delays + decoy searches)
python find_herman_miller.py --prod --scheduler  # run 12x/day on schedule
python find_herman_miller.py --backend anthropic # use Anthropic API directly
python find_herman_miller.py -n 50               # check 50 listings
python find_herman_miller.py --test              # test mode (sample images, no Facebook)
python find_herman_miller.py --benchmark         # compare AI models
```

| Flag | What it does |
|------|--------------|
| `--prod` | Slower, human-like delays + decoy searches |
| `--scheduler` | Run continuously (12x/day, 9am-2am) |
| `--backend` | `openrouter` (default) or `anthropic` |
| `-n NUM` | Number of listings to check (default: 20) |
| `--test` | Test with sample images, no Facebook |
| `--benchmark` | Compare AI model accuracy |
| `--verbose` | Detailed output |

---

## Configuration

All settings in `.env`. Only API keys + cookies are required.

### AI Backend

| Setting | Default | Description |
|---------|---------|-------------|
| `AI_BACKEND` | `openrouter` | `openrouter` or `anthropic` |
| `OPENROUTER_API_KEYS` | - | OpenRouter API key(s), comma-separated |
| `ANTHROPIC_API_KEY` | - | Anthropic API key (only if backend=anthropic) |
| `AI_MODEL` | `anthropic/claude-opus-4` | Model ID for analysis |

### Alerts

| Setting | Default | Description |
|---------|---------|-------------|
| `RESEND_API_KEY` | - | Resend API key |
| `FROM_EMAIL` | - | Sender email (verify domain in Resend) |
| `TO_EMAIL` | - | Where to send alerts |
| `MIN_DEAL_SCORE` | `0` | Only alert for deals >= this score (0-10) |

### Location

| Setting | Default | Description |
|---------|---------|-------------|
| `MARKETPLACE_LOCATIONS` | `melbourne` | City slug(s), comma-separated |
| `TIMEZONE` | `UTC` | Your timezone |
| `LATITUDE` / `LONGITUDE` | `0` | Browser geolocation |
| `LOCALE` | `en-US` | Browser locale |

Find your marketplace slug from the URL: `facebook.com/marketplace/SLUG/search`

### Scheduler

| Setting | Default | Description |
|---------|---------|-------------|
| `RUNS_PER_DAY` | `12` | Scans per day |
| `START_HOUR` | `9` | Start hour (24h) |
| `END_HOUR` | `2` | End hour (2 = 2am next day) |

### Browser

| Setting | Default | Description |
|---------|---------|-------------|
| `HEADLESS` | auto | `true` = invisible, `false` = show window |
| `LISTING_COUNT` | `20` | Listings per run |

---

## Notes

- Facebook cookies expire. Grab fresh ones if scraping stops working.
- A persistent browser profile (`.fb_profile/`) is maintained across runs so Facebook sees a consistent browser identity.
- All processed listings are cached in SQLite - re-runs skip already-seen listings.

## License

MIT
