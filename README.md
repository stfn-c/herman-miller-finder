# Herman Miller Chair Finder

Automatically find Herman Miller chairs on Facebook Marketplace using AI-powered image analysis.

This tool scrapes Facebook Marketplace listings, uses vision-capable LLMs (Claude, GPT-4o, Gemini, etc.) to identify authentic Herman Miller chairs, calculates deal scores based on retail pricing, and sends email alerts for good finds.

## Features

- **AI-Powered Identification**: Uses Claude Opus 4.5 and other vision models via OpenRouter to analyze chair images
- **Deal Scoring**: Calculates value based on listing price vs retail (0-10 scale)
- **Email Alerts**: Sends notifications when high-value Herman Miller chairs are found
- **Benchmark Mode**: Compare accuracy across 14+ vision models
- **Scheduler**: Runs 12x daily during waking hours with randomized intervals
- **Persistent Storage**: SQLite database tracks all analyzed listings

## Supported Chair Models

Aeron, Sayl, Embody, Mirra, Mirra 2, Cosm, Setu, Eames Soft Pad, Eames Aluminum Group, Celle, Lino, Verus, Caper, and more.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/herman-miller-finder.git
cd herman-miller-finder

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

## Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEYS` | Comma-separated OpenRouter API keys |
| `RESEND_API_KEY` | Resend.com API key for email notifications |
| `FROM_EMAIL` | Sender email address (must be verified in Resend) |
| `TO_EMAIL` | Recipient email for alerts |
| `FB_COOKIES` | Facebook session cookies as JSON array |

### Getting Facebook Cookies

1. Log into Facebook in your browser
2. Open Developer Tools (F12) → Application → Cookies
3. Export these cookies: `datr`, `sb`, `c_user`, `xs`, `fr`, `locale`
4. Format as JSON array (see `.env.example`)

## Usage

```bash
# Standard run - scrape Facebook and analyze listings
python find_herman_miller.py

# Test mode - use embedded test images only (no Facebook scraping)
python find_herman_miller.py --test

# Production mode - slower, more human-like delays
python find_herman_miller.py --prod

# Dev mode - faster delays (default)
python find_herman_miller.py --dev

# Specify number of listings to scrape
python find_herman_miller.py -n 50

# Verbose or quiet logging
python find_herman_miller.py --verbose
python find_herman_miller.py --quiet

# Run as scheduler (12x/day, 9am-2am Perth time)
python find_herman_miller.py --scheduler
```

### Benchmarking

Compare vision model accuracy:

```bash
# Run benchmark across all models
python find_herman_miller.py --benchmark

# List previous benchmark runs
python find_herman_miller.py --list-benchmarks

# Compare two benchmark runs
python find_herman_miller.py --compare latest previous
```

## Deal Score System

| Score | Label | Price vs Retail |
|-------|-------|-----------------|
| 10 | FUMBLE | < 15% |
| 8-9 | STEAL | 15-25% |
| 6-7 | GREAT | 25-40% |
| 4-5 | GOOD | 40-60% |
| 2-3 | FAIR | 60-80% |
| 0-1 | RETAIL/OVERPRICED | > 80% |

## Output

Found chairs are saved to `found_chairs/`:
- `found_listings.db` - SQLite database of all analyzed listings
- `{listing_id}/info.json` - Listing details and AI analysis
- `{listing_id}/image.jpg` - Downloaded listing image
- `benchmark_*.html` - Benchmark comparison reports

## Models Tested

The benchmark compares these vision-capable models:
- Claude Opus 4.5, Claude Sonnet 4, Claude Sonnet 3.5
- GPT-4o, GPT-4o Mini
- Gemini 2.5 Pro, Gemini 2.0 Flash
- Llama 3.2 90B Vision
- Pixtral Large, Pixtral 12B
- Qwen 2.5 VL 72B
- And more...

## License

MIT License - See [LICENSE](LICENSE) for details.

## Disclaimer

This tool scrapes Facebook Marketplace, which may violate Facebook's Terms of Service. Use at your own risk and responsibility. The authors are not responsible for any account actions taken by Facebook.
