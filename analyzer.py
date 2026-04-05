"""AI analysis: image analysis via OpenRouter or Anthropic SDK, response parsing, deal scoring."""

import json
import time
import base64
import random
import re
import requests
from datetime import datetime
from config import (
    OPENROUTER_API_KEY,
    VERBOSE_LOGGING,
    DEFAULT_MODEL,
    OUTPUT_DIR,
    AI_BACKEND,
    ANTHROPIC_API_KEY,
)


def analyze_image_with_model(
    image_base64,
    model_id="anthropic/claude-opus-4",
    listing_title="",
    listing_price="",
    api_key=None,
):
    """Use specified model via OpenRouter to analyze if chair is Herman Miller."""

    url = "https://openrouter.ai/api/v1/chat/completions"
    key = api_key or OPENROUTER_API_KEY

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://yaupicks.com",
        "X-Title": "Herman Miller Finder",
    }

    # Build context from listing info (price excluded - irrelevant to identification)
    listing_context = ""
    if listing_title:
        listing_context = f'\n\nListing title: "{listing_title}"\n(Ignore the title for identification - sellers often mislabel chairs. Judge ONLY by visual features.)'

    # Simple HM-only prompt
    prompt_text = f"""Is this a Herman Miller chair?

Herman Miller models:
- AERON: Mesh with horizontal bands, curved figure-8 frame, PostureFit lumbar
- EMBODY: Pixelated spine-like back
- MIRRA: Butterfly-shaped frame
- SAYL: Y-shaped suspension back
- COSM: Continuous flowing frame

Answer in JSON:
{{"reasoning":"why or why not","model":"Aeron|Embody|Mirra|Sayl|Cosm|None","confidence":"high|medium|low","is_herman_miller":true|false}}"""

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "max_tokens": 4000,
    }

    max_retries = 4
    base_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                original_content = content  # Keep for logging

                # Clean up the response - handle markdown, control chars, etc.
                content = content.strip()

                # Remove markdown code blocks
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                # Remove control characters (except newlines/tabs)
                content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", content)

                # Try to extract JSON object from response (in case there's extra text)
                json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)

                # Fix common JSON issues
                content = content.replace("\n", " ").replace("\r", "")
                # Fix unescaped quotes in reasoning field
                content = re.sub(
                    r'("reasoning"\s*:\s*")(.*?)(",\s*"(?:brand|model|confidence|is_premium|is_herman_miller))',
                    lambda m: m.group(1) + m.group(2).replace('"', "'") + m.group(3),
                    content,
                    flags=re.DOTALL,
                )

                # Try to parse JSON, handle truncated/malformed responses
                try:
                    parsed = json.loads(content)
                    if VERBOSE_LOGGING:
                        brand = parsed.get("brand", parsed.get("model", "Unknown"))
                        is_p = parsed.get(
                            "is_premium", parsed.get("is_herman_miller", False)
                        )
                        print(
                            f"      \u2713 {model_id.split('/')[-1]}: {brand} (premium={is_p})"
                        )
                    return parsed
                except json.JSONDecodeError as e:
                    # Log the parse failure
                    print(
                        f"      \u26a0\ufe0f {model_id.split('/')[-1]} JSON parse failed: {str(e)[:50]}"
                    )
                    if VERBOSE_LOGGING:
                        print(f"         Raw response: {original_content[:150]}...")

                    # Try to extract key fields from malformed response
                    is_premium_match = re.search(
                        r'"is_premium"\s*:\s*(true|false)', content, re.IGNORECASE
                    )
                    is_hm_match = re.search(
                        r'"is_herman_miller"\s*:\s*(true|false)', content, re.IGNORECASE
                    )
                    brand_match = re.search(r'"brand"\s*:\s*"([^"]*)"', content)
                    model_match = re.search(r'"model"\s*:\s*"([^"]*)"', content)
                    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)', content)
                    conf_match = re.search(r'"confidence"\s*:\s*"([^"]*)"', content)

                    # Determine is_premium from either field
                    is_premium = False
                    if is_premium_match:
                        is_premium = is_premium_match.group(1).lower() == "true"
                    elif is_hm_match:
                        is_premium = is_hm_match.group(1).lower() == "true"

                    if is_premium_match or is_hm_match or brand_match:
                        salvaged = {
                            "reasoning": f"[SALVAGED] {reason_match.group(1) if reason_match else 'parse error'}",
                            "brand": brand_match.group(1) if brand_match else "Unknown",
                            "model": model_match.group(1) if model_match else "Unknown",
                            "confidence": conf_match.group(1) if conf_match else "low",
                            "is_premium": is_premium,
                            "is_herman_miller": is_hm_match.group(1).lower() == "true"
                            if is_hm_match
                            else False,
                            "salvaged": True,
                        }
                        print(
                            f"      \U0001f527 {model_id.split('/')[-1]} SALVAGED: {salvaged.get('brand')} {salvaged.get('model')}"
                        )
                        return salvaged

                    # Can't salvage - return minimal response
                    print(
                        f"      \u274c {model_id.split('/')[-1]} FAILED: couldn't parse or salvage"
                    )
                    return {
                        "reasoning": f"[PARSE ERROR] {content[:100]}...",
                        "brand": "Unknown",
                        "model": "None",
                        "confidence": "low",
                        "is_premium": False,
                        "error": True,
                    }
            elif response.status_code == 429:
                # Rate limited - retry with exponential backoff
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                if attempt < max_retries - 1:
                    print(
                        f"      \u23f3 {model_id.split('/')[-1]} rate limited, retry {attempt + 1}/{max_retries} in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    print(
                        f"      \u274c {model_id.split('/')[-1]} rate limited after {max_retries} retries"
                    )
                    return {
                        "error": True,
                        "reasoning": "Rate limited",
                        "brand": "Unknown",
                        "model": "None",
                        "is_premium": False,
                    }
            elif response.status_code >= 500:
                # Server error - retry
                if attempt < max_retries - 1:
                    print(
                        f"      \u23f3 {model_id.split('/')[-1]} server error {response.status_code}, retrying..."
                    )
                    time.sleep(base_delay * (2**attempt))
                    continue
                else:
                    print(
                        f"      \u274c {model_id.split('/')[-1]} server error: {response.status_code}"
                    )
                    return {
                        "error": True,
                        "reasoning": f"Server error {response.status_code}",
                        "brand": "Unknown",
                        "model": "None",
                        "is_premium": False,
                    }
            else:
                print(
                    f"      \u274c {model_id.split('/')[-1]} API error: {response.status_code} - {response.text[:100]}"
                )
                return {
                    "error": True,
                    "reasoning": f"API error {response.status_code}",
                    "brand": "Unknown",
                    "model": "None",
                    "is_premium": False,
                }
        except requests.exceptions.Timeout:
            print(
                f"      \u23f3 {model_id.split('/')[-1]} timeout, retry {attempt + 1}/{max_retries}..."
            )
            if attempt < max_retries - 1:
                time.sleep(base_delay)
                continue
            print(
                f"      \u274c {model_id.split('/')[-1]} timed out after {max_retries} retries"
            )
            return {
                "error": True,
                "reasoning": "Timeout",
                "brand": "Unknown",
                "model": "None",
                "is_premium": False,
            }
        except Exception as e:
            print(
                f"      \u274c {model_id.split('/')[-1]} error: {type(e).__name__}: {str(e)[:50]}"
            )
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
                continue
            return {
                "error": True,
                "reasoning": f"{type(e).__name__}",
                "brand": "Unknown",
                "model": "None",
                "is_premium": False,
            }

    return {
        "error": True,
        "reasoning": "Unknown error",
        "brand": "Unknown",
        "model": "None",
        "is_premium": False,
    }


def _analyze_image_anthropic(
    image_base64,
    model_id="claude-sonnet-4-20250514",
    listing_title="",
    listing_price="",
):
    """Analyze image using the Anthropic Python SDK directly."""
    try:
        import anthropic
    except ImportError:
        print(
            "      ERROR: 'anthropic' package not installed. Run: pip install anthropic"
        )
        return {
            "error": True,
            "reasoning": "anthropic package not installed",
            "model": "None",
            "is_herman_miller": False,
        }

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt_text = """Is this a Herman Miller chair?

Herman Miller models:
- AERON: Mesh with horizontal bands, curved figure-8 frame, PostureFit lumbar
- EMBODY: Pixelated spine-like back
- MIRRA: Butterfly-shaped frame
- SAYL: Y-shaped suspension back
- COSM: Continuous flowing frame

Answer in JSON:
{"reasoning":"why or why not","model":"Aeron|Embody|Mirra|Sayl|Cosm|None","confidence":"high|medium|low","is_herman_miller":true|false}"""

    anthropic_model = model_id
    if "/" in anthropic_model:
        anthropic_model = anthropic_model.split("/")[-1]

    max_retries = 3
    base_delay = 5

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=anthropic_model,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
            )

            content = response.content[0].text.strip()

            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", content)
            json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if json_match:
                content = json_match.group(0)

            parsed = json.loads(content)
            if VERBOSE_LOGGING:
                model_name = parsed.get("model", "Unknown")
                is_hm = parsed.get("is_herman_miller", False)
                print(
                    f"      [anthropic] {anthropic_model}: {model_name} (is_hm={is_hm})"
                )
            return parsed

        except json.JSONDecodeError:
            return {
                "reasoning": f"[PARSE ERROR] {content[:100]}",
                "model": "None",
                "confidence": "low",
                "is_herman_miller": False,
                "error": True,
            }
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                print(f"      [anthropic] rate limited, retry in {delay}s...")
                time.sleep(delay)
                continue
            return {
                "error": True,
                "reasoning": "Rate limited",
                "model": "None",
                "is_herman_miller": False,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(base_delay)
                continue
            return {
                "error": True,
                "reasoning": str(e)[:100],
                "model": "None",
                "is_herman_miller": False,
            }

    return {
        "error": True,
        "reasoning": "Unknown error",
        "model": "None",
        "is_herman_miller": False,
    }


def analyze_image_with_claude(image_base64):
    """Route to configured backend - OpenRouter or Anthropic SDK."""
    if AI_BACKEND == "anthropic":
        return _analyze_image_anthropic(image_base64, DEFAULT_MODEL)
    return analyze_image_with_model(image_base64, DEFAULT_MODEL)


def save_herman_miller_listing(listing, analysis, image_data):
    """Save a Herman Miller listing to the output folder."""

    listing_dir = (
        OUTPUT_DIR / f"{listing['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    listing_dir.mkdir(parents=True, exist_ok=True)

    # Save listing info
    info = {
        "id": listing["id"],
        "url": listing["url"],
        "title": listing.get("title", "Unknown"),
        "price": listing.get("price", "Unknown"),
        "analysis": analysis,
        "found_at": datetime.now().isoformat(),
    }

    with open(listing_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Save image
    if image_data:
        with open(listing_dir / "image.jpg", "wb") as f:
            f.write(base64.b64decode(image_data))

    print(f"  \U0001f4be Saved to: {listing_dir}")
    return listing_dir


def analyze_single_model(args):
    """Worker function for parallel model testing."""
    model_id, model_name, cost, image_base64, title, price, api_key = args
    start_time = time.time()
    try:
        analysis = analyze_image_with_model(
            image_base64, model_id, title, price, api_key
        )
        elapsed = time.time() - start_time
        return {
            "model_id": model_id,
            "model_name": model_name,
            "cost": cost,
            "analysis": analysis,
            "elapsed_seconds": round(elapsed, 2),
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "model_id": model_id,
            "model_name": model_name,
            "cost": cost,
            "analysis": None,
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e),
        }
