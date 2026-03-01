"""Main Zara scraper - fetches category product APIs, extracts products, embeds, imports to Supabase."""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from supabase import create_client

from config import (
    API_URLS_FILE,
    BRAND,
    GENDER,
    SECOND_HAND,
    SOURCE,
    SUPABASE_ANON_KEY,
    SUPABASE_URL,
    ZARA_CATEGORIES,
    ZARA_CATEGORY_ID_RE,
)
from embeddings import get_image_embedding, get_text_embedding
from parsers import detect_api_type, parse_products_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_api_urls(file_path: Path) -> list[str]:
    """Load API URLs from file, one per line, ignoring empty lines and comments."""
    if not file_path.exists():
        return []

    urls = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def get_category_and_gender_for_url(url: str) -> tuple[str | None, str | None]:
    """Extract category ID from Zara category products URL and return (category_name, gender) from config."""
    m = ZARA_CATEGORY_ID_RE.search(url)
    if not m:
        return None, None
    cat_id = int(m.group(1))
    if cat_id not in ZARA_CATEGORIES:
        return None, None
    name, gender = ZARA_CATEGORIES[cat_id]
    return name, gender


def fetch_json(url_or_path: str) -> Optional[dict]:
    """Fetch JSON from URL or load from local file path."""
    path = Path(url_or_path)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Failed to load file %s: %s", path, e)
            return None

    try:
        resp = requests.get(
            url_or_path,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error("Failed to fetch %s: %s", url_or_path[:80], e)
        return None


def build_info_text(record: dict) -> str:
    """Build concatenated text for info_embedding."""
    parts = [
        record.get("title", ""),
        record.get("description", ""),
        record.get("category", ""),
        record.get("gender", ""),
        record.get("price", ""),
        record.get("sale", ""),
    ]
    metadata = record.get("metadata")
    if metadata:
        try:
            m = json.loads(metadata)
            if isinstance(m, dict):
                parts.append(json.dumps(m, default=str))
        except (json.JSONDecodeError, TypeError):
            parts.append(str(metadata))
    return " ".join(str(p) for p in parts if p).strip()


def record_to_db_row(record: dict, image_embedding: list[float] | None, info_embedding: list[float] | None) -> dict:
    """Convert parsed record to Supabase products table row."""
    row = {
        "id": record["id"],
        "source": SOURCE,
        "product_url": record["product_url"],
        "image_url": record["image_url"],
        "brand": BRAND,
        "title": record["title"],
        "description": record.get("description"),
        "category": record.get("category"),
        "gender": record.get("gender") or GENDER,
        "metadata": record.get("metadata"),
        "size": None,
        "second_hand": SECOND_HAND,
        "country": None,
        "tags": None,
        "other": None,
        "price": record.get("price"),
        "sale": record.get("sale"),
        "additional_images": record.get("additional_images"),
        "affiliate_url": None,
        "compressed_image_url": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if image_embedding:
        row["image_embedding"] = image_embedding
    if info_embedding:
        row["info_embedding"] = info_embedding

    return row


def run_scraper(
    api_urls: Optional[list[str]] = None,
    skip_embeddings: bool = False,
    limit: Optional[int] = None,
) -> dict:
    """
    Run the full scrape: fetch each category products API, parse products, generate embeddings, import to Supabase.
    Category name and gender are derived from the URL (category ID) via config.ZARA_CATEGORIES.
    Returns stats dict with counts.
    If limit is set, only that many products are processed (for testing).
    """
    urls = api_urls or load_api_urls(API_URLS_FILE)
    if not urls:
        logger.warning("No API URLs found. Paste Zara category product URLs in %s (one per line)", API_URLS_FILE)
        return {"products_parsed": 0, "products_imported": 0, "errors": 0}

    all_records: dict[str, dict] = {}  # id -> record (dedupe by id)

    for url in urls:
        category_name, gender_override = get_category_and_gender_for_url(url)
        logger.info("Fetching %s (category=%s, gender=%s)", url[:80], category_name or "?", gender_override or "?")
        data = fetch_json(url)
        if not data:
            continue

        try:
            api_type = detect_api_type(data)
        except ValueError as e:
            logger.warning("Skipping URL (unknown format): %s", e)
            continue

        if api_type == "products":
            records = parse_products_api(data, category_name=category_name, gender_override=gender_override)
            for r in records:
                all_records[r["id"]] = r
            logger.info("Parsed %d products from products API", len(records))

    if not all_records:
        logger.warning("No product records to import. Ensure at least one Zara category products URL in %s.", API_URLS_FILE)
        return {"products_parsed": 0, "products_imported": 0, "errors": 0}

    if limit is not None and limit > 0:
        items = list(all_records.items())[:limit]
        all_records = dict(items)
        logger.info("Limited to %d product(s) for this run", len(all_records))

    supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    imported = 0
    errors = 0

    for i, (pid, record) in enumerate(all_records.items(), 1):
        logger.info("Processing %d/%d: %s", i, len(all_records), record["title"][:50])

        image_embedding = None
        info_embedding = None

        if not skip_embeddings:
            image_embedding = get_image_embedding(record["image_url"])
            if not image_embedding:
                logger.warning("No image embedding for %s", pid)

            info_text = build_info_text(record)
            info_embedding = get_text_embedding(info_text)
            if not info_embedding:
                logger.warning("No info embedding for %s", pid)

        row = record_to_db_row(record, image_embedding, info_embedding)

        try:
            supabase.table("products").upsert(row, on_conflict="id").execute()
            imported += 1
        except Exception as e:
            logger.error("Failed to upsert %s: %s", pid, e)
            errors += 1

    logger.info("Done. Imported %d, errors %d", imported, errors)
    return {"products_parsed": len(all_records), "products_imported": imported, "errors": errors}


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Zara scraper")
    parser.add_argument("--urls", nargs="*", help="Category product API URLs (overrides api_urls.txt)")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation (faster)")
    parser.add_argument("--limit", type=int, default=None, help="Process only this many products (for testing)")
    args = parser.parse_args()

    run_scraper(api_urls=args.urls, skip_embeddings=args.skip_embeddings, limit=args.limit)


if __name__ == "__main__":
    main()
