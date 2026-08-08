"""Main Zara scraper - fetches category product APIs, extracts products, embeds, imports to Supabase."""
import json
import logging
import sys
import time
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

BATCH_SIZE = 50
EMBEDDING_DELAY = 0.5
MAX_RETRIES = 3
STALE_THRESHOLD_RUNS = 2
FAILED_PRODUCTS_LOG = Path(__file__).parent / "failed_products.log"
STALE_TRACKER_FILE = Path(__file__).parent / "stale_tracker.json"


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


def load_stale_tracker() -> dict[str, int]:
    """Load stale product tracker from local file."""
    if STALE_TRACKER_FILE.exists():
        try:
            return json.loads(STALE_TRACKER_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load stale tracker: %s", e)
    return {}


def save_stale_tracker(tracker: dict[str, int]):
    """Save stale product tracker to local file."""
    try:
        STALE_TRACKER_FILE.write_text(json.dumps(tracker, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error("Failed to save stale tracker: %s", e)


def fetch_existing_products(supabase) -> dict[str, dict]:
    """Fetch all existing products for this source from Supabase."""
    try:
        response = supabase.table("products").select(
            "id, product_url, image_url, title, description, category, gender, price, sale, metadata, additional_images, created_at, image_embedding, info_embedding"
        ).eq("source", SOURCE).execute()
        
        existing = {}
        for row in response.data:
            key = row["product_url"]
            existing[key] = row
        logger.info("Fetched %d existing products from database", len(existing))
        return existing
    except Exception as e:
        logger.error("Failed to fetch existing products: %s", e)
        return {}


def has_product_changed(existing: dict, new_record: dict) -> bool:
    """Check if any meaningful fields have changed."""
    fields_to_check = [
        ("title", str),
        ("description", str),
        ("category", str),
        ("gender", str),
        ("price", str),
        ("sale", str),
        ("image_url", str),
        ("additional_images", list),
    ]
    
    for field, field_type in fields_to_check:
        existing_val = existing.get(field)
        new_val = new_record.get(field)
        
        if field_type in (str,):
            existing_val = existing_val or ""
            new_val = new_val or ""
        elif field_type == list:
            existing_val = existing_val or []
            new_val = new_val or []
        
        if existing_val != new_val:
            return True
    
    return False


def log_failed_products(failed: list[dict]):
    """Log failed products to a local file."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(FAILED_PRODUCTS_LOG, "a", encoding="utf-8") as f:
        for product in failed:
            f.write(f"{timestamp} - {product['product_url']}: {product.get('error', 'Unknown error')}\n")


def insert_batch_with_retry(supabase, batch: list[dict]) -> tuple[int, list[dict]]:
    """Insert a batch of products with retry logic. Returns (success_count, failed_products)."""
    failed_products = []
    
    for attempt in range(MAX_RETRIES):
        try:
            supabase.table("products").upsert(
                batch, on_conflict="source, product_url"
            ).execute()
            return len(batch), []
        except Exception as e:
            logger.warning("Batch insert attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, e)
            if attempt == MAX_RETRIES - 1:
                for product in batch:
                    failed_products.append({
                        "product_url": product.get("product_url"),
                        "error": str(e)
                    })
    
    return len(batch) - len(failed_products), failed_products


def update_stale_tracker_and_delete(
    supabase, 
    product_urls_seen: set[str], 
    existing_products: dict[str, dict],
    stale_tracker: dict[str, int]
) -> tuple[int, dict[str, int]]:
    """Update stale tracker and delete products that have been missed for 2+ consecutive runs."""
    deleted = 0
    updated_tracker = {}
    
    for product_url in existing_products:
        if product_url in product_urls_seen:
            updated_tracker[product_url] = 0
        else:
            current_misses = stale_tracker.get(product_url, 0) + 1
            updated_tracker[product_url] = current_misses
            
            if current_misses >= STALE_THRESHOLD_RUNS:
                try:
                    supabase.table("products").delete().eq("id", existing_products[product_url]["id"]).execute()
                    logger.info("Deleted stale product: %s", product_url[:60])
                    deleted += 1
                except Exception as e:
                    logger.error("Failed to delete stale product %s: %s", product_url[:60], e)
    
    return deleted, updated_tracker


def process_product(record: dict, existing_product: dict | None, generate_embeddings: bool, embedding_delay: float) -> dict | None:
    """Process a single product, determining if embeddings need to be regenerated."""
    needs_embedding = False
    image_embedding = None
    info_embedding = None
    
    if existing_product is None:
        needs_embedding = True
    elif existing_product.get("image_url") != record.get("image_url"):
        needs_embedding = True
    
    if generate_embeddings and needs_embedding:
        time.sleep(embedding_delay)
        
        image_embedding = get_image_embedding(record["image_url"])
        if not image_embedding:
            logger.warning("No image embedding for %s", record["id"])
        
        time.sleep(embedding_delay)
        
        info_text = build_info_text(record)
        info_embedding = get_text_embedding(info_text)
        if not info_embedding:
            logger.warning("No info embedding for %s", record["id"])
    elif existing_product is not None:
        image_embedding = existing_product.get("image_embedding")
        info_embedding = existing_product.get("info_embedding")
    
    existing_created_at = existing_product.get("created_at") if existing_product is not None else None
    return record_to_db_row(record, image_embedding, info_embedding, needs_embedding or existing_product is None, existing_created_at)


def record_to_db_row(record: dict, image_embedding: list[float] | None, info_embedding: list[float] | None, include_embeddings: bool, created_at: str | None = None) -> dict:
    """Convert parsed record to Supabase products table row.

    created_at is preserved from the existing row when present so updates
    don't bump the original creation time (the table has no updated_at column).
    """
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
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }

    if include_embeddings:
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
        return {"products_parsed": 0, "products_imported": 0, "products_updated": 0, "products_skipped": 0, "products_deleted": 0, "errors": 0}

    all_records: dict[str, dict] = {}

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
                all_records[r["product_url"]] = r
            logger.info("Parsed %d products from products API", len(records))

    if not all_records:
        logger.warning("No product records to import. Ensure at least one Zara category products URL in %s.", API_URLS_FILE)
        return {"products_parsed": 0, "products_imported": 0, "products_updated": 0, "products_skipped": 0, "products_deleted": 0, "errors": 0}

    if limit is not None and limit > 0:
        items = list(all_records.items())[:limit]
        all_records = dict(items)
        logger.info("Limited to %d product(s) for this run", len(all_records))

    supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    
    existing_products = fetch_existing_products(supabase)
    stale_tracker = load_stale_tracker()
    
    product_urls_seen = set(all_records.keys())
    
    new_products_added = 0
    products_updated = 0
    products_skipped = 0
    total_errors = 0
    failed_products = []
    
    batch_to_insert = []
    
    for i, (product_url, record) in enumerate(all_records.items(), 1):
        logger.info("Processing %d/%d: %s", i, len(all_records), record["title"][:50])
        
        existing = existing_products.get(product_url)
        
        if existing is None:
            processed = process_product(record, None, not skip_embeddings, EMBEDDING_DELAY)
            if processed:
                batch_to_insert.append(processed)
                new_products_added += 1
        else:
            changed = has_product_changed(existing, record)
            
            if changed:
                processed = process_product(record, existing, not skip_embeddings, EMBEDDING_DELAY)
                if processed:
                    batch_to_insert.append(processed)
                    products_updated += 1
            else:
                products_skipped += 1
                logger.debug("Skipping unchanged product: %s", product_url[:60])
        
        if len(batch_to_insert) >= BATCH_SIZE:
            success_count, batch_failed = insert_batch_with_retry(supabase, batch_to_insert)
            total_errors += len(batch_failed)
            failed_products.extend(batch_failed)
            batch_to_insert = []
    
    if batch_to_insert:
        success_count, batch_failed = insert_batch_with_retry(supabase, batch_to_insert)
        total_errors += len(batch_failed)
        failed_products.extend(batch_failed)
    
    products_deleted, updated_tracker = update_stale_tracker_and_delete(
        supabase, product_urls_seen, existing_products, stale_tracker
    )
    save_stale_tracker(updated_tracker)
    
    if failed_products:
        log_failed_products(failed_products)
        logger.warning("Logged %d failed products to %s", len(failed_products), FAILED_PRODUCTS_LOG)
    
    logger.info("=" * 60)
    logger.info("SCRAPER RUN SUMMARY")
    logger.info("=" * 60)
    logger.info("New products added: %d", new_products_added)
    logger.info("Products updated: %d", products_updated)
    logger.info("Products unchanged (skipped): %d", products_skipped)
    logger.info("Stale products deleted: %d", products_deleted)
    logger.info("Errors: %d", total_errors)
    logger.info("=" * 60)
    
    return {
        "products_parsed": len(all_records),
        "products_imported": new_products_added,
        "products_updated": products_updated,
        "products_skipped": products_skipped,
        "products_deleted": products_deleted,
        "errors": total_errors,
    }


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Zara scraper")
    parser.add_argument("--urls", nargs="*", help="Category product API URLs (overrides api_urls.txt)")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation (faster)")
    parser.add_argument("--limit", type=int, default=None, help="Process only this many products (for testing)")
    args = parser.parse_args()

    stats = run_scraper(api_urls=args.urls, skip_embeddings=args.skip_embeddings, limit=args.limit)
    
    print(f"\nScrape complete:")
    print(f"  {stats['products_imported']} new products added")
    print(f"  {stats['products_updated']} products updated")
    print(f"  {stats['products_skipped']} products unchanged (skipped)")
    print(f"  {stats['products_deleted']} stale products deleted")
    print(f"  {stats['errors']} errors")


if __name__ == "__main__":
    main()
