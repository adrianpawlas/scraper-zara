#!/usr/bin/env python3
"""Run the Zara scraper - paste category product API URLs in api_urls.txt and run this script."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scraper import run_scraper

if __name__ == "__main__":
    stats = run_scraper()
    print(f"\nScrape complete: {stats['products_imported']} imported, {stats['errors']} errors")
