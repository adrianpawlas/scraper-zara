"""Configuration for Zara scraper."""
import os
import re
from pathlib import Path

# Path to API URLs file - paste your category product URLs here (one per line)
API_URLS_FILE = Path(__file__).parent / "api_urls.txt"

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://yqawmzggcgpeyaaynrjk.supabase.co")
SUPABASE_ANON_KEY = os.getenv(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4",
)

# Product defaults
SOURCE = "scraper-zara"
BRAND = "Zara"
GENDER = "man"
SECOND_HAND = False

# Base URLs for Zara
BASE_PRODUCT_URL = "https://www.zara.com"
IMAGE_BASE_URL = "https://static.zara.net"
# Product URL locale (country/language)
PRODUCT_URL_LOCALE = "be/en"

# Category ID -> (category_name, gender) for URLs like .../category/2536906/products?ajax=true
ZARA_CATEGORIES: dict[int, tuple[str, str]] = {
    # MEN
    2536906: ("Jackets & Gilets", "man"),
    2431994: ("Shirts", "man"),
    2431956: ("Linen", "man"),
    2431959: ("Linen", "man"),
    2432042: ("T-shirts", "man"),
    2432131: ("Jeans", "man"),
    2432096: ("Trousers", "man"),
    2642257: ("Suits", "man"),
    2432232: ("Hoodies & Sweatshirts", "man"),
    2432265: ("Sweaters & Cardigans", "man"),
    2436828: ("Leather", "man"),
    2432280: ("Overshirts", "man"),
    2432049: ("Polo shirts", "man"),
    2432164: ("Shorts", "man"),
    2436311: ("Blazers", "man"),
    2436920: ("Tracksuits", "man"),
    2436948: ("Summery Garments", "man"),
    2436382: ("Shoes", "man"),
    2436405: ("Bags", "man"),
    2606626: ("Underwear & Socks", "man"),
    2436444: ("Accessories", "man"),
    2436468: ("Perfumes", "man"),
    2436513: ("Makeup", "man"),
    2436823: ("Special prices", "man"),
    # WOMEN
    2417772: ("Jackets", "woman"),
    2420942: ("Blazers", "woman"),
    2419032: ("Trench Coats", "woman"),
    2419844: ("Cardigans & Jumpers", "woman"),
    2420417: ("T-shirts", "woman"),
    2419940: ("Tops", "woman"),
    2420490: ("Bodies", "woman"),
    2420369: ("Shirts", "woman"),
    2419185: ("Jeans", "woman"),
    2420795: ("Trousers", "woman"),
    2420896: ("Dresses & Jumpsuits", "woman"),
    2420306: ("Knitwear", "woman"),
    2418883: ("Leather", "woman"),
    2420454: ("Skirts", "woman"),
    2420480: ("Shorts & Bermuda shorts", "woman"),
    2420285: ("Co-ord sets", "woman"),
    2467841: ("Sweatshirts & Joggers", "woman"),
    2419807: ("Lingerie", "woman"),
    2419737: ("Special Prices", "woman"),
    2419160: ("Shoes", "woman"),
    2417728: ("Bags", "woman"),
    2418989: ("Accessories & Jewelry", "woman"),
    2419833: ("Perfumes", "woman"),
    2418919: ("Makeup", "woman"),
}

# Regex to extract category ID from Zara category products URL
ZARA_CATEGORY_ID_RE = re.compile(r"/category/(\d+)/products")

# Currency mapping by country (ISO 4217) - kept for compatibility
COUNTRY_TO_CURRENCY = {
    "BELGIUM": "EUR",
    "SPAIN": "EUR",
    "FRANCE": "EUR",
    "ITALY": "EUR",
    "GERMANY": "EUR",
    "UNITED KINGDOM": "GBP",
    "UNITED STATES": "USD",
    "NETHERLANDS": "EUR",
}
