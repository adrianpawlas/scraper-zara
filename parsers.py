"""Parse Zara API responses (category products JSON: productGroups with commercialComponents)."""
import json

from config import BASE_PRODUCT_URL, PRODUCT_URL_LOCALE


def detect_api_type(data: dict) -> str:
    """Detect API response type: Zara uses 'productGroups' with commercialComponents."""
    if "productGroups" in data:
        for group in data.get("productGroups", []):
            for elem in group.get("elements", []):
                if elem.get("commercialComponents"):
                    return "products"
        return "products"  # productGroups present even if empty
    raise ValueError("Unknown API response format")


def extract_product_ids_from_grid(data: dict) -> set[int]:
    """Extract product IDs from grid/category API. Zara uses productGroups; IDs are in commercialComponents."""
    product_ids = set()
    for group in data.get("productGroups", []):
        for elem in group.get("elements", []):
            for comp in elem.get("commercialComponents", []):
                pid = comp.get("id")
                if isinstance(pid, int):
                    product_ids.add(pid)
    return product_ids


def get_image_urls_from_zara_component(comp: dict) -> tuple[str | None, list[str]]:
    """
    Extract main image URL and additional image URLs from a Zara commercialComponent.
    Uses detail.colors[].xmedia[].extraInfo.deliveryUrl or .url (replace {width} with 1024).
    Returns (main_image_url, [additional_urls]).
    """
    main_url = None
    additional_urls: list[str] = []
    seen: set[str] = set()

    detail = comp.get("detail", {})
    colors = detail.get("colors", [])

    for color in colors:
        for xm in color.get("xmedia", []):
            url = (xm.get("extraInfo") or {}).get("deliveryUrl")
            if not url:
                url = xm.get("url", "").replace("{width}", "1024")
            if not url or not url.startswith("https://"):
                continue
            if url in seen:
                continue
            seen.add(url)
            if main_url is None:
                main_url = url
            else:
                additional_urls.append(url)

    if not main_url and additional_urls:
        main_url = additional_urls.pop(0)
    return main_url, additional_urls


def format_price(price_cents: str | int, currency: str = "EUR") -> str:
    """Convert price from cents to formatted string: 179.00EUR."""
    try:
        cents = int(price_cents)
        amount = cents / 100
        return f"{amount:.2f}{currency}"
    except (ValueError, TypeError):
        return ""


def build_zara_product_url(comp: dict) -> str:
    """Build Zara product page URL: base/locale/{keyword}-p{seoProductId}.html."""
    seo = comp.get("seo", {})
    keyword = (seo.get("keyword") or "").strip() or "product"
    seo_id = (seo.get("seoProductId") or "").strip()
    if not seo_id:
        # Fallback: use discernProductId for ID
        pid = comp.get("id")
        return f"{BASE_PRODUCT_URL}/{PRODUCT_URL_LOCALE}/{keyword}.html?v1={pid}" if pid else f"{BASE_PRODUCT_URL}/{PRODUCT_URL_LOCALE}/{keyword}.html"
    return f"{BASE_PRODUCT_URL}/{PRODUCT_URL_LOCALE}/{keyword}-p{seo_id}.html"


def parse_products_api(
    data: dict,
    category_name: str | None = None,
    gender_override: str | None = None,
) -> list[dict]:
    """
    Parse Zara category products API (productGroups[].elements[].commercialComponents) into flat product records.
    One record per product. category_name and gender_override come from the URL (config mapping).
    """
    records = []
    seen_ids: set[int] = set()

    for group in data.get("productGroups", []):
        for elem in group.get("elements", []):
            for comp in elem.get("commercialComponents", []):
                if comp.get("type") != "Product":
                    continue

                product_id = comp.get("id")
                if product_id is None or product_id in seen_ids:
                    continue
                seen_ids.add(product_id)

                main_image, additional_images = get_image_urls_from_zara_component(comp)
                if not main_image:
                    continue

                name = comp.get("name") or "Unknown"
                price_cents = comp.get("price")
                price_str = format_price(price_cents) if price_cents is not None else None
                # Zara API doesn't expose oldPrice in this view; sale can be added later if available
                sale_str = None
                if comp.get("extraInfo", {}).get("highlightPrice"):
                    sale_str = price_str  # placeholder if we find sale logic elsewhere

                gender = gender_override or (comp.get("sectionName") or "").strip().lower()
                if gender == "man":
                    gender = "man"
                elif gender == "woman":
                    gender = "woman"
                else:
                    gender = "man"

                product_url = build_zara_product_url(comp)
                detail = comp.get("detail", {})
                ref = detail.get("reference") or detail.get("displayReference") or ""

                # Category: from URL mapping, or fallback to product family/subfamily (e.g. when using file path)
                category = category_name or comp.get("familyName") or comp.get("subfamilyName") or None

                metadata = json.dumps(
                    {
                        "product_id": product_id,
                        "reference": ref,
                        "seoProductId": comp.get("seo", {}).get("seoProductId"),
                        "familyName": comp.get("familyName"),
                        "subfamilyName": comp.get("subfamilyName"),
                    },
                    default=str,
                )

                records.append(
                    {
                        "id": f"zara_{product_id}",
                        "product_id": product_id,
                        "product_url": product_url,
                        "gender": gender,
                        "image_url": main_image,
                        "additional_images": " , ".join(additional_images) if additional_images else None,
                        "title": name,
                        "description": comp.get("description") or None,
                        "category": category,
                        "price": price_str,
                        "sale": sale_str,
                        "metadata": metadata,
                    }
                )

    return records
