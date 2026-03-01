# Zara Scraper

Scrapes product data from Zara category product API URLs, generates image and text embeddings (768-dim SigLIP), and imports to Supabase.

## How it works

- **Category product URLs**: Each URL is like `https://www.zara.com/be/en/category/<categoryId>/products?ajax=true`. The scraper extracts `categoryId` from the URL and uses `config.ZARA_CATEGORIES` to assign category name and gender (man/woman) to every product from that page.
- **API format**: Zara returns JSON with `productGroups[].elements[].commercialComponents`; each component has id, name, price, images, and SEO slug for the product page URL.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API URLs**
   - Open `api_urls.txt`
   - Add Zara category product URLs (one per line). Example: `https://www.zara.com/be/en/category/2536906/products?ajax=true`
   - Category name and gender are resolved from `config.ZARA_CATEGORIES` by the category ID in the URL
   - For local testing you can use a file path (e.g. `sample2.txt`) if the file contains the same JSON structure

3. **Configure Supabase** (optional)
   - Edit `config.py` or set env vars: `SUPABASE_URL`, `SUPABASE_ANON_KEY`
   - Default values are pre-configured

## Usage

### Manual run
```bash
python run.py
```

Or with CLI options:
```bash
python -m scraper --skip-embeddings   # Skip embedding generation (faster testing)
python -m scraper --urls "https://www.zara.com/be/en/category/2536906/products?ajax=true"  # Use URLs directly
```

### Automated daily run
GitHub Actions runs the scraper daily at midnight UTC. Setup:

1. **API URLs**: Either commit URLs to `api_urls.txt`, or add secret `API_URLS` (newline-separated URLs)

2. **Supabase** (optional): Default credentials are in `config.py`. To override, add secrets:
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`

3. **Manual trigger**: Actions → "Run Zara Scraper" → Run workflow

## Output

Products are upserted to the `products` table with:
- `source`: "scraper"
- `brand`: "Zara"
- `gender`: "man" or "woman" (from category mapping)
- `category`: e.g. "Jackets & Gilets", "T-shirts"
- `image_embedding`: 768-dim from google/siglip-base-patch16-384
- `info_embedding`: 768-dim from SigLIP text encoder
