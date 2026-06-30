import argparse
import gzip
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from fit_kit.core.schemas import Category

# ══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

DEFAULT_PATH = "data/raw/meta_Amazon_Fashion.jsonl.gz"
DEFAULT_REVIEWS_PATH = "data/raw/review_categories/Amazon_Fashion.jsonl"
DEFAULT_OUTPUT = "data/catalog"
MIN_RATING = 3.5
MIN_REVIEWS = 5

# catalog schema
CATALOG_SCHEMA = {
    # identifiers
    "asin": "str    — Amazon product ID (primary key)",
    "title": "str    — raw product title",
    "rich_text": "str    — enriched text (title + features + details, ordered by importance) used for text embedding",
    # raw product data
    "features": "list[str] — product feature bullets (native list, variable length)",
    "details_json": "str    — product details dict as JSON (inconsistent schema across products)",
    "store": "str    — seller/brand name",
    "price": "float? — product price in USD (nullable, only ~6% coverage)",
    "average_rating": "float  — average star rating (≥4.0, pre-filtered)",
    "rating_number": "int    — number of reviews (≥10, pre-filtered)",
    # image data
    "image_url": "str    — source image URL",
    "image_path": "str    — local cached image path (data/catalog/images/{asin}.jpg)",
    "has_image": "bool   — whether image was successfully downloaded",
    # classified attributes (FashionSigLIP zero-shot)
    "clip_category": "str    — product category (21 labels, CLIP on enriched text)",
    "clip_gender": "str    — target demographic (6 labels: women/men/girls/boys/unisex_adults/unisex_kids, regex+CLIP hybrid)",
    "gender_source": "str    — how gender was determined: 'regex' (keyword match) or 'clip' (model fallback)",
    "clip_season": "str?   — seasonality (4 labels, nullable for non-clothing categories)",
    "clip_formality": "str?   — formality level (5 labels, nullable for non-clothing, soft signal)",
    "clip_color": "str    — primary color from text (15 labels, CLIP on enriched text)",
    "clip_color_image": "str?   — primary color from image (15 labels, nullable if no image)",
    # review enrichment (optional, from --reviews-path)
    "review_text": "str    — concatenated review titles + body text (verified, ≥3★, max 10 per product)",
}

CATALOG_SCHEMA_STR = "\n".join(f"  {col:22} {desc}" for col, desc in CATALOG_SCHEMA.items())

# gender regex (from explore_dataset.py)
_RE_WOMEN = re.compile(r"\b(women'?s?|woman'?s?|ladies|female)\b", re.I)
_RE_MEN = re.compile(r"\b(men'?s?|man'?s?|male)\b", re.I)
_RE_GIRLS = re.compile(r"\b(girls?'?s?)\b", re.I)
_RE_BOYS = re.compile(r"\b(boys?'?s?)\b", re.I)
_RE_UNISEX = re.compile(r"\b(unisex)\b", re.I)
_RE_KIDS = re.compile(
    r"\b(toddler|infant|baby|babies|kid|kids|child|children|newborn|nursery)\b", re.I
)

# attribute prompts (from explore_dataset.py)
CLIP_CATEGORIES = [
    ("tops", "a shirt, blouse, tee, or top"),
    ("dresses", "a dress, gown, romper, or jumpsuit"),
    ("sweaters", "a sweater, hoodie, cardigan, or sweatshirt"),
    ("pants", "pants, jeans, trousers, or leggings"),
    ("skirts", "a skirt"),
    ("shorts", "shorts"),
    ("activewear", "athletic wear, gym clothes, yoga pants, or sportswear"),
    ("swimwear", "a swimsuit, bikini, or swim trunks"),
    ("outerwear", "a jacket, coat, blazer, vest, or suit"),
    ("sleepwear", "pajamas, sleepwear, a robe, or loungewear"),
    ("underwear", "underwear, bra, lingerie, or shapewear"),
    ("socks", "socks, hosiery, tights, or stockings"),
    ("shoes", "shoes, boots, sneakers, sandals, heels, or slippers"),
    ("bags", "a handbag, purse, backpack, wallet, or tote"),
    ("jewelry", "jewelry, a necklace, bracelet, earrings, or ring"),
    ("watches", "a watch or watch band"),
    ("sunglasses", "sunglasses, eyeglasses, or eyewear"),
    ("belts", "a belt or suspenders"),
    ("scarves", "a scarf, shawl, wrap, or bandana"),
    ("hats", "a hat, cap, beanie, or headwear"),
    ("accessories", "gloves, a tie, hair accessories, or a face mask"),
]

CLIP_GENDER = [
    ("women", "women's clothing or fashion"),
    ("men", "men's clothing or fashion"),
    ("girls", "girls' clothing for children"),
    ("boys", "boys' clothing for children"),
    ("unisex_adults", "unisex adult clothing for men and women"),
    ("unisex_kids", "unisex children's clothing for kids boys and girls"),
]

CLIP_SEASON = [
    ("summer", "lightweight summer clothing for warm weather"),
    ("winter", "warm winter clothing for cold weather"),
    ("spring_fall", "transitional spring or fall layering clothing"),
    ("all_season", "basic year-round all-season clothing"),
]

CLIP_FORMALITY = [
    ("casual", "casual relaxed everyday clothing"),
    ("business_casual", "business casual smart office clothing"),
    ("formal", "formal dressy evening or wedding clothing"),
    ("athletic", "athletic sportswear gym workout clothing"),
    ("loungewear", "comfortable lounge or sleepwear clothing"),
]

CLIP_COLOR = [
    ("black", "black colored clothing"),
    ("white", "white colored clothing"),
    ("blue", "blue colored clothing"),
    ("red", "red colored clothing"),
    ("pink", "pink colored clothing"),
    ("green", "green colored clothing"),
    ("yellow", "yellow colored clothing"),
    ("orange", "orange colored clothing"),
    ("purple", "purple colored clothing"),
    ("brown", "brown or tan colored clothing"),
    ("grey", "grey or silver colored clothing"),
    ("navy", "navy blue colored clothing"),
    ("beige", "beige cream or off-white clothing"),
    ("gold", "gold or metallic colored clothing"),
    ("multicolor", "multicolor patterned or printed clothing"),
]

CLOTHING_CATEGORIES = {
    "tops",
    "dresses",
    "sweaters",
    "pants",
    "skirts",
    "shorts",
    "activewear",
    "swimwear",
    "outerwear",
    "sleepwear",
    "underwear",
    "socks",
    "shoes",
    "hats",
    "scarves",
}

CONF_THRESHOLDS = {
    "season": (None, 15),
    "formality": (None, 15),
    "color": (None, 10),
}

DETAIL_KEYS_FOR_TEXT = [
    "Department",
    "Material",
    "Color",
    "Style",
    "Fabric Type",
    "Closure Type",
    "Pattern",
]


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════


def parse_details(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {}
    return {}


def get_all_image_urls(p):
    """Get all available image URLs for a product, priority order (hi_res → large)."""
    images = p.get("images")
    if not images:
        return []
    urls = []
    if isinstance(images, list):
        for img in images:
            if isinstance(img, dict):
                for key in ["hi_res", "large"]:
                    u = img.get(key)
                    if u and u not in urls:
                        urls.append(u)
            elif isinstance(img, str) and img not in urls:
                urls.append(img)
    elif isinstance(images, dict):
        for key in ["hi_res", "large"]:
            for u in images.get(key) or []:
                if u and u not in urls:
                    urls.append(u)
    return urls


def regex_gender(title):
    """Extract gender from title keywords. Returns (label, 'regex') or (None, None).
    Distinguishes adult unisex from kids unisex using child indicator words."""
    has_women = bool(_RE_WOMEN.search(title))
    has_men = bool(_RE_MEN.search(title))
    has_girls = bool(_RE_GIRLS.search(title))
    has_boys = bool(_RE_BOYS.search(title))
    is_kids = bool(_RE_KIDS.search(title))

    # explicit "unisex" keyword
    if _RE_UNISEX.search(title):
        return ("unisex_kids" if is_kids else "unisex_adults"), "regex"

    # both adult genders
    if has_women and has_men:
        return "unisex_adults", "regex"

    # both child genders, or one child gender + kids indicator
    if has_girls and has_boys:
        return "unisex_kids", "regex"

    # single gender matches
    if has_women:
        return "women", "regex"
    if has_men:
        return ("unisex_kids" if is_kids else "men"), "regex"
    if has_girls:
        return "girls", "regex"
    if has_boys:
        return "boys", "regex"

    # kids indicator without gender → unisex_kids
    if is_kids:
        return "unisex_kids", "regex"

    return None, None


def build_rich_text(prod):
    """Build enriched text from title + features + details, ordered by importance.
    No word limit — the SigLIP tokenizer truncates to 64 tokens internally.
    Title comes first (always included), then features, then details."""
    parts = [prod["title"]]
    for f in prod.get("features", [])[:3]:
        if isinstance(f, str) and f.strip():
            parts.append(f.strip())
    details = parse_details(prod.get("details"))
    for key in DETAIL_KEYS_FOR_TEXT:
        val = details.get(key)
        if val and isinstance(val, str) and val.strip():
            parts.append(f"{key}: {val.strip()}")
    return ". ".join(parts)


# ══════════════════════════════════════════════════════════════════════════
#  STEP 1: LOAD & FILTER
# ══════════════════════════════════════════════════════════════════════════


def load_and_filter(path, min_rating=4.0, min_reviews=10, max_products=None):
    """Load dataset and filter to high-quality products."""
    print(f"loading {path} (min_rating={min_rating}, min_reviews={min_reviews})...")
    products = []
    total = 0
    skipped_no_title = 0
    skipped_quality = 0

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
            except Exception:
                continue
            total += 1

            if total % 100_000 == 0:
                print(f"  scanned {total:,} records, kept {len(products):,}...")

            title = (p.get("title") or "").strip()
            if not title:
                skipped_no_title += 1
                continue

            rating = p.get("average_rating")
            rating_count = p.get("rating_number") or 0
            if rating is None or float(rating) < min_rating or int(rating_count) < min_reviews:
                skipped_quality += 1
                continue

            image_urls = get_all_image_urls(p)

            prod = {
                "asin": p.get("parent_asin", ""),
                "title": title,
                "features": p.get("features") or [],
                "details": p.get("details") or {},
                "store": (p.get("store") or "").strip(),
                "price": _parse_price(p.get("price")),
                "average_rating": float(rating),
                "rating_number": int(rating_count),
                "image_urls": image_urls,
                "image_url": image_urls[0] if image_urls else "",
            }
            products.append(prod)

            if max_products and len(products) >= max_products:
                break

    print(f"  total scanned: {total:,}")
    print(f"  skipped (no title): {skipped_no_title:,}")
    print(f"  skipped (quality): {skipped_quality:,}")
    print(f"  kept: {len(products):,} high-quality products")
    return products


def _parse_price(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if v > 0 else None
    if isinstance(v, str):
        v = v.replace("$", "").replace(",", "").strip()
        try:
            val = float(v.split()[0])
            return val if val > 0 else None
        except (ValueError, IndexError):
            return None
    return None


# ══════════════════════════════════════════════════════════════════════════
#  STEP 2: CLASSIFY ATTRIBUTES (batched)
# ══════════════════════════════════════════════════════════════════════════


def classify_all(products, model, tokenizer, device, batch_size=128):
    """Classify all attributes using batched encoding."""
    import torch

    n = len(products)
    print(f"\n{'=' * 70}")
    print(f" CLASSIFYING {n:,} PRODUCTS")
    print(f"{'=' * 70}")

    # encode attribute prompts
    print("encoding attribute prompts...")
    attr_features = {}
    for attr_name, prompts in [
        ("category", CLIP_CATEGORIES),
        ("gender", CLIP_GENDER),
        ("season", CLIP_SEASON),
        ("formality", CLIP_FORMALITY),
        ("color", CLIP_COLOR),
    ]:
        names = [name for name, _ in prompts]
        texts = [text for _, text in prompts]
        tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        attr_features[attr_name] = (names, feats.cpu())

    # batch-encode enriched texts
    print(f"encoding enriched texts (batches of {batch_size})...")
    rich_texts = [build_rich_text(p) for p in products]
    all_rich_features = _batch_encode_texts(
        rich_texts, model, tokenizer, device, batch_size, "rich text"
    )

    # classify category from rich text
    print("classifying categories...")
    cat_names, cat_feats = attr_features["category"]
    cat_sims = 100.0 * all_rich_features @ cat_feats.T  # (N, 21)
    cat_indices = cat_sims.argmax(dim=-1)  # (N,)
    cat_confs = cat_sims.gather(1, cat_indices.unsqueeze(1)).squeeze(1)

    for i, p in enumerate(products):
        p["clip_category"] = cat_names[cat_indices[i].item()]
        p["clip_category_conf"] = cat_confs[i].item()

    # classify gender: regex first, CLIP fallback
    print("classifying gender (regex + CLIP hybrid)...")
    needs_clip = []
    for i, p in enumerate(products):
        label, src = regex_gender(p["title"])
        if label is not None:
            p["clip_gender"] = label
            p["gender_source"] = "regex"
        else:
            needs_clip.append(i)
            p["gender_source"] = "clip"

    if needs_clip:
        print(f"  regex handled {n - len(needs_clip):,}, CLIP needed for {len(needs_clip):,}")
        titles_for_gender = [products[i]["title"] for i in needs_clip]
        gender_features = _batch_encode_texts(
            titles_for_gender, model, tokenizer, device, batch_size, "gender titles"
        )
        gen_names, gen_feats = attr_features["gender"]
        gen_sims = 100.0 * gender_features @ gen_feats.T
        gen_indices = gen_sims.argmax(dim=-1)
        for j, idx in enumerate(needs_clip):
            products[idx]["clip_gender"] = gen_names[gen_indices[j].item()]

    # classify season, formality, color from rich text
    for attr_name, display in [
        ("season", "season"),
        ("formality", "formality"),
        ("color", "color"),
    ]:
        print(f"classifying {display}...")
        names, feats = attr_features[attr_name]
        sims = 100.0 * all_rich_features @ feats.T
        indices = sims.argmax(dim=-1)
        confs = sims.gather(1, indices.unsqueeze(1)).squeeze(1)

        clothing_only = attr_name in ("season", "formality")
        threshold_info = CONF_THRESHOLDS.get(attr_name)

        for i, p in enumerate(products):
            if clothing_only and p["clip_category"] not in CLOTHING_CATEGORIES:
                p[f"clip_{attr_name}"] = None
                continue
            label = names[indices[i].item()]
            conf = confs[i].item()
            if threshold_info:
                fallback, thresh = threshold_info
                if conf < thresh:
                    label = fallback
            p[f"clip_{attr_name}"] = label

    # convert all_season → null (year-round items pass through any season filter)
    for p in products:
        if p.get("clip_season") == "all_season":
            p["clip_season"] = None

    # validate categories match schema
    schema_categories = {c.value for c in Category}
    catalog_categories = {name for name, _ in CLIP_CATEGORIES}
    if schema_categories != catalog_categories:
        missing = schema_categories - catalog_categories
        extra = catalog_categories - schema_categories
        raise ValueError(
            f"Category mismatch between schema and CLIP prompts. "
            f"Missing from CLIP: {missing}. Extra in CLIP: {extra}"
        )

    # store rich_text for reference
    for i, p in enumerate(products):
        p["rich_text"] = rich_texts[i]

    # print distributions
    _print_distributions(products)

    return all_rich_features


def _batch_encode_texts(texts, model, tokenizer, device, batch_size, label="text"):
    """Batch-encode texts, return normalized features tensor on CPU."""
    import torch

    all_features = []
    n = len(texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = texts[start:end]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_features.append(feats.cpu())
        if (end % (batch_size * 10) == 0) or end == n:
            print(f"  {label}: {end:,}/{n:,}")
    return torch.cat(all_features, dim=0)


def _print_distributions(products):
    """Print attribute distributions."""
    from collections import Counter

    n = len(products)
    print(f"\n┌- ATTRIBUTE DISTRIBUTIONS ({n:,} products) --------------")

    for attr, label in [
        ("clip_category", "category"),
        ("clip_gender", "gender"),
        ("clip_season", "season"),
        ("clip_formality", "formality"),
        ("clip_color", "color"),
    ]:
        dist = Counter(p.get(attr) for p in products if p.get(attr) is not None)
        total = sum(dist.values())
        print(f"│  -- {label.upper()} --")
        for val, count in dist.most_common(10):
            print(f"│    {val:16} {count:>6,} ({count / total:.1%})")
        if len(dist) > 10:
            print(f"│    ... and {len(dist) - 10} more")
        print("│")

    # gender source breakdown
    from_regex = sum(1 for p in products if p.get("gender_source") == "regex")
    from_clip = sum(1 for p in products if p.get("gender_source") == "clip")
    print(f"│  gender source: regex={from_regex:,} clip={from_clip:,}")
    print("└--------------------------------------------------------------")


# ══════════════════════════════════════════════════════════════════════════
#  STEP 2.5: ENRICH WITH REVIEWS (optional)
# ══════════════════════════════════════════════════════════════════════════


def enrich_with_reviews(products, reviews_path):
    """Load reviews and store raw review text per product.

    BM25 handles term weighting natively — no need to extract keywords.
    Keeps the top 10 most-helpful reviews per product (by helpful_vote).
    Filters to verified purchases with rating ≥ 3 to reduce noise.
    """
    import heapq

    MAX_REVIEWS_PER_PRODUCT = 10

    print(f"\n{'=' * 70}")
    print(" ENRICHING WITH REVIEWS")
    print(f"{'=' * 70}")

    asin_to_idx = {p["asin"]: i for i, p in enumerate(products)}
    asin_set = set(asin_to_idx.keys())

    # min-heap per product: (helpful_vote, review_text)
    # keeps top N most-helpful reviews in bounded memory
    review_heaps = {asin: [] for asin in asin_set}

    print(f"  loading {reviews_path}...")
    total_reviews = 0
    matched_reviews = 0
    skipped = 0

    # handle both .jsonl and .jsonl.gz
    opener = gzip.open if reviews_path.endswith(".gz") else open

    with opener(reviews_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            total_reviews += 1

            asin = r.get("parent_asin", "")
            if asin not in asin_set:
                continue

            # quality filter
            if not r.get("verified_purchase", False) or (r.get("rating") or 0) < 3:
                skipped += 1
                continue

            matched_reviews += 1

            title = (r.get("title") or "").strip()
            body = (r.get("text") or "").strip()
            review_text = f"{title}. {body}" if title and body else (title or body)
            if not review_text:
                continue

            helpful = r.get("helpful_vote") or 0
            heap = review_heaps[asin]

            if len(heap) < MAX_REVIEWS_PER_PRODUCT:
                heapq.heappush(heap, (helpful, review_text))
            elif helpful > heap[0][0]:
                heapq.heapreplace(heap, (helpful, review_text))

            if total_reviews % 500_000 == 0:
                print(f"  scanned {total_reviews:,} reviews, matched {matched_reviews:,}...")

    print(f"  total reviews: {total_reviews:,}")
    print(f"  matched (verified, ≥3★): {matched_reviews:,}")
    print(f"  skipped: {skipped:,}")

    # store concatenated review text per product (sorted most-helpful first)
    products_with_reviews = 0
    for asin, idx in asin_to_idx.items():
        heap = review_heaps[asin]
        if heap:
            products_with_reviews += 1
            sorted_reviews = [text for _, text in sorted(heap, reverse=True)]
            products[idx]["review_text"] = " ".join(sorted_reviews)
        else:
            products[idx]["review_text"] = ""

    avg_reviews = matched_reviews / max(products_with_reviews, 1)
    print(f"  products with reviews: {products_with_reviews:,}")
    print(f"  avg reviews per product: {avg_reviews:.1f}")

    # samples
    samples = [p for p in products if len(p.get("review_text", "")) > 100][:2]
    for p in samples:
        rt = p["review_text"]
        print(f"\n  -- {p['title'][:70]} --")
        print(f"    review text ({len(rt)} chars): {rt[:200]}...")


# ══════════════════════════════════════════════════════════════════════════
#  STEP 3: DOWNLOAD IMAGES
# ══════════════════════════════════════════════════════════════════════════


def download_images(products, output_dir, num_workers=16):
    """Download product images in parallel with resume support."""
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    to_download = []
    for p in products:
        if not p.get("image_urls"):
            p["image_path"] = ""
            p["has_image"] = False
            continue
        ext = ".jpg"
        path = images_dir / f"{p['asin']}{ext}"
        p["image_path"] = str(path)
        if path.exists():
            p["has_image"] = True
        else:
            p["has_image"] = False
            to_download.append(p)

    already = len(products) - len(to_download) - sum(1 for p in products if not p.get("image_urls"))
    no_url = sum(1 for p in products if not p.get("image_urls"))
    print(f"\n{'=' * 70}")
    print(" DOWNLOADING IMAGES")
    print(f"{'=' * 70}")
    print(f"  total products: {len(products):,}")
    print(f"  no image URL: {no_url:,}")
    print(f"  already cached: {already:,}")
    print(f"  to download: {len(to_download):,}")

    if not to_download:
        print("  nothing to download")
        return

    success = 0
    failed = 0
    start_time = time.time()

    def _download_one(prod, max_retries=3):
        for url in prod.get("image_urls", []):
            for attempt in range(max_retries):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        data = resp.read()
                    with open(prod["image_path"], "wb") as f:
                        f.write(data)
                    prod["image_url"] = url
                    return True
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
                    continue
            # all retries for this URL failed, try next resolution
        return False

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_download_one, p): p for p in to_download}
        for i, future in enumerate(as_completed(futures)):
            prod = futures[future]
            try:
                ok = future.result()
                if ok:
                    prod["has_image"] = True
                    success += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

            done = success + failed
            if done % 1000 == 0 or done == len(to_download):
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                print(
                    f"  {done:,}/{len(to_download):,} "
                    f"(ok={success:,} fail={failed:,}) "
                    f"{rate:.0f} img/s"
                )

    total_images = sum(1 for p in products if p["has_image"])
    print(f"  done: {total_images:,} images available ({total_images / len(products):.1%})")


# ══════════════════════════════════════════════════════════════════════════
#  STEP 4: ENCODE IMAGES
# ══════════════════════════════════════════════════════════════════════════


def encode_images(products, model, preprocess, device, batch_size=64, color_attr_data=None):
    """Batch-encode images with FashionSigLIP. Optionally classify color from image."""
    import torch
    from PIL import Image

    # filter to products with images
    with_images = [(i, p) for i, p in enumerate(products) if p.get("has_image")]
    n = len(with_images)
    embed_dim = None

    print(f"\n{'=' * 70}")
    print(f" ENCODING IMAGES ({n:,} products)")
    print(f"{'=' * 70}")

    if n == 0:
        print("  no images to encode")
        return None, []

    all_features = []
    image_indices = []  # track which product indices have image embeddings
    failed = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_items = with_images[start:end]

        images = []
        valid_indices = []
        for idx, prod in batch_items:
            try:
                img = Image.open(prod["image_path"]).convert("RGB")
                img_tensor = preprocess(img)
                images.append(img_tensor)
                valid_indices.append(idx)
            except Exception:
                failed += 1
                prod["has_image"] = False

        if not images:
            continue

        batch_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        if embed_dim is None:
            embed_dim = feats.shape[1]

        # classify color from image if attr data provided
        if color_attr_data is not None:
            color_names, color_feats = color_attr_data
            color_sims = 100.0 * feats @ color_feats.to(device).T
            color_indices = color_sims.argmax(dim=-1)
            for j, idx in enumerate(valid_indices):
                products[idx]["clip_color_image"] = color_names[color_indices[j].item()]

        all_features.append(feats.cpu())
        image_indices.extend(valid_indices)

        done = min(end, n)
        if done % (batch_size * 10) == 0 or done == n:
            print(f"  images: {done:,}/{n:,} (failed: {failed:,})")

    if not all_features:
        return None, []

    all_features = torch.cat(all_features, dim=0)
    print(f"  encoded: {all_features.shape[0]:,} images, dim={all_features.shape[1]}")
    print(f"  failed: {failed:,}")
    return all_features, image_indices


# ══════════════════════════════════════════════════════════════════════════
#  STEP 5: SAVE EVERYTHING
# ══════════════════════════════════════════════════════════════════════════


def save_catalog(products, text_embeddings, image_embeddings, image_indices, output_dir):
    """Save catalog parquet, embeddings as safetensors, and stats."""
    import polars as pl
    import torch
    from safetensors.torch import save_file

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f" SAVING TO {out}")
    print(f"{'=' * 70}")

    # build DataFrame
    rows = []
    for i, p in enumerate(products):
        rows.append(
            {
                "asin": p["asin"],
                "title": p["title"],
                "rich_text": p.get("rich_text", ""),
                "features": [f for f in p.get("features", []) if isinstance(f, str)],
                "details_json": json.dumps(parse_details(p.get("details", {}))),
                "store": p.get("store", ""),
                "price": p.get("price"),
                "average_rating": p["average_rating"],
                "rating_number": p["rating_number"],
                "image_url": p.get("image_url", ""),
                "image_path": p.get("image_path", ""),
                "has_image": p.get("has_image", False),
                "clip_category": p.get("clip_category"),
                "clip_gender": p.get("clip_gender"),
                "gender_source": p.get("gender_source"),
                "clip_season": p.get("clip_season"),
                "clip_formality": p.get("clip_formality"),
                "clip_color": p.get("clip_color"),
                "clip_color_image": p.get("clip_color_image"),
                "review_text": p.get("review_text", ""),
            }
        )

    df = pl.DataFrame(rows)
    parquet_path = out / "catalog.parquet"

    # embed schema description in parquet file metadata
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = df.to_arrow()
    metadata = table.schema.metadata or {}
    metadata[b"catalog_schema"] = CATALOG_SCHEMA_STR.encode("utf-8")
    metadata[b"min_rating"] = str(MIN_RATING).encode("utf-8")
    metadata[b"min_reviews"] = str(MIN_REVIEWS).encode("utf-8")
    metadata[b"source"] = b"Amazon Fashion (McAuley-Lab/Amazon-Reviews-2023)"
    metadata[b"model"] = b"Marqo/marqo-fashionSigLIP"
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, str(parquet_path))
    print(f"  catalog: {parquet_path} ({len(df):,} rows, schema in file metadata)")

    # save text embeddings
    if text_embeddings is not None:
        text_path = out / "text_embeddings.safetensors"
        save_file({"embeddings": text_embeddings}, str(text_path))
        print(f"  text embeddings: {text_path} {tuple(text_embeddings.shape)}")

    # save image embeddings with index mapping
    if image_embeddings is not None and len(image_indices) > 0:
        img_path = out / "image_embeddings.safetensors"
        save_file(
            {
                "embeddings": image_embeddings,
                "indices": torch.tensor(image_indices, dtype=torch.int64),
            },
            str(img_path),
        )
        print(f"  image embeddings: {img_path} {tuple(image_embeddings.shape)}")
        print(f"  image index mapping: {len(image_indices):,} entries")

    # save stats
    from collections import Counter

    stats = {
        "total_products": len(products),
        "has_image": sum(1 for p in products if p.get("has_image")),
        "text_embedding_shape": list(text_embeddings.shape)
        if text_embeddings is not None
        else None,
        "image_embedding_shape": list(image_embeddings.shape)
        if image_embeddings is not None
        else None,
        "category_dist": dict(Counter(p.get("clip_category") for p in products)),
        "gender_dist": dict(Counter(p.get("clip_gender") for p in products)),
        "gender_source_dist": dict(Counter(p.get("gender_source") for p in products)),
    }
    stats_path = out / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  stats: {stats_path}")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description="Prepare enriched product catalog")
    ap.add_argument("--path", default=DEFAULT_PATH, help="path to .jsonl.gz")
    ap.add_argument(
        "--reviews-path", default=DEFAULT_REVIEWS_PATH, help="path to reviews .jsonl or .jsonl.gz"
    )
    ap.add_argument("--no-reviews", action="store_true", help="skip review enrichment")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="output directory")
    ap.add_argument("--device", default="cpu", help="torch device (cpu/cuda/cuda:1)")
    ap.add_argument("--batch-size", type=int, default=128, help="encoding batch size")
    ap.add_argument("--image-batch-size", type=int, default=64, help="image encoding batch size")
    ap.add_argument("--download-workers", type=int, default=16, help="parallel image downloads")
    ap.add_argument("--max-products", type=int, default=None, help="limit products (for testing)")
    ap.add_argument("--skip-images", action="store_true", help="skip image download and encoding")
    ap.add_argument("--min-rating", type=float, default=MIN_RATING, help="minimum average rating")
    ap.add_argument("--min-reviews", type=int, default=MIN_REVIEWS, help="minimum review count")
    ap.add_argument("--schema", action="store_true", help="print catalog schema and exit")
    args = ap.parse_args()

    if args.schema:
        print("CATALOG SCHEMA (catalog.parquet)")
        print("=" * 70)
        print(CATALOG_SCHEMA_STR)
        print()
        print("EMBEDDINGS")
        print("=" * 70)
        print("  text_embeddings.safetensors:")
        print("    embeddings    (N, D) float32 — FashionSigLIP enriched-text embeddings")
        print("    aligned by row index with catalog.parquet")
        print()
        print("  image_embeddings.safetensors:")
        print("    embeddings    (M, D) float32 — FashionSigLIP image embeddings")
        print("    indices       (M,)   int64   — maps embedding row to catalog row index")
        sys.exit(0)

    start = time.time()

    # step 1: load & filter
    products = load_and_filter(
        args.path,
        min_rating=args.min_rating,
        min_reviews=args.min_reviews,
        max_products=args.max_products,
    )
    if not products:
        print("no products found")
        sys.exit(1)

    # step 2: load model & classify
    try:
        import open_clip
        import torch
    except ImportError:
        print("ERROR: requires: pip install torch open_clip_torch ftfy polars safetensors")
        sys.exit(1)

    print(f"\nloading FashionSigLIP on {args.device}...")
    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    model = model.to(args.device).eval()

    text_embeddings = classify_all(
        products, model, tokenizer, args.device, batch_size=args.batch_size
    )

    # step 2.5: enrich with reviews (optional)
    if not args.no_reviews and args.reviews_path and os.path.exists(args.reviews_path):
        enrich_with_reviews(products, args.reviews_path)
    elif not args.no_reviews and args.reviews_path and not os.path.exists(args.reviews_path):
        print(f"\n  reviews file not found: {args.reviews_path} (skipping review enrichment)")

    # step 3 & 4: images
    image_embeddings = None
    image_indices = []

    if not args.skip_images:
        download_images(products, args.output_dir, num_workers=args.download_workers)

        # prepare color attr data for image-based color classification
        color_names = [n for n, _ in CLIP_COLOR]
        color_texts = [t for _, t in CLIP_COLOR]
        color_tokens = tokenizer(color_texts).to(args.device)
        with torch.no_grad():
            color_feats = model.encode_text(color_tokens)
            color_feats = color_feats / color_feats.norm(dim=-1, keepdim=True)
        color_attr_data = (color_names, color_feats.cpu())

        image_embeddings, image_indices = encode_images(
            products,
            model,
            preprocess,
            args.device,
            batch_size=args.image_batch_size,
            color_attr_data=color_attr_data,
        )

    # step 5: save
    save_catalog(products, text_embeddings, image_embeddings, image_indices, args.output_dir)

    elapsed = time.time() - start
    print(f"\n✓ done in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
