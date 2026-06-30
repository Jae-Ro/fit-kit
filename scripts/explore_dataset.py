import argparse
import gzip
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from io import BytesIO
from statistics import mean, median

DATASET_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/meta_categories/meta_Amazon_Fashion.jsonl.gz"
)
DEFAULT_PATH = "data/raw/meta_Amazon_Fashion.jsonl.gz"


# ══════════════════════════════════════════════════════════════════════════
#  DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════


def do_download(path):
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1e6
        print(f"already exists: {path} ({mb:.0f} MB)")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"downloading {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, path, reporthook=_progress)
    print(f"\nsaved: {path}")


def _progress(block, block_size, total):
    done = block * block_size
    pct = done / total * 100 if total > 0 else 0
    sys.stdout.write(f"\r  {done / 1e6:.0f} MB ({pct:.0f}%)")
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════
#  REGEX CATEGORY CLASSIFIER (v2 — fixes from validation)
# ══════════════════════════════════════════════════════════════════════════

GENDER_PATTERNS = {
    "women": re.compile(r"\b(women'?s?|woman'?s?|ladies|female|girls?)\b", re.I),
    "men": re.compile(r"\b(men'?s?|male|boys?)\b", re.I),
    "unisex": re.compile(r"\b(unisex|kids?|children'?s?|toddler|infant|baby)\b", re.I),
}

# ORDER MATTERS — checked top to bottom, first match wins.
# Specific categories first to prevent false positives from general keywords.
CATEGORY_RULES = [
    # specific first (prevent "swimsuit"→suits, "nightgown"→dresses, etc)
    (
        "sunglasses",
        [
            r"\bsun\s?glasses?\b",
            r"\beyewear\b",
            r"\baviator\b(?!.*jacket)",
            r"\breading\s+glasses\b",
            r"\bblue\s+light\s+glasses\b",
            r"\boptical\s+frame\b",
        ],
    ),
    (
        "watches",
        [
            r"\bwatch(?:es)?\b",
            r"\bwatch\s?band\b",
            r"\bwatch\s?strap\b",
            r"\bsmartwatch\b",
            r"\btimepiece\b",
            r"\bquartz\b",
        ],
    ),
    (
        "swimwear",
        [
            r"\bswimsuit\b",
            r"\bswimwear\b",
            r"\bbikini\b",
            r"\bswim\s?trunk\b",
            r"\bswim\s?short\b",
            r"\bswim\b",
            r"\bboardshort\b",
            r"\brash\s?guard\b",
            r"\bbathing\s?suit\b",
        ],
    ),
    (
        "sleepwear",
        [
            r"\bpajama\b",
            r"\bpyjama\b",
            r"\bsleepwear\b",
            r"\bnightgown\b",
            r"\bnighti?e\b",
            r"\bnightshirt\b",
            r"\bbathrobe\b",
            r"\bsleep\s?shirt\b",
            r"\bpj\b",
            r"\bloungew?ear\b",
            r"\blounger\b",
        ],
    ),
    (
        "underwear",
        [
            r"\bunderwear\b",
            r"\bboxer\b(?!.*puppy)",
            r"\bbriefs?\b",
            r"\bpanties\b",
            r"\bpanty\b",
            r"\bthong\b(?!.*sandal)",
            r"\bbra\b(?!.*celet|.*nd|.*ss|.*ve|.*nch)",
            r"\bbralette\b",
            r"\blingerie\b",
            r"\bundershirt\b",
            r"\bshapewear\b",
            r"\bcorset\b",
        ],
    ),
    (
        "socks",
        [
            r"\bsocks?\b",
            r"\bhosiery\b",
            r"\bstockings?\b",
            r"\bknee.?high\b(?!.*boot|.*dress)",
            r"\btights\b(?!.*running|.*workout|.*yoga|.*athletic|.*compression|.*sport)",
            r"\bleg\s?warmer\b",
        ],
    ),
    (
        "activewear",
        [
            r"\bactivewear\b",
            r"\bathleisure\b",
            r"\byoga\s+(?:pant|legging|short|set|outfit)\b",
            r"\bgym\s+(?:short|pant|set|outfit|wear)\b",
            r"\bsport(?:s)?\s+(?:bra|wear|suit)\b(?!.*watch)",
            r"\bworkout\s+(?:set|outfit|wear)\b",
            r"\brunning\s+(?:short|tight|outfit|set)\b",
        ],
    ),
    # main clothing
    (
        "dresses",
        [
            r"\bdress(?:es)?\b(?!\s*(?:shoe|belt|shirt|sock|pant|code|form|ing|er\b|ed\b))",
            r"\bgown\b(?!.*night)",
            r"\bromper\b",
            r"\bjumpsuit\b",
            r"\bplaysuit\b",
        ],
    ),
    (
        "tops",
        [
            r"\bshirt\b(?!.*dress)",
            r"\bblouse\b",
            r"\bt-?shirt\b",
            r"\btee\b(?!th)",
            r"\btank\s?top\b",
            r"\bcrop\s?top\b",
            r"\bpeplum\b",
            r"\bcamisole\b",
            r"\bcami\b",
            r"\bhenley\b",
            r"\btunic\b",
            r"\bpolo\b(?!\s+ralph)",
            r"(?<!flat )(?<!low )(?<!high )\btop\b(?!\s*(?:quality|rated|seller|notch|coat|hat|load|grain|soil|seed|shelf|tier|stitch|handle))",
        ],
    ),
    (
        "sweaters",
        [
            r"\bsweater\b",
            r"\bcardigan\b",
            r"\bpullover\b",
            r"\bhoodie\b",
            r"\bsweatshirt\b",
            r"\bfleece\b(?=.*(?:jacket|pullover|vest|sweater|hoodie|sweatshirt|zip|quarter))",
        ],
    ),
    (
        "pants",
        [
            r"\bpants\b",
            r"\btrousers\b",
            r"\bjeans\b(?!.*\bcaps?\b)",
            r"\bleggings?\b",
            r"\bjeggings?\b",
            r"\bchinos?\b(?!.*\bcaps?\b|\b.*\bhat\b)",
            r"\bslacks?\b",
            r"\bjogg(?:er|ing)\b(?!.*shoe|.*sneaker)",
            r"\bcapri\b(?=.*(?:pant|legging|jean|short|trouser|crop)|\s+\b(?:pant|legging))",
            r"\bwide\s?leg\b(?!.*short)",
            r"\brunning\s+tights?\b",
            r"\byoga\s+tights?\b",
        ],
    ),
    (
        "shorts",
        [
            r"\bshorts\b",
            r"\bshort\b(?=.*(?:men|women|boy|girl|athletic|cargo|swim|running|gym|casual|denim|jean))",
        ],
    ),
    (
        "skirts",
        [
            r"\bskirt\b",
            r"\bskort\b",
        ],
    ),
    (
        "outerwear",
        [
            # jackets + coats + blazers + suits + vests merged
            r"\bjacket\b",
            r"\bblazer\b",
            r"\bwindbreaker\b",
            r"\bparka\b",
            r"\bgilet\b",
            r"\banorak\b",
            r"\bbomber\b(?!.*hat)",
            r"\bcoat\b(?!.*coating|.*of\s+arms)",
            r"\bovercoat\b",
            r"\btrench\b(?!.*er)",
            r"\bpeacoat\b",
            r"\bduster\b(?!.*cloth)",
            r"\bvest\b(?!.*life|.*harvest)",
            r"\btuxedo\b",
            r"\bsuit\b(?!.*swim|.*case|.*luggage|.*ed\b|.*\bkey\b|.*\bphone\b|.*\bfor\s+(?:ford|toyota|honda|nissan|car|iphone|samsung|pixel)\b)",
            r"\bwaistcoat\b",
        ],
    ),
    # footwear
    (
        "shoes",
        [
            r"\bshoes?\b",
            r"\bsneakers?\b",
            r"\bboots?\b(?!.*cut|.*leg|.*camp|.*strap)",
            r"\bsandals?\b",
            r"\bflats\b",
            r"\bballet\s?flat\b",
            r"\bheels?\b(?!.*pain|.*spur)",
            r"\bpumps?\b(?!.*bottle|.*water)",
            r"\bloafers?\b",
            r"\bslippers?\b",
            r"\bslide\b(?!.*r\b)",
            r"\bflip.?flops?\b",
            r"\bmoccasins?\b",
            r"\boxfords?\b",
            r"\bclogs?\b",
            r"\bchukka\b",
            r"\bderby\b(?!.*hat)",
            r"\bespadrilles?\b",
            r"\bwedges?\b(?=.*(?:shoe|sandal|heel|women|platform))",
        ],
    ),
    # accessories
    (
        "bags",
        [
            r"\bbags?\b(?!.*sleep|.*bean|.*ice|.*tea|.*trash|.*garbage|.*punching)",
            r"\bpurse\b",
            r"\bhandbag\b",
            r"\btote\b(?!.*board|.*m\b)",
            r"\bclutch\b(?!.*cable|.*kit)",
            r"\bbackpack\b",
            r"\bwallet\b",
            r"\bcrossbody\b",
            r"\bfanny\s?pack\b",
            r"\bsatchel\b",
            r"\bwristlet\b",
        ],
    ),
    (
        "hats",
        [
            r"\bhat\b(?!.*manhattan)",
            r"\bcaps?\b(?!.*acity|.*ital|.*tain|.*sule)",
            r"\bbeanie\b",
            r"\bberet\b",
            r"\bfedora\b",
            r"\bbucket\s?hat\b",
            r"\bsun\s?hat\b",
            r"\bbalaclava\b",
        ],
    ),
    (
        "scarves",
        [
            r"\bscarf\b",
            r"\bscarves\b",
            r"\bshawl\b(?!\s+collar)",
            r"\bbandana\b",
            r"\bstole\b(?!.*stolen)",
            # "wrap" only when it's a garment (not "wire wrap" or "wrap dress")
            r"\bwrap\b(?!.*dress|.*top|.*skirt|.*wire|.*around|.*gift|.*bracelet)",
        ],
    ),
    (
        "belts",
        [
            r"\bbelt\b(?!.*seat)",
            r"\bsuspenders?\b",
        ],
    ),
    (
        "jewelry",
        [
            r"\bjewel(?:ry|lery)\b",
            r"\bnecklace\b",
            r"\bbracelet\b",
            r"\bearrings?\b",
            r"\bpendant\b",
            r"\bcufflinks?\b",
            r"\bbrooch\b",
            r"\banklet\b",
            r"\bchoker\b",
            r"\bpiercing\b",
            r"\bbarbell\b(?!.*exercise|.*gym)",
            # "ring" with jewelry context
            r"\bring\b(?=.*(?:sterling|silver|gold|diamond|titanium|tungsten|wedding|engagement|band|stainless|gemstone|zirconia|plated|ct\b|karat|carat))",
        ],
    ),
    (
        "accessories",
        [
            r"\bgloves?\b(?!.*boxing|.*latex|.*nitrile|.*exam)",
            r"\bmittens?\b",
            r"\bnecktie\b",
            r"\bbow\s?tie\b",
            r"\btie\b(?!.*dye|.*tied|.*knot|.*front|.*back|.*waist|.*neck|.*side|.*string|.*down)",
            r"\bumbrella\b",
            r"\bkeychain\b",
            r"\bhair\s?(?:clip|tie|accessories|pin|bow|elastic)\b",
            r"\bscrunchie\b",
            r"\bheadband\b",
            r"\bhead\s?wrap\b",
            r"\bface\s?(?:mask|cover|shield)\b",
            r"\bneck\s?(?:gaiter|warmer)\b",
        ],
    ),
]

_CAT_COMPILED = [(cat, [re.compile(p, re.I) for p in patterns]) for cat, patterns in CATEGORY_RULES]


def extract_gender(title):
    for gender, pattern in GENDER_PATTERNS.items():
        if pattern.search(title):
            return gender
    return "unknown"


def extract_category(title):
    for cat, patterns in _CAT_COMPILED:
        for p in patterns:
            if p.search(title):
                return cat
    return "unknown"


# ══════════════════════════════════════════════════════════════════════════
#  FIELD PARSERS
# ══════════════════════════════════════════════════════════════════════════


def parse_price(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if v > 0 else None
    if isinstance(v, str):
        if v.strip().lower() in ("", "none"):
            return None
        v = v.replace("$", "").replace(",", "").strip()
        try:
            val = float(v.split()[0])
            return val if val > 0 else None
        except (ValueError, IndexError):
            return None
    return None


def parse_details(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {}
    return {}


def get_image_url(p):
    """Get the best available image URL for a product."""
    images = p.get("images")
    if not images:
        return None
    if isinstance(images, list):
        for img in images:
            if isinstance(img, dict):
                return img.get("hi_res") or img.get("large") or img.get("thumb")
            elif isinstance(img, str):
                return img
    elif isinstance(images, dict):
        for key in ["hi_res", "large", "thumb"]:
            urls = images.get(key) or []
            for u in urls:
                if u:
                    return u
    return None


def get_image_count(p):
    images = p.get("images")
    if not images:
        return 0, False
    has_hires = False
    count = 0
    if isinstance(images, list):
        count = len(images)
        has_hires = any(isinstance(img, dict) and img.get("hi_res") for img in images)
    elif isinstance(images, dict):
        large = images.get("large") or []
        hires = images.get("hi_res") or []
        count = len(large) or len(hires)
        has_hires = any(u for u in hires if u)
    return count, has_hires


# ══════════════════════════════════════════════════════════════════════════
#  ANALYZE
# ══════════════════════════════════════════════════════════════════════════


def do_analyze(path, limit=None):
    if not os.path.exists(path):
        print(f"not found: {path}\nrun --download first")
        sys.exit(1)

    N_SAMPLES = 16
    random.seed(42)

    s = {
        "total": 0,
        "bt_present": 0,
        "has_image": 0,
        "has_hi_res": 0,
        "image_counts": [],
        "has_raw_categories": 0,
        "title_categories": Counter(),
        "title_genders": Counter(),
        "has_title": 0,
        "title_lengths": [],
        "has_description": 0,
        "has_features": 0,
        "feature_counts": [],
        "has_price": 0,
        "prices": [],
        "has_rating": 0,
        "ratings": [],
        "rating_counts": [],
        "rating_buckets": Counter(),
        "high_quality": 0,
        "low_quality": 0,
        "has_details": 0,
        "detail_keys": Counter(),
        "department_values": Counter(),
        "has_store": 0,
        "stores": Counter(),
        "cat_samples": {},
        "unclassified_samples": [],
    }

    print(f"analyzing {path}" + (f" (first {limit:,})" if limit else "") + "...")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                p = json.loads(line)
            except Exception:
                continue
            s["total"] += 1
            title = (p.get("title") or "").strip()

            bt = p.get("bought_together")
            if bt and isinstance(bt, list) and len(bt) > 0:
                s["bt_present"] += 1

            count, hires = get_image_count(p)
            if count > 0:
                s["has_image"] += 1
                s["image_counts"].append(count)
            if hires:
                s["has_hi_res"] += 1

            if p.get("categories") and len(p["categories"]) > 0:
                s["has_raw_categories"] += 1

            if title:
                s["has_title"] += 1
                s["title_lengths"].append(len(title.split()))
                cat = extract_category(title)
                gender = extract_gender(title)
                s["title_categories"][cat] += 1
                s["title_genders"][gender] += 1

                if cat not in s["cat_samples"]:
                    s["cat_samples"][cat] = []
                bucket = s["cat_samples"][cat]
                cat_count = s["title_categories"][cat]
                if len(bucket) < N_SAMPLES:
                    bucket.append(title)
                else:
                    j = random.randint(0, cat_count - 1)
                    if j < N_SAMPLES:
                        bucket[j] = title

                if cat == "unknown" and len(s["unclassified_samples"]) < 20:
                    s["unclassified_samples"].append(title)

            desc = p.get("description") or []
            text = ""
            if isinstance(desc, list):
                text = " ".join(d for d in desc if isinstance(d, str) and d.strip())
            elif isinstance(desc, str):
                text = desc
            if text.strip():
                s["has_description"] += 1
            features = p.get("features") or []
            if features and len(features) > 0:
                s["has_features"] += 1
                s["feature_counts"].append(len(features))

            price = parse_price(p.get("price"))
            if price is not None:
                s["has_price"] += 1
                s["prices"].append(price)

            rating = p.get("average_rating")
            rn = p.get("rating_number") or 0
            if rating is not None:
                rating = float(rating)
                if rating > 0:
                    s["has_rating"] += 1
                    s["ratings"].append(rating)
                    s["rating_buckets"][int(rating)] += 1
                rn = int(rn)
                if rn > 0:
                    s["rating_counts"].append(rn)
                if rating >= 4.0 and rn >= 10:
                    s["high_quality"] += 1
                if rating < 3.0 or rn < 2:
                    s["low_quality"] += 1

            details = parse_details(p.get("details"))
            if details:
                s["has_details"] += 1
                for k, v in details.items():
                    s["detail_keys"][k] += 1
                    if k == "Department" and isinstance(v, str):
                        s["department_values"][v.strip()] += 1

            store = (p.get("store") or "").strip()
            if store:
                s["has_store"] += 1
                s["stores"][store] += 1

            if (i + 1) % 100_000 == 0:
                print(f"  {i + 1:,} records...")

    FASHION_KEYS = {
        "Department",
        "Material",
        "Color",
        "Style",
        "Closure Type",
        "Size",
        "Brand",
        "Fabric Type",
        "Pattern",
        "Fit Type",
    }
    s["fashion_coverage"] = {
        k: s["detail_keys"].get(k, 0)
        for k in sorted(FASHION_KEYS)
        if s["detail_keys"].get(k, 0) > 0
    }

    _print_analysis(s)

    saveable = {
        "total": s["total"],
        "title_categories": dict(s["title_categories"].most_common(50)),
        "title_genders": dict(s["title_genders"].most_common()),
        "category_samples": {k: v for k, v in s["cat_samples"].items() if k != "unknown"},
        "unclassified_samples": s["unclassified_samples"],
        "bt_present": s["bt_present"],
        "has_image": s["has_image"],
        "has_hi_res": s["has_hi_res"],
        "has_title": s["has_title"],
        "has_description": s["has_description"],
        "has_features": s["has_features"],
        "has_price": s["has_price"],
        "has_rating": s["has_rating"],
        "high_quality": s["high_quality"],
        "low_quality": s["low_quality"],
        "has_store": s["has_store"],
        "unique_stores": len(s["stores"]),
        "top_stores": dict(s["stores"].most_common(20)),
        "fashion_detail_coverage": s["fashion_coverage"],
        "department_values": dict(s["department_values"].most_common(20)),
    }
    with open("exploration_stats.json", "w") as f:
        json.dump(saveable, f, indent=2)
    print("stats saved to exploration_stats.json")


def _print_analysis(s):

    def pct(x):
        return f"{x:,} ({x / n:.1%})" if n > 0 else "0"

    n = s["total"]
    classified = n - s["title_categories"].get("unknown", 0)

    print(f"\n{'=' * 70}")
    print(f" AMAZON FASHION — {n:,} products")
    print(f"{'=' * 70}")

    print("\n┌- CATEGORY CLASSIFICATION ------------------------------------")
    print(
        f"│  title-classified: {pct(classified)}  unclassified: {pct(s['title_categories'].get('unknown', 0))}"
    )
    print("│")
    for cat, count in s["title_categories"].most_common(30):
        if cat == "unknown":
            continue
        print(f"│  {cat:15} {count:>8,} ({count / n:>5.1%})")
    print("└--------------------------------------------------------------")

    print("\n┌- CATEGORY SAMPLES -------------------------------------------")
    for cat, count in s["title_categories"].most_common(30):
        if cat == "unknown":
            continue
        samples = s["cat_samples"].get(cat, [])
        print(f"│  -- {cat.upper()} ({count:,}) --")
        for t in samples[:8]:
            print(f"│    • {t}")
        print("│")
    print("└--------------------------------------------------------------")

    unk = s["title_categories"].get("unknown", 0)
    print(f"\n┌- UNCLASSIFIED ({unk:,}) ------------------------------------")
    for t in s["unclassified_samples"]:
        print(f"│  • {t}")
    print("└--------------------------------------------------------------")

    print("\n┌- GENDER -----------------------------------------------------")
    for g, c in s["title_genders"].most_common():
        print(f"│  {g:10} {c:>8,} ({c / n:.1%})")
    print("└--------------------------------------------------------------")

    print("\n┌- OTHER FIELDS -----------------------------------------------")
    print(f"│  bought_together: {pct(s['bt_present'])}")
    print(f"│  images:          {pct(s['has_image'])}  hi-res: {pct(s['has_hi_res'])}")
    print(
        f"│  titles:          {pct(s['has_title'])}"
        + (f"  avg {mean(s['title_lengths']):.0f} words" if s["title_lengths"] else "")
    )
    print(f"│  features:        {pct(s['has_features'])}")
    print(f"│  descriptions:    {pct(s['has_description'])}")
    print(f"│  price:           {pct(s['has_price'])}")
    print(
        f"│  ratings:         {pct(s['has_rating'])}"
        + (f"  avg {mean(s['ratings']):.2f}" if s["ratings"] else "")
    )
    print(f"│  high quality:    {pct(s['high_quality'])}")
    print(f"│  stores:          {pct(s['has_store'])}  unique: {len(s['stores']):,}")
    print("└--------------------------------------------------------------")

    print("\n┌- DETAILS (fashion-relevant) --------------------------------")
    for key, count in sorted(s["fashion_coverage"].items(), key=lambda x: -x[1]):
        print(f"│  {key:20} {count:>8,} ({count / n:.1%})")
    print("└--------------------------------------------------------------")

    gender_known = n - s["title_genders"].get("unknown", 0)
    print(f"\n{'=' * 70}")
    print(" SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"  title→category   {classified / n:.0%}    regex ({s['title_categories'].get('unknown', 0) / n:.0%} unclassified)"
    )
    print(f"  title→gender     {gender_known / n:.0%}")
    print(f"  images           {s['has_image'] / n:.0%}    FashionSigLIP ready")
    print(f"  features         {s['has_features'] / n:.0%}    BM25 supplement")
    print(f"  price            {s['has_price'] / n:.0%}    de-scope budget")
    print(f"  bought_together  {s['bt_present'] / n:.0%}    must LLM-curate outfits")
    print()


# ══════════════════════════════════════════════════════════════════════════
#  CLASSIFY — compare regex vs FashionSigLIP zero-shot
# ══════════════════════════════════════════════════════════════════════════

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
    ("unisex", "unisex clothing for anyone"),
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

ALL_ATTRIBUTES = {
    "category": CLIP_CATEGORIES,
    "gender": CLIP_GENDER,
    "season": CLIP_SEASON,
    "formality": CLIP_FORMALITY,
    "color": CLIP_COLOR,
}

# season and formality only make sense for wearable clothing, not accessories
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
CLOTHING_ONLY_ATTRS = {"season", "formality"}


def _encode_prompts(model, tokenizer, prompts, device):
    """Encode a list of (name, prompt) pairs, return (names, normalized features)."""
    import torch

    names = [n for n, _ in prompts]
    texts = [p for _, p in prompts]
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return names, features


def _classify_against(title_features, attr_names, attr_features):
    """Classify one title embedding against one attribute's prompt set."""
    sims = (100.0 * title_features @ attr_features.T).squeeze()
    idx = sims.argmax().item()
    return attr_names[idx], sims[idx].item()


# regex patterns for explicit gender keywords in titles (separate from analyze patterns)
_RE_WOMEN = re.compile(r"\b(women'?s?|woman'?s?|ladies|female)\b", re.I)
_RE_MEN = re.compile(r"\b(men'?s?|man'?s?|male)\b", re.I)
_RE_GIRLS = re.compile(r"\b(girls?'?s?)\b", re.I)
_RE_BOYS = re.compile(r"\b(boys?'?s?)\b", re.I)
_RE_UNISEX = re.compile(r"\b(unisex)\b", re.I)


def _regex_gender(title):
    """Extract gender from title using keyword matching.
    If both male+female or both boys+girls → unisex.
    Returns (label, source) or (None, None) if no keywords found."""
    # explicit unisex keyword wins immediately
    if _RE_UNISEX.search(title):
        return "unisex", "regex"

    has_women = bool(_RE_WOMEN.search(title))
    has_men = bool(_RE_MEN.search(title))
    has_girls = bool(_RE_GIRLS.search(title))
    has_boys = bool(_RE_BOYS.search(title))

    # both genders mentioned → unisex
    if (has_women and has_men) or (has_girls and has_boys):
        return "unisex", "regex"
    # single match
    if has_women:
        return "women", "regex"
    if has_men:
        return "men", "regex"
    if has_girls:
        return "girls", "regex"
    if has_boys:
        return "boys", "regex"
    return None, None


# confidence thresholds — below these, fall back to a safe default
CONF_THRESHOLDS = {
    # no gender threshold — regex handles explicit, CLIP handles the rest
    "season": (None, 15),  # below 15 → skip
    "formality": (None, 15),  # below 15 → skip
    "color": (None, 10),  # below 10 → skip
}

# max words for enriched text (SigLIP tokenizer ~64 tokens ≈ 50 words)
MAX_RICH_TEXT_WORDS = 45

DETAIL_KEYS_FOR_TEXT = [
    "Department",
    "Material",
    "Color",
    "Style",
    "Fabric Type",
    "Closure Type",
    "Pattern",
]


def build_rich_text(prod):
    """Build enriched text from title + features + details for better CLIP classification.
    Stays within ~45 words to fit SigLIP token budget."""
    parts = [prod.get("title", "").strip()]

    # add first 2-3 feature bullets
    features = prod.get("features") or []
    for f in features[:3]:
        if isinstance(f, str) and f.strip():
            parts.append(f.strip())

    # add key detail fields
    details = parse_details(prod.get("details"))
    for key in DETAIL_KEYS_FOR_TEXT:
        val = details.get(key)
        if val and isinstance(val, str) and val.strip():
            parts.append(f"{key}: {val.strip()}")

    # join and truncate to word budget
    full = ". ".join(parts)
    words = full.split()
    if len(words) > MAX_RICH_TEXT_WORDS:
        full = " ".join(words[:MAX_RICH_TEXT_WORDS])
    return full


def do_classify(path, sample_size=200, device="cpu", with_images=False):
    """Multi-attribute classification: category, gender, season, formality, color.
    Uses enriched text (title + features + details) for better CLIP accuracy."""
    try:
        import open_clip
        import torch
    except ImportError:
        print("ERROR: --classify requires: pip install torch open_clip_torch ftfy")
        sys.exit(1)
    if with_images:
        try:
            from PIL import Image
        except ImportError:
            print("ERROR: --with-images requires: pip install Pillow")
            sys.exit(1)

    if not os.path.exists(path):
        print(f"not found: {path}\nrun --download first")
        sys.exit(1)

    # sample products (capture full record for enriched text)
    print(f"sampling {sample_size} products...")
    random.seed(42)
    all_products = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
            except Exception:
                continue
            title = (p.get("title") or "").strip()
            if not title:
                continue
            prod = {
                "title": title,
                "asin": p.get("parent_asin", ""),
                "features": p.get("features") or [],
                "details": p.get("details") or {},
            }
            if with_images:
                url = get_image_url(p)
                if url:
                    prod["image_url"] = url
            all_products.append(prod)
    random.shuffle(all_products)
    sample = all_products[:sample_size]
    print(f"  selected {len(sample)} from {len(all_products):,}")

    # count how many have enrichment data
    has_features = sum(1 for p in sample if p["features"])
    has_details = sum(1 for p in sample if parse_details(p["details"]))
    print(
        f"  with features: {has_features} ({has_features / len(sample):.0%})"
        f"  with details: {has_details} ({has_details / len(sample):.0%})"
    )

    # load model
    print(f"loading FashionSigLIP via open_clip on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    model = model.to(device).eval()

    # encode ALL attribute prompt sets
    print("encoding attribute prompts...")
    attr_data = {}
    for attr_name, prompts in ALL_ATTRIBUTES.items():
        names, features = _encode_prompts(model, tokenizer, prompts, device)
        attr_data[attr_name] = (names, features)

    # classify each product
    mode = "text + image" if with_images else "text only"
    print(f"classifying {len(sample)} products across {len(ALL_ATTRIBUTES)} attributes ({mode})...")
    results = []
    for i, prod in enumerate(sample):
        rich_text = build_rich_text(prod)
        result = {
            "title": prod["title"],
            "rich_text": rich_text,
            "has_enrichment": bool(prod["features"] or parse_details(prod["details"])),
            "regex_cat": extract_category(prod["title"]),
            "regex_gender": extract_gender(prod["title"]),
        }

        # encode title (for gender) and enriched text (for everything else)
        try:
            title_tokens = tokenizer([prod["title"]]).to(device)
            rich_tokens = tokenizer([rich_text]).to(device)
            with torch.no_grad():
                title_features = model.encode_text(title_tokens)
                title_features = title_features / title_features.norm(dim=-1, keepdim=True)
                rich_features = model.encode_text(rich_tokens)
                rich_features = rich_features / rich_features.norm(dim=-1, keepdim=True)

            # category from enriched text
            cat_label, cat_conf = _classify_against(rich_features, *attr_data["category"])
            result["clip_category"] = cat_label
            result["clip_category_conf"] = cat_conf

            # remaining attributes
            for attr_name, (names, features) in attr_data.items():
                if attr_name == "category":
                    continue
                # season/formality only for clothing categories
                if attr_name in CLOTHING_ONLY_ATTRS and cat_label not in CLOTHING_CATEGORIES:
                    result[f"clip_{attr_name}"] = None
                    result[f"clip_{attr_name}_conf"] = None
                    continue
                # gender: regex first (explicit keywords), CLIP fallback
                if attr_name == "gender":
                    regex_label, regex_src = _regex_gender(prod["title"])
                    if regex_label is not None:
                        result["clip_gender"] = regex_label
                        result["clip_gender_conf"] = 99  # high conf for regex match
                        result["gender_source"] = "regex"
                        continue
                    # no keyword found → CLIP on title-only
                    label, conf = _classify_against(title_features, names, features)
                    result["clip_gender"] = label
                    result["clip_gender_conf"] = conf
                    result["gender_source"] = "clip"
                    continue
                # everything else uses enriched text
                label, conf = _classify_against(rich_features, names, features)
                # apply confidence threshold
                if attr_name in CONF_THRESHOLDS:
                    fallback, threshold = CONF_THRESHOLDS[attr_name]
                    if conf < threshold:
                        label = fallback
                result[f"clip_{attr_name}"] = label
                result[f"clip_{attr_name}_conf"] = conf
        except Exception:
            for attr_name in ALL_ATTRIBUTES:
                result[f"clip_{attr_name}"] = "error"
                result[f"clip_{attr_name}_conf"] = 0

        # image-based color classification (primary for color when available)
        if with_images and prod.get("image_url"):
            try:
                req = urllib.request.Request(
                    prod["image_url"], headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    img_data = resp.read()
                img = Image.open(BytesIO(img_data)).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_features = model.encode_image(img_tensor)
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                color_label, color_conf = _classify_against(img_features, *attr_data["color"])
                result["clip_color_text"] = result.get("clip_color")  # save text-based
                result["clip_color"] = color_label  # override with image-based
                result["clip_color_conf"] = color_conf
                result["color_from_image"] = True
            except Exception:
                result["color_from_image"] = False
        else:
            result["color_from_image"] = False

        results.append(result)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sample)} done...")

    # ══════════════════════════════════════════════════════════════════════
    #  REPORT
    # ══════════════════════════════════════════════════════════════════════
    valid = [r for r in results if r.get("clip_category") != "error"]
    n = len(valid)

    # CATEGORY: regex vs CLIP
    cat_agree = [r for r in valid if r["regex_cat"] == r["clip_category"]]
    cat_disagree = [r for r in valid if r["regex_cat"] != r["clip_category"]]
    regex_unk = [r for r in valid if r["regex_cat"] == "unknown"]
    clip_filled = [r for r in regex_unk if r["clip_category"] != "unknown"]

    print(f"\n{'=' * 70}")
    print(f" MULTI-ATTRIBUTE CLASSIFICATION ({n} products)")
    print(f"{'=' * 70}")

    print("\n┌- CATEGORY: regex vs CLIP-text ------------------------------")
    print(f"│  agree:         {len(cat_agree):>4} ({len(cat_agree) / n:.1%})")
    print(f"│  disagree:      {len(cat_disagree):>4} ({len(cat_disagree) / n:.1%})")
    print(f"│  regex unknown: {len(regex_unk):>4} → CLIP filled {len(clip_filled)}")
    print("└--------------------------------------------------------------")

    if cat_disagree:
        print("\n┌- CATEGORY DISAGREEMENTS (first 20) -------------------------")
        for r in cat_disagree[:20]:
            print(f'│  regex={r["regex_cat"]:15} clip={r["clip_category"]:15} "{r["title"]}"')
        if len(cat_disagree) > 20:
            print(f"│  ... and {len(cat_disagree) - 20} more")
        print("└--------------------------------------------------------------")

    # per-category agreement
    ct_agree, ct_total = Counter(), Counter()
    for r in valid:
        if r["regex_cat"] != "unknown":
            ct_total[r["regex_cat"]] += 1
            if r["regex_cat"] == r["clip_category"]:
                ct_agree[r["regex_cat"]] += 1
    if ct_total:
        print("\n┌- PER-CATEGORY AGREEMENT ------------------------------------")
        for cat in sorted(ct_total, key=lambda c: -ct_total[c]):
            t, a = ct_total[cat], ct_agree[cat]
            rate = a / t if t > 0 else 0
            bar = "█" * int(rate * 15)
            print(f"│  {cat:15} {a:>3}/{t:<3} ({rate:>5.0%}) {bar}")
        print("└--------------------------------------------------------------")

    # GENDER: regex+CLIP hybrid
    gen_from_regex = [r for r in valid if r.get("gender_source") == "regex"]
    gen_from_clip = [r for r in valid if r.get("gender_source") == "clip"]

    print("\n┌- GENDER (regex+CLIP hybrid) ------------------------------")
    print(
        f"│  from regex:    {len(gen_from_regex):>4} ({len(gen_from_regex) / n:.1%})  (explicit keywords in title)"
    )
    print(
        f"│  from CLIP:     {len(gen_from_clip):>4} ({len(gen_from_clip) / n:.1%})  (no keywords, CLIP fallback)"
    )
    print("│")
    print("│  gender distribution:")
    gen_dist = Counter(r["clip_gender"] for r in valid)
    for g, c in gen_dist.most_common():
        # show source breakdown per label
        from_r = sum(
            1 for r in valid if r["clip_gender"] == g and r.get("gender_source") == "regex"
        )
        from_c = sum(1 for r in valid if r["clip_gender"] == g and r.get("gender_source") == "clip")
        print(f"│    {g:10} {c:>5} ({c / n:.1%})  regex={from_r} clip={from_c}")
    print("└--------------------------------------------------------------")

    # gender samples per label (show source)
    print("\n┌- GENDER SAMPLES PER LABEL ---------------------------------")
    gen_samples = defaultdict(list)
    for r in valid:
        g = r.get("clip_gender")
        src = r.get("gender_source", "?")
        if g and g != "error" and len(gen_samples[g]) < 10:
            conf = r.get("clip_gender_conf", 0)
            tag = "R" if src == "regex" else f"C{conf:.0f}"
            gen_samples[g].append(f"[{tag}] {r['title']}")
    for g, c in gen_dist.most_common():
        print(f"│  -- {g} ({c}) --")
        for s in gen_samples.get(g, []):
            print(f"│    • {s}")
    print("└--------------------------------------------------------------")

    # SEASON, FORMALITY, COLOR: distributions + samples
    for attr_name, display_name in [
        ("season", "SEASON"),
        ("formality", "FORMALITY"),
        ("color", "COLOR"),
    ]:
        clip_key = f"clip_{attr_name}"
        classified = [r for r in valid if r.get(clip_key) and r[clip_key] not in ("error", None)]
        skipped = [r for r in valid if r.get(clip_key) is None]
        dist = Counter(r[clip_key] for r in classified)
        total = len(classified)

        scope = "clothing only" if attr_name in CLOTHING_ONLY_ATTRS else "all products"
        print(f"\n┌- {display_name} (CLIP-text, {scope}) ----------------------")
        if skipped:
            print(f"│  classified: {total}  skipped (non-clothing): {len(skipped)}")
            print("│")
        for label, count in dist.most_common():
            bar = "█" * int(count / (max(dist.values()) or 1) * 15)
            print(f"│  {label:16} {count:>5} ({count / total:.1%})  {bar}")
        print("│")

        # show samples per label
        label_samples = defaultdict(list)
        for r in classified:
            if len(label_samples[r[clip_key]]) < 6:
                conf = r.get(f"clip_{attr_name}_conf", 0)
                label_samples[r[clip_key]].append(f"({conf:.0f}) {r['title']}")

        print("│  samples per label:")
        for label, count in dist.most_common():
            print(f"│  -- {label} --")
            for s in label_samples[label]:
                print(f"│    • {s}")
        print("└--------------------------------------------------------------")

    # IMAGE vs TEXT color comparison (if --with-images)
    if with_images:
        img_color = [r for r in valid if r.get("color_from_image")]
        if img_color:
            matches = sum(1 for r in img_color if r.get("clip_color") == r.get("clip_color_text"))
            print(f"\n┌- COLOR: image vs text ({len(img_color)} products) ----------")
            print(f"│  image=text agree: {matches} ({matches / len(img_color):.1%})")
            print("│  image source used for color distribution above")
            print("│")
            # show disagreements
            disagree = [r for r in img_color if r.get("clip_color") != r.get("clip_color_text")]
            if disagree:
                print("│  disagreements (first 10):")
                for r in disagree[:10]:
                    print(
                        f"│    text={r.get('clip_color_text', '?'):12} "
                        f'img={r["clip_color"]:12} "{r["title"]}"'
                    )
            print("└--------------------------------------------------------------")


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description="Amazon Fashion dataset tools")
    ap.add_argument("--download", action="store_true", help="download the dataset")
    ap.add_argument("--analyze", action="store_true", help="analyze the dataset")
    ap.add_argument(
        "--classify",
        action="store_true",
        help="compare regex vs FashionSigLIP zero-shot on a sample",
    )
    ap.add_argument(
        "--with-images",
        action="store_true",
        help="also classify from images (slower, requires Pillow)",
    )
    ap.add_argument("--path", default=DEFAULT_PATH, help="path to .jsonl.gz")
    ap.add_argument("--output", default=None, help="download output path")
    ap.add_argument("--limit", type=int, default=None, help="first N records only")
    ap.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="number of products to classify (for --classify)",
    )
    ap.add_argument("--device", default="cpu", help="torch device (cpu/cuda)")
    args = ap.parse_args()

    if not args.download and not args.analyze and not args.classify:
        ap.print_help()
        sys.exit(1)

    if args.download:
        do_download(args.output or args.path)
    if args.analyze:
        do_analyze(args.path, limit=args.limit)
    if args.classify:
        do_classify(
            args.path,
            sample_size=args.sample_size,
            device=args.device,
            with_images=args.with_images,
        )


if __name__ == "__main__":
    main()
