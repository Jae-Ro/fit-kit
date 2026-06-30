import argparse
import gzip
import heapq
import json
from pathlib import Path

import polars as pl

DEFAULT_CATALOG = "data/catalog/catalog.parquet"
DEFAULT_REVIEWS = "data/raw/review_categories/Amazon_Fashion.jsonl"
MAX_REVIEWS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--reviews-path", default=DEFAULT_REVIEWS)
    args = ap.parse_args()

    df = pl.read_parquet(args.catalog)
    asin_set = set(df["asin"].to_list())
    print(f"catalog: {len(df):,} products")

    # collect top reviews per product (by helpful_vote)
    review_heaps: dict[str, list] = {asin: [] for asin in asin_set}
    opener = gzip.open if args.reviews_path.endswith(".gz") else open

    print(f"reading {args.reviews_path}...")
    total = 0
    matched = 0

    with opener(args.reviews_path, "rt", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                r = json.loads(line)
            except Exception:
                continue

            asin = r.get("parent_asin", "")
            if asin not in asin_set:
                continue
            if not r.get("verified_purchase", False) or (r.get("rating") or 0) < 3:
                continue

            matched += 1
            title = (r.get("title") or "").strip()
            body = (r.get("text") or "").strip()
            text = f"{title}. {body}" if title and body else (title or body)
            if not text:
                continue

            helpful = r.get("helpful_vote") or 0
            rating = r.get("rating") or 0
            heap = review_heaps[asin]

            entry = (
                helpful,
                json.dumps(
                    {
                        "title": title,
                        "text": body,
                        "rating": rating,
                        "helpful_votes": helpful,
                    }
                ),
            )

            if len(heap) < MAX_REVIEWS:
                heapq.heappush(heap, entry)
            elif helpful > heap[0][0]:
                heapq.heapreplace(heap, entry)

            if total % 500_000 == 0:
                print(f"  scanned {total:,}, matched {matched:,}")

    print(f"  total: {total:,}, matched: {matched:,}")

    # build the column
    reviews_col = []
    for asin in df["asin"].to_list():
        heap = review_heaps.get(asin, [])
        if heap:
            sorted_reviews = [json.loads(r) for _, r in sorted(heap, reverse=True)]
            reviews_col.append(json.dumps(sorted_reviews))
        else:
            reviews_col.append("[]")

    df = df.with_columns(pl.Series("reviews_json", reviews_col))

    has_reviews = (df["reviews_json"].str.len_chars() > 2).sum()
    print(f"products with reviews: {has_reviews:,}")

    # save (overwrite)
    out = Path(args.catalog)
    df.write_parquet(out)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
