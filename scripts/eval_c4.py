import argparse
import math
import time

import open_clip
import polars as pl
import torch
from datasets import load_dataset

from fit_kit.core.search import CatalogSearch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog-dir", default="data/catalog")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--embed-mode", default="adaptive")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--min-dense-ratio", type=float, default=0.0)
    ap.add_argument("--min-dense-score", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument(
        "--use-planner",
        action="store_true",
        help="Use LLM planner to rewrite queries and add filters",
    )
    ap.add_argument(
        "--use-router",
        action="store_true",
        help="Use centroid router to classify intent before planner",
    )
    ap.add_argument("--planner-model", default="openai:gpt-5.5")
    args = ap.parse_args()

    # load search index
    search = CatalogSearch(
        args.catalog_dir,
        device=args.device,
        embed_mode=args.embed_mode,
        alpha=args.alpha,
    )

    # load model for query encoding
    print("\nloading FashionSigLIP for query encoding...")
    model, _, _ = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    model = model.to(search.device).eval()

    def encode(text):
        tokens = tokenizer([text]).to(search.device)
        with torch.no_grad():
            feat = model.encode_text(tokens)
            return feat / feat.norm(dim=-1, keepdim=True)

    # optionally load planner
    planner_agent = None
    if args.use_planner or args.use_router:
        from fit_kit.core.planner import create_planner_agent, plan
        from fit_kit.core.recommender import _build_filters
        from fit_kit.core.schemas import SlotPlan

        planner_agent = create_planner_agent(model=args.planner_model)
        print(f"planner loaded ({args.planner_model})")

    # optionally load router
    query_router = None
    if args.use_router:
        from fit_kit.core.recommender import QueryEncoder, QueryRouter

        router_encoder = QueryEncoder(search.device)
        query_router = QueryRouter(router_encoder)
        print("router loaded")

    # load C4 dataset and find matching queries
    print("\nloading Amazon-C4 dataset...")
    ds = load_dataset("McAuley-Lab/Amazon-C4", split="test")

    catalog_asins = set(search.df["asin"].to_list())
    matching = [
        (row["qid"], row["query"], row["item_id"], row["ori_review"])
        for row in ds
        if row["item_id"] in catalog_asins
    ]

    print(f"  {len(matching)} queries match our catalog")
    if args.use_router:
        print("  mode: ROUTER → planner (outfit) or direct search (single_item)")
    elif args.use_planner:
        print("  mode: PLANNER (query rewriting + filters)")
    else:
        print("  mode: RAW (direct query, no filters)")
    print()

    # run evaluation
    hits_at = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
    reciprocal_ranks = []
    query_times_ms = []

    for qid, query, target_asin, review in matching:
        t_query_start = time.perf_counter()
        query_emb = encode(query)

        # determine intent
        use_planner_for_query = args.use_planner
        intent = None
        if query_router is not None:
            intent = query_router.route(query_emb)
            use_planner_for_query = intent == "outfit"

        if use_planner_for_query:
            # planner rewrites query + adds filters per slot
            try:
                slot_plan = plan(query, agent=planner_agent)
            except Exception as e:
                print(f"  Q{qid}: planner failed ({e}), falling back to raw")
                slot_plan = None

            if slot_plan is not None:
                # search each slot, merge results
                all_results = {}  # asin → best score
                all_rows = {}  # asin → row dict

                for sq in slot_plan.slot_queries:
                    filters = _build_filters(sq, slot_plan)
                    query_emb = encode(sq.query)

                    slot_results = search.query(
                        text=sq.query,
                        query_emb=query_emb,
                        filters=filters,
                        top_k=args.top_k,
                        mode="hybrid",
                        beta=args.beta,
                        min_dense_ratio=args.min_dense_ratio,
                        min_dense_score=args.min_dense_score,
                    )

                    if len(slot_results) > 0:
                        for row in slot_results.iter_rows(named=True):
                            asin = row["asin"]
                            score = row.get("score", 0)
                            if asin not in all_results or score > all_results[asin]:
                                all_results[asin] = score
                                all_rows[asin] = row

                # sort by score descending
                ranked = sorted(all_results.items(), key=lambda x: -x[1])
                result_asins = [asin for asin, _ in ranked]

                # print planner info
                cats = ["+".join(c.value for c in sq.category) for sq in slot_plan.slot_queries]
                print(f"{'=' * 80}")
                print(f"  Q{qid}: {query}")
                intent_str = f"  INTENT: {intent} → planner\n" if intent else ""
                print(
                    f"{intent_str}  PLANNER: {len(slot_plan.slot_queries)} slots [{', '.join(cats)}] "
                    f"constraints={slot_plan.constraints.model_dump(exclude_none=True)}"
                )
                for sq in slot_plan.slot_queries:
                    cat_str = "+".join(c.value for c in sq.category)
                    fmt_str = f" formality={sq.formality}" if sq.formality else ""
                    print(f'    → [{cat_str}] "{sq.query}"{fmt_str}')
            else:
                result_asins = []
        else:
            # raw query — no planner, no filters
            results = search.query(
                text=query,
                query_emb=query_emb,
                filters=None,
                top_k=args.top_k,
                mode="hybrid",
                beta=args.beta,
                min_dense_ratio=args.min_dense_ratio,
                min_dense_score=args.min_dense_score,
            )

            result_asins = results["asin"].to_list() if len(results) > 0 else []
            all_rows = (
                {row["asin"]: row for row in results.iter_rows(named=True)}
                if len(results) > 0
                else {}
            )

            print(f"{'=' * 80}")
            print(f"  Q{qid}: {query}")
            if intent:
                print(f"  INTENT: {intent} → direct search")

        # find rank of target
        rank = None
        if target_asin in result_asins:
            rank = result_asins.index(target_asin) + 1

        # update metrics
        for k in hits_at:
            if rank is not None and rank <= k:
                hits_at[k] += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        # print result
        hit_str = f"✅ rank={rank}" if rank else "❌ not found"

        target_row = search.df.filter(pl.col("asin") == target_asin)
        if len(target_row) > 0:
            target_title = target_row["title"].item()
            target_cat = target_row["clip_category"].item()
            target_gender = target_row["clip_gender"].item()
            print(f"  TARGET: [{target_asin}] {target_title}")
            print(f"          {target_cat} | {target_gender}")
        print(f"  RESULT: {hit_str}")

        # show top 10
        top_asins = result_asins[:10]
        for i, asin in enumerate(top_asins):
            row = all_rows.get(asin, {})
            title = row.get("title", "?")
            marker = " ←←←" if asin == target_asin else ""
            print(f"    {i + 1:>2}. [{asin}] {title}{marker}")

        t_query_elapsed = (time.perf_counter() - t_query_start) * 1000
        query_times_ms.append(t_query_elapsed)
        print(f"  TIME: {t_query_elapsed:.0f}ms")
        print()

    # summary
    n = len(matching)
    mrr = sum(reciprocal_ranks) / n if n > 0 else 0

    # compute NDCG@k (single relevant item per query: NDCG = 1/log2(rank+1) if found)
    ndcg_ks = [10, 20, 50, 100]
    ndcg_at = {}
    for k in ndcg_ks:
        scores = []
        for rr in reciprocal_ranks:
            if rr > 0:
                rank = round(1.0 / rr)
                scores.append(1.0 / math.log2(rank + 1) if rank <= k else 0.0)
            else:
                scores.append(0.0)
        ndcg_at[k] = sum(scores) / n if n > 0 else 0

    print("=" * 80)
    print(f"  EVALUATION SUMMARY ({n} queries)")
    print("=" * 80)
    for k in sorted(hits_at):
        print(f"  Recall@{k:>2}: {hits_at[k]:>2}/{n} ({hits_at[k] / n:.1%})")
    print(f"  MRR:        {mrr:.3f}")
    for k in ndcg_ks:
        print(f"  NDCG@{k:<3}:   {ndcg_at[k]:.3f}")
    print()
    if query_times_ms:
        avg_ms = sum(query_times_ms) / len(query_times_ms)
        p50 = sorted(query_times_ms)[len(query_times_ms) // 2]
        p95 = sorted(query_times_ms)[int(len(query_times_ms) * 0.95)]
        outfit_times = [
            t
            for t, (_, q, _, _) in zip(query_times_ms, matching)
            if query_router and query_router.route(encode(q)) == "outfit"
        ]
        single_times = (
            [t for t in query_times_ms if t not in outfit_times] if outfit_times else query_times_ms
        )
        print("  LATENCY:")
        print(
            f"    avg={avg_ms:.0f}ms  p50={p50:.0f}ms  p95={p95:.0f}ms  total={sum(query_times_ms):.0f}ms"
        )
        if outfit_times:
            print(
                f"    outfit avg={sum(outfit_times) / len(outfit_times):.0f}ms ({len(outfit_times)} queries)"
            )
            print(
                f"    single avg={sum(single_times) / len(single_times):.0f}ms ({len(single_times)} queries)"
            )
    print()


if __name__ == "__main__":
    main()
