def main() -> None:
    import argparse
    import time

    ap = argparse.ArgumentParser(description="End-to-end outfit recommendation")
    ap.add_argument("--catalog-dir", default="data/catalog")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model", default="openai:gpt-5.5")
    ap.add_argument("--gender", default=None)
    ap.add_argument("--season", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--embed-mode", default="adaptive")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--min-dense-ratio", type=float, default=0.7)
    ap.add_argument("--min-dense-score", type=float, default=0.3)
    ap.add_argument(
        "--use-router",
        action="store_true",
        help="Use centroid-based router to skip planner for single-item queries",
    )
    ap.add_argument(
        "--ot-checkpoint",
        default=None,
        help="OutfitTransformer checkpoint for CP scoring (e.g. checkpoints/cp_best.pt)",
    )
    ap.add_argument(
        "--image-weight",
        type=float,
        default=1.0,
        help="Image vs text weight in OutfitTransformer (>1 = heavier image)",
    )
    ap.add_argument(
        "--ot-simulate",
        action="store_true",
        help="Run full greedy pass per anchor candidate to show alternative outfits",
    )
    ap.add_argument(
        "queries",
        nargs="*",
        default=[
            "outfit for a beach wedding this summer",
            "something warm and cozy for staying home",
            "I need a business casual look for the office",
        ],
    )
    args = ap.parse_args()

    from fit_kit.core.planner import create_planner_agent
    from fit_kit.core.recommender import OutfitScorer, QueryEncoder, QueryRouter, recommend
    from fit_kit.core.search import CatalogSearch

    # build user context
    user_context = {}
    if args.gender:
        user_context["gender"] = args.gender
    if args.season:
        user_context["season"] = args.season
    user_context = user_context or None

    # initialize components
    search = CatalogSearch(
        args.catalog_dir,
        device=args.device,
        embed_mode=args.embed_mode,
        alpha=args.alpha,
    )
    encoder = QueryEncoder(search.device)
    planner_agent = create_planner_agent(model=args.model)
    query_router = QueryRouter(encoder) if args.use_router else None
    scorer = (
        OutfitScorer(args.ot_checkpoint, search.device, args.image_weight)
        if args.ot_checkpoint
        else None
    )

    # run recommendations
    for query in args.queries:
        print()
        print("=" * 90)
        print(f'  QUERY: "{query}"')
        if user_context:
            print(f"  USER CONTEXT: {user_context}")
        print("=" * 90)

        t_start = time.perf_counter()

        rec = recommend(
            query=query,
            search=search,
            encoder=encoder,
            planner_agent=planner_agent,
            user_context=user_context,
            top_k_per_slot=args.top_k,
            embed_mode=args.embed_mode,
            alpha=args.alpha,
            beta=args.beta,
            min_dense_ratio=args.min_dense_ratio,
            min_dense_score=args.min_dense_score,
            router=query_router,
            outfit_scorer=scorer,
            simulate_outfits=args.ot_simulate,
        )

        t_end = time.perf_counter()

        print(
            f"\n  PLAN: occasion={rec.plan.occasion} "
            f"constraints={rec.plan.constraints.model_dump(exclude_none=True)} "
            f"({rec.plan_elapsed_s:.1f}s)"
        )

        for sr in rec.slot_results:
            cats = ", ".join(c.value for c in sr.slot_query.category)
            formality_str = (
                f" [{', '.join(sr.slot_query.formality)}]" if sr.slot_query.formality else ""
            )
            print(f"\n  -- {cats.upper()}{formality_str} --")
            print(f'     query: "{sr.slot_query.query}"')
            print(f"     filters: {sr.filters_used}")
            print(f"     {len(sr.products)} results ({sr.elapsed_ms:.0f}ms)")

            if len(sr.products) > 0:
                for i, row in enumerate(sr.products.iter_rows(named=True)):
                    title = row.get("title", "")
                    score = row.get("score", 0)
                    color = row.get("clip_color", "")
                    rating = row.get("average_rating", 0)
                    dense = row.get("dense_score", None)
                    sparse = row.get("sparse_score", None)
                    txt = row.get("text_score", None)
                    img = row.get("image_score", None)

                    print(f"     {i + 1}. [{score:.4f}] {title}")
                    parts = [f"{color}", f"★{rating:.1f}"]
                    if dense is not None:
                        parts.append(f"dense={dense:.3f}")
                    if sparse is not None:
                        parts.append(f"sparse={sparse:.1f}")
                    if txt is not None and img is not None:
                        parts.append(f"txt={txt:.3f}")
                        parts.append(f"img={img:.3f}")
                    print(f"        {' | '.join(parts)}")
            else:
                print("     (no results)")

        print(
            f"\n  TOTAL: plan={rec.plan_elapsed_s:.1f}s "
            f"search={rec.search_elapsed_ms:.0f}ms "
            f"({len(rec.slot_results)} slots)"
        )

        # OT outfit display
        if rec.ot_result and rec.ot_result.outfits:
            cats = rec.ot_result.slot_categories

            print(f"\n  {'═' * 80}")
            label = "ANCHOR SIMULATION" if args.ot_simulate else "RECOMMENDED OUTFIT"
            print(f"  {label} (OutfitTransformer CP scoring)")
            print(f"  {'═' * 80}")

            for rank, outfit in enumerate(rec.ot_result.outfits, 1):
                best = " ★ BEST" if rank == 1 and args.ot_simulate else ""
                print(f"\n  ┌- Outfit {rank}  (outfit_cp={outfit.outfit_cp:.3f}){best}")
                for i, item in enumerate(outfit.items):
                    cat = cats[i] if i < len(cats) else "?"
                    is_anchor = i == rec.ot_result.anchor_slot
                    tag = "[anchor]" if is_anchor else f"(cp={item.cp_score:.3f})"
                    print(f"  │  {tag:>10s}  {cat:>15s}:  {item.title}")
                print(f"  └{'-' * 79}")

        e2e_ms = (t_end - t_start) * 1000
        print(f"\n  ⏱  end-to-end: {e2e_ms:.0f}ms")
        print()
