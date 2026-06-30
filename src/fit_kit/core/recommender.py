import time
from typing import Literal

import open_clip
import polars as pl
import torch
from pydantic_ai import Agent

from fit_kit.core.planner import plan
from fit_kit.core.schemas import (
    Category,
    Constraints,
    OutfitItem,
    OutfitScoringResult,
    OutfitSearchSet,
    ScoredOutfit,
    SlotPlan,
    SlotQuery,
    SlotSearchResult,
)
from fit_kit.core.search import CatalogSearch
from fit_kit.utils.log_utils import get_custom_logger

logger = get_custom_logger()


# Filters that can be relaxed. Category and gender are never dropped.
_RELAXABLE_FILTERS = ["formality", "season"]


class QueryEncoder:
    """Class to encode text queries using FashionSigLIP."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        logger.info("loading FashionSigLIP for query encoding...")
        self.model, _, _ = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")
        self.tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
        self.model = self.model.to(device).eval()

    def encode(self, text: str) -> torch.Tensor:
        """Method to encode a text query to a (1, 768) embedding.

        :param text: input string
        :return: output embedding tensor
        """
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Method to encode multiple texts to (N, 768) embeddings.

        :param texts: list of input texts to encode
        :return: output batch embedding tensor
        """
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features


class QueryRouter:
    """Class to route queries to planner (outfit) or direct search (single item).

    Uses FashionSigLIP centroids computed from example queries.
    Zero additional latency at query time — just a cosine comparison.
    """

    def __init__(self, encoder: QueryEncoder) -> None:
        from fit_kit.core.query_intents import OUTFIT_QUERIES, SINGLE_ITEM_QUERIES

        logger.info("building query router centroids...")
        with torch.no_grad():
            outfit_embs = encoder.encode_batch(OUTFIT_QUERIES)
            single_embs = encoder.encode_batch(SINGLE_ITEM_QUERIES)

        self.outfit_centroid = outfit_embs.mean(dim=0, keepdim=True)
        self.outfit_centroid = self.outfit_centroid / self.outfit_centroid.norm(
            dim=-1,
            keepdim=True,
        )

        self.single_centroid = single_embs.mean(dim=0, keepdim=True)
        self.single_centroid = self.single_centroid / self.single_centroid.norm(
            dim=-1,
            keepdim=True,
        )

        logger.info(f"router ready (outfit={len(OUTFIT_QUERIES)} examples, single={len(SINGLE_ITEM_QUERIES)} examples)")  # fmt: skip

    def route(self, query_emb: torch.Tensor) -> Literal["outfit", "single_item"]:
        """Method to classify query as 'outfit' or 'single_item'.

        :param query_emb: input query embedding tensor
        :return: intent string of either "outfit" or "single_item"
        """
        outfit_sim = (query_emb @ self.outfit_centroid.T).item()
        single_sim = (query_emb @ self.single_centroid.T).item()
        intent = "outfit" if outfit_sim > single_sim else "single_item"
        logger.debug(f"router: outfit={outfit_sim:.3f} single={single_sim:.3f} → {intent}")
        return intent


class OutfitScorer:
    """Class to score outfit combinations using a trained OutfitTransformer.

    * Uses greedy CP scoring: picks the best item per slot conditioned on
    previously selected items.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        image_weight: float = 1.0,
    ) -> None:
        """Init method

        :param checkpoint_path: path to trained model checkpoint
        :param device: pytorch device to use
        :param image_weight: how much to weight image embeddings when running inference, defaults to 1.0
        """
        from fit_kit.models.outfit_transformer import OutfitTransformer, OutfitTransformerConfig

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = OutfitTransformerConfig(**ckpt["config"])
        config.image_weight = image_weight
        self.model = OutfitTransformer(config)
        self.model.load_state_dict(ckpt["model"])
        self.model = self.model.to(device).eval()
        self.device = device
        logger.info(f"outfit scorer loaded from {checkpoint_path}")

    def _score_outfit(self, embs: list[torch.Tensor]) -> float:
        """Internal method to score a complete outfit from a list of item embeddings.

        :param embs: list of item embedding tensors
        :return: outfit compatibility float score
        """
        with torch.no_grad():
            return self.model.predict_compatibility([torch.stack(embs).to(self.device)]).item()

    def _greedy_pass(
        self,
        get_emb,
        anchor_slot: int,
        anchor_idx: int,
        non_anchor: list[int],
        slot_lengths: list[int],
    ) -> tuple[dict[int, tuple[int, float]], list[torch.Tensor]]:
        """Internal method to run a single greedy pass from a given anchor.

        For each non-anchor slot, tries every candidate alongside what's already
        been selected and keeps the one with the highest compatibility score.

        :param get_emb: callable (slot_idx, item_idx) -> embedding tensor
        :param anchor_slot: index of the anchor slot
        :param anchor_idx: index of the anchor item within the anchor slot
        :param non_anchor: list of non-anchor slot indices
        :param slot_lengths: number of candidates per slot
        :return: tuple of (picks, selected_embs) where picks maps slot_idx → (item_idx, cp_score)
        """
        selected_embs = [get_emb(anchor_slot, anchor_idx)]
        picks = {anchor_slot: (anchor_idx, 1.0)}

        for slot_idx in non_anchor:
            k = slot_lengths[slot_idx]
            if k == 0:
                continue

            candidates = [get_emb(slot_idx, j) for j in range(k)]
            batch = [torch.stack(selected_embs + [c]).to(self.device) for c in candidates]

            with torch.no_grad():
                scores = self.model.predict_compatibility(batch)

            best = scores.argmax().item()
            picks[slot_idx] = (best, scores[best].item())
            selected_embs.append(candidates[best])

        return picks, selected_embs

    @staticmethod
    def _find_anchor_slot(slot_results: list[SlotSearchResult]) -> int:
        """Internal static method to find the best anchor slot (rule of thumb is shoes first, then first slot).

        :param slot_results: list of SlotSearchResult instances
        :return: index of anchor slot to use
        """
        for i, sr in enumerate(slot_results):
            cats = {c.value for c in sr.slot_query.category}
            if "shoes" in cats and len(sr.products) > 0:
                return i
        return 0

    def _get_item_embedding(self, asin: str, search: CatalogSearch) -> torch.Tensor:
        """Internal method to build a 1536-dim item embedding from catalog embeddings.
        (2 x 768-dim = 1536-dim) for text + image embeddings.

        :param asin: product identifier
        :param search: instance of CatalogSearch
        :return: concatenated item embedding comprised of text and image embeddings.
        """
        idx = search.df["asin"].to_list().index(asin)
        return torch.cat([search.image_emb[idx], search.text_emb[idx]], dim=-1)

    def score_outfit(self, asins: list[str], search: CatalogSearch) -> float:
        """Method to score a single complete outfit (list of ASINs) for compatibility.

        :param asins: list of product identifier strings
        :param search: instance of CatalogSearch
        :return: outfit compatibility float score
        """
        embs = [self._get_item_embedding(a, search) for a in asins]
        return self._score_outfit(embs)

    def greedy_select(
        self,
        slot_results: list[SlotSearchResult],
        search: CatalogSearch,
        max_anchors: int | None = None,
    ) -> OutfitScoringResult:
        """Top-level method to run greedy CP scoring for each anchor candidate.
        Avoids having to score all possible outfit combinations.

        Order of Operations (run process per anchor):
        1. Pick one anchor slot (the most "defining" category, usually shoes)
        2. Take its top search result as the starting piece
        3. Then for each remaining slot, try every candidate alongside what's already been selected and
            keep the one with the highest compatibility score.

        Set max_anchors=1 for a single greedy pass with just the top search result as anchor.

        Cost: min(n_anchors, max_anchors) * (n_slots - 1) * k forward passes.

        :param slot_results: list of SlotSearchResult instances
        :param search: instance of CatalogSearch
        :param max_anchors: limit number of anchor candidates to try (None = all)
        :return: instance of OutfitScoringResult
        """
        asin_to_idx = {a: i for i, a in enumerate(search.df["asin"].to_list())}

        def get_emb(slot_idx: int, item_idx: int) -> torch.Tensor:
            asin = slot_results[slot_idx].products["asin"].to_list()[item_idx]
            idx = asin_to_idx[asin]
            return torch.cat([search.image_emb[idx], search.text_emb[idx]], dim=-1)

        slot_cats = ["+".join(c.value for c in sr.slot_query.category) for sr in slot_results]

        if not slot_results or len(slot_results[0].products) == 0:
            return OutfitScoringResult(outfits=[], slot_categories=slot_cats)

        anchor_slot = self._find_anchor_slot(slot_results)
        slot_lengths = [len(sr.products) for sr in slot_results]
        n_anchors = slot_lengths[anchor_slot]
        if max_anchors:
            n_anchors = min(n_anchors, max_anchors)
        non_anchor = [i for i in range(len(slot_results)) if i != anchor_slot]

        all_outfits = []
        for anchor_idx in range(n_anchors):
            picks, selected_embs = self._greedy_pass(
                get_emb,
                anchor_slot,
                anchor_idx,
                non_anchor,
                slot_lengths,
            )
            outfit_cp = self._score_outfit(selected_embs)

            items = []
            for i in range(len(slot_results)):
                if i in picks:
                    idx, cp = picks[i]
                    p = slot_results[i].products.iloc[idx]
                    items.append(OutfitItem(asin=p["asin"], title=p["title"], cp_score=cp))

            all_outfits.append(ScoredOutfit(items=items, outfit_cp=outfit_cp))

        all_outfits.sort(key=lambda x: -x.outfit_cp)
        return OutfitScoringResult(
            outfits=all_outfits,
            slot_categories=slot_cats,
            anchor_slot=anchor_slot,
        )

    def greedy_select_from_embeddings(
        self,
        slot_embeddings: list[dict[str, torch.Tensor]],
        slot_products: list[list[dict]],
    ) -> list[dict]:
        """Top-level method for gRPC serving path to score outfits from pre-computed embeddings

        Order of operations:
        1. Score the "naive" outfit formed by taking the top search result per slot.
            This is the best semantic match — it might already be the best outfit.
        2. Run greedy anchor-based scoring (one full pass per anchor candidate).
        3. Return all outfits sorted by CP score, best first.


        Cost: 1 naive forward pass + (n_anchors x n_remaining_slots x k) greedy passes.
        With 5 slots x 5 candidates, that's 1 + 100 = 101 forward passes.

        :param slot_embeddings: per-slot embeddings from gRPC search results
            [{"image": (K, 768), "text": (K, 768)}, ...]
        :param slot_products: per-slot product metadata
            [[{"asin": ..., "title": ...}, ...], ...]
        :return: list of scored outfits sorted by outfit_cp (best first)
        """
        n_slots = len(slot_products)
        if n_slots == 0:
            return []

        slot_lengths = [len(prods) for prods in slot_products]
        active = [i for i in range(n_slots) if slot_lengths[i] > 0]

        def get_emb(slot_idx: int, item_idx: int) -> torch.Tensor:
            return torch.cat(
                [
                    slot_embeddings[slot_idx]["image"][item_idx],
                    slot_embeddings[slot_idx]["text"][item_idx],
                ],
                dim=-1,
            )

        def build_items(picks: dict[int, tuple[int, float]]) -> list[dict]:
            return [
                {
                    "asin": slot_products[i][idx]["asin"],
                    "title": slot_products[i][idx]["title"],
                    "cp_score": cp,
                }
                for i in range(n_slots)
                if i in picks
                for idx, cp in [picks[i]]
            ]

        # phase 1: naive (top result per slot, 1 forward pass)
        naive_embs = [get_emb(i, 0) for i in active]
        naive_cp = self._score_outfit(naive_embs)
        naive_picks = {i: (0, 1.0) for i in active}
        all_outfits = [{"items": build_items(naive_picks), "outfit_cp": naive_cp}]
        logger.info(f"OT naive: cp={naive_cp:.3f}")

        # phase 2: greedy per anchor (prefer shoes)
        anchor = 0
        for i in active:
            if any("shoe" in p.get("title", "").lower() for p in slot_products[i]):
                anchor = i
                break
        non_anchor = [i for i in range(n_slots) if i != anchor]

        for anchor_idx in range(slot_lengths[anchor]):
            picks, selected_embs = self._greedy_pass(
                get_emb,
                anchor,
                anchor_idx,
                non_anchor,
                slot_lengths,
            )
            outfit_cp = self._score_outfit(selected_embs)
            all_outfits.append({"items": build_items(picks), "outfit_cp": outfit_cp})
            logger.info(f"OT greedy[{anchor_idx}]: cp={outfit_cp:.3f}")

        all_outfits.sort(key=lambda x: -x["outfit_cp"])
        if all_outfits:
            logger.info(f"OT best: cp={all_outfits[0]['outfit_cp']:.3f}")
        return all_outfits


def recommend(
    query: str,
    search: CatalogSearch,
    encoder: QueryEncoder,
    planner_agent: Agent[None, SlotPlan],
    user_context: dict | None = None,
    top_k_per_slot: int = 5,
    embed_mode: str = "adaptive",
    alpha: float = 0.5,
    beta: float = 0.5,
    min_dense_ratio: float = 0.7,
    min_dense_score: float = 0.3,
    router: QueryRouter | None = None,
    outfit_scorer: OutfitScorer | None = None,
    simulate_outfits: bool = False,
) -> OutfitSearchSet:
    """Top-level function to generate outfit recommendations from a natural language query.

    Produces an end-to-end outfit recommendation.
    Wires the planner (LLM decomposition) to the search module (hybrid retrieval) to produce complete
        outfit recommendations from natural language queries.

    If a router is provided, queries classified as 'single_item' bypass the
    planner and go directly to hybrid search (no category filtering).

    :param query: user's natural language request
    :param search: initialized CatalogSearch instance
    :param encoder: instance of FashionSigLIP query encoder
    :param planner_agent: reusable planner Agent instance
    :param user_context: dictionary representing user profile (gender, season preferences, etc.), defaults to None
    :param top_k_per_slot: number of product candidates per outfit slot, defaults to 5
    :param embed_mode: embedding mode for dense search, defaults to "adaptive"
    :param alpha: dense vs sparse weight, defaults to 0.5
    :param beta: text vs image weight (adaptive mode). 0 = use image entirely, 1 = use text entirely, defaults to 0.5
    :param min_dense_ratio: relative dense score floor, defaults to 0.7
    :param min_dense_score: absolute dense score floor, defaults to 0.3
    :param router: optional QueryRouter instance for intent classification, defaults to None
    :param outfit_scorer: optional OutfitScorer instance for outfit scoring, defaults to None
    :param simulate_outfits: flag to score multiple outfit combinations or just score the top results from each slot, defaults to False
    :return: instance of OutfitSearchSet with plan and per-slot product results
    """
    query_emb = encoder.encode(query)

    # route: outfit (planner) vs single-item (direct search)
    intent = "outfit"
    if router is not None:
        intent = router.route(query_emb)
        logger.info(f"intent: {intent}")

    if intent == "single_item":
        return _search_single_item(
            query=query,
            query_emb=query_emb,
            search=search,
            user_context=user_context,
            top_k=top_k_per_slot,
            embed_mode=embed_mode,
            alpha=alpha,
            beta=beta,
            min_dense_ratio=min_dense_ratio,
            min_dense_score=min_dense_score,
        )

    # step 1: plan
    t0 = time.perf_counter()
    slot_plan = plan(query, user_context=user_context, agent=planner_agent)
    plan_elapsed = time.perf_counter() - t0

    logger.info(
        f"plan: "
        f"occasion={slot_plan.occasion}, "
        f"{len(slot_plan.slot_queries)} slots, "
        f"constraints={slot_plan.constraints.model_dump(exclude_none=True)} "
        f"({plan_elapsed:.1f}s)"
    )

    # step 2: search each slot with filter fallback
    slot_results = []
    total_search_ms = 0.0

    for sq in slot_plan.slot_queries:
        filters = _build_filters(sq, slot_plan)
        query_emb = encoder.encode(sq.query)

        products, filters_used, elapsed_ms = _search_with_fallback(
            search=search,
            text=sq.query,
            query_emb=query_emb,
            filters=filters,
            top_k=top_k_per_slot,
            embed_mode=embed_mode,
            alpha=alpha,
            beta=beta,
            min_dense_ratio=min_dense_ratio,
            min_dense_score=min_dense_score,
        )
        total_search_ms += elapsed_ms

        logger.info(
            f"{'+'.join(c.value for c in sq.category)}: {len(products)} "
            f"results ({elapsed_ms:.0f}ms) "
            f"query='{sq.query}' "
            f"filters={filters_used}"
        )

        slot_results.append(
            SlotSearchResult(
                slot_query=sq,
                products=products,
                filters_used=filters_used,
                elapsed_ms=elapsed_ms,
            )
        )

    # step 3: outfit scoring (optional)
    ot_result = None
    if outfit_scorer is not None and intent == "outfit" and len(slot_results) > 1:
        t_ot = time.perf_counter()
        max_anchors = None
        if not simulate_outfits:
            max_anchors = 1

        ot_result = outfit_scorer.greedy_select(slot_results, search, max_anchors=max_anchors)
        ot_ms = (time.perf_counter() - t_ot) * 1000

        best_cp = ot_result.outfits[0].outfit_cp if ot_result.outfits else 0
        logger.info(f"OT scoring: {ot_ms:.0f}ms (outfit_cp={best_cp:.3f}, {len(ot_result.outfits)} outfits)")  # fmt: skip

    return OutfitSearchSet(
        query=query,
        user_context=user_context,
        plan=slot_plan,
        slot_results=slot_results,
        plan_elapsed_s=plan_elapsed,
        search_elapsed_ms=total_search_ms,
        ot_result=ot_result,
    )


def _build_filters(sq: SlotQuery, plan: SlotPlan) -> dict:
    """Internal helper function to build search filters from a SlotQuery + global constraints.

    :param sq: instance of SlotQuery
    :param plan: instance of SlotPlan
    :return: filter dictionary
    """
    filters = {"category": [c.value for c in sq.category]}

    # global constraints
    if plan.constraints.gender:
        filters["gender"] = plan.constraints.gender
    if plan.constraints.season and plan.constraints.season != "all_season":
        filters["season"] = plan.constraints.season

    # per-slot formality
    if sq.formality:
        filters["formality"] = sq.formality

    return filters


def _search_with_fallback(
    search: CatalogSearch,
    text: str,
    query_emb: torch.Tensor,
    filters: dict,
    top_k: int,
    **search_kwargs,
) -> tuple[pl.DataFrame, dict, float]:
    """Internal function to run search with progressive filter relaxation, accumulating results.

    Order of Operations:
    1. Start with the most restrictive filters and keeps all results found.
    2. If fewer than top_k, relax filters one at a time and fill remaining
        slots with new (unseen) products.
    3. Results from stricter filters always rank above results from relaxed filters.

    :param search: instance of CatalogSearch
    :param text: search query text string
    :param query_emb: query embedding torch tensor
    :param filters: filters dictionary
    :param top_k: number of top search results to return
    :return: tuple of (products DataFrame, final filters used, total elapsed ms)
    """
    filters_used = dict(filters)
    accumulated = pl.DataFrame()
    seen_asins: set[str] = set()
    total_ms = 0.0

    # first pass: full filters
    t0 = time.perf_counter()
    products = search.query(
        text=text,
        query_emb=query_emb,
        filters=filters_used,
        top_k=top_k,
        mode="hybrid",
        **search_kwargs,
    )
    total_ms = (time.perf_counter() - t0) * 1000

    if len(products) > 0:
        accumulated = products
        seen_asins = set(products["asin"].to_list())

    remaining = top_k - len(accumulated)

    # progressive relaxation: fill remaining slots
    for drop_key in _RELAXABLE_FILTERS:
        if remaining <= 0:
            break
        if drop_key not in filters_used:
            continue

        dropped_value = filters_used.pop(drop_key)
        logger.info(f"relaxing filter: dropped {drop_key}={dropped_value}, filling {remaining} remaining slots")  # fmt: skip

        t0 = time.perf_counter()
        products = search.query(
            text=text,
            query_emb=query_emb,
            filters=filters_used,
            top_k=top_k,
            mode="hybrid",
            **search_kwargs,
        )
        total_ms += (time.perf_counter() - t0) * 1000

        if len(products) > 0:
            new_products = products.filter(~pl.col("asin").is_in(seen_asins))
            if len(new_products) > 0:
                new_products = new_products.head(remaining)
                accumulated = pl.concat([accumulated, new_products])
                seen_asins.update(new_products["asin"].to_list())
                remaining = top_k - len(accumulated)

    return products, filters_used, total_ms


def _search_single_item(
    query: str,
    query_emb: torch.Tensor,
    search: CatalogSearch,
    user_context: dict | None = None,
    top_k: int = 10,
    **search_kwargs,
) -> OutfitSearchSet:
    """Internal function to run direct search without planner — for single-item queries.

    Only applies gender filter from user_context. No category, season,
    or formality filtering — lets hybrid search handle relevance.

    :param query: search query string
    :param query_emb: search query embedding torch tensor
    :param search: instance of CatalogSearch
    :param user_context: optional user context dictionary, defaults to None
    :param top_k: number of top search results to return, defaults to 10
    :return: instance of OutfitSearchSet
    """

    filters = {}
    if user_context and user_context.get("gender"):
        filters["gender"] = user_context["gender"]

    t0 = time.perf_counter()
    products = search.query(
        text=query,
        query_emb=query_emb,
        filters=filters or None,
        top_k=top_k,
        mode="hybrid",
        **search_kwargs,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(f"direct search: {len(products)} results ({elapsed_ms:.0f}ms) query='{query}' filters={filters}")  # fmt: skip

    # wrap in OutfitSearchSet with a synthetic single-slot plan
    slot_query = SlotQuery(
        category=[Category.accessories],  # placeholder
        query=query,
    )
    slot_result = SlotSearchResult(
        slot_query=slot_query,
        products=products,
        filters_used=filters,
        elapsed_ms=elapsed_ms,
    )

    return OutfitSearchSet(
        query=query,
        user_context=user_context,
        plan=SlotPlan(
            occasion="single item search",
            slot_queries=[slot_query],
            constraints=Constraints(
                gender=user_context.get("gender") if user_context else None,
            ),
        ),
        slot_results=[slot_result],
        plan_elapsed_s=0.0,
        search_elapsed_ms=elapsed_ms,
    )
