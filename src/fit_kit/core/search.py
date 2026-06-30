import json
import re
from pathlib import Path

import bm25s
import polars as pl
import Stemmer
import torch
from safetensors.torch import load_file

from fit_kit.utils.device_utils import get_device
from fit_kit.utils.log_utils import get_custom_logger

logger = get_custom_logger()


class CatalogSearch:
    """Hybrid search over the enriched product catalog.

    Combines exact kNN (PyTorch matmul) with BM25 sparse search,
    with Polars-based pre-filtering on product attributes.

    Usage Example:
    ```python
    from fit_kit.search import CatalogSearch

    search = CatalogSearch("data/catalog", device="auto")
    results = search.query(
        text="casual summer sundress",
        filters={
            "category": "dresses",
            "gender": "women",
            "season": "summer",
        },
        top_k=20,
        mode="hybrid",  # "dense", "sparse", or "hybrid"
    )
    ```
    """

    def __init__(
        self,
        catalog_dir: str,
        device: str = "auto",
        embed_mode: str = "adaptive",
        alpha: float = 0.7,
        min_title_words: int = 2,
        min_rating: float = 3.5,
        min_reviews: int = 5,
        english_only: bool = True,
        bm25_mmap: bool = True,
        rebuild_bm25: bool = False,
    ) -> None:
        """Init method

        :param catalog_dir: path to catalog directory (parquet + safetensors)
        :param device: torch device ("auto", "cuda", "cpu", "mps"), defaults to "auto"
        :param embed_mode: which embeddings to use for dense search ("text", "image", "fused"), defaults to "adaptive"
        :param alpha: weight for dense scores in hybrid search (1-alpha for BM25), defaults to 0.7
        :param min_title_words: minimum words in title to include product (filters junk), defaults to 2
        :param min_rating: minimum average rating to include product, defaults to 3.5
        :param min_reviews: minimum number of reviews to include product, defaults to 5
        :param english_only: filter out non-English titles (>50% Latin characters), defaults to True
        :param bm25_mmap: memory-map the BM25 index from disk (lower memory usage), defaults to True
        :param rebuild_bm25: force rebuild of BM25 index even if saved version exists, defaults to False
        """
        self.device = get_device(device)
        self.alpha = alpha
        self.embed_mode = embed_mode

        catalog_path = Path(catalog_dir)
        logger.info(f"loading catalog from {catalog_path}...")

        # load catalog with quality filters
        df = pl.read_parquet(catalog_path / "catalog.parquet")
        n_total = len(df)

        # apply all index-time filters
        df = df.with_row_index("_orig_idx")
        mask = pl.lit(True)
        if min_title_words > 1:
            mask = mask & (pl.col("title").str.split(" ").list.len() >= min_title_words)
        if min_rating > 0:
            mask = mask & (pl.col("average_rating") >= min_rating)
        if min_reviews > 0:
            mask = mask & (pl.col("rating_number") >= min_reviews)
        if english_only:
            # require >50% Latin characters in title
            mask = mask & (
                pl.col("title").str.count_matches(r"[a-zA-Z]")
                > pl.col("title").str.len_chars() * 0.5
            )

        df = df.filter(mask)
        keep_indices = df["_orig_idx"].to_list()
        df = df.drop("_orig_idx")
        # n_filtered = n_total - len(df)
        logger.info(
            f"{n_total:,} in parquet → {len(df):,} after filters "
            f"(title≥{min_title_words}w, rating≥{min_rating}, reviews≥{min_reviews}"
            f"{', english_only' if english_only else ''})"
        )

        self.df = df.with_row_index("_row_idx")
        logger.info(f"{len(self.df):,} products loaded")

        # load embeddings (select only kept rows)
        text_emb = load_file(str(catalog_path / "text_embeddings.safetensors"))["embeddings"]
        image_emb_data = load_file(str(catalog_path / "image_embeddings.safetensors"))
        image_emb_all = image_emb_data["embeddings"]
        image_indices_all = image_emb_data["indices"].long()

        # filter to kept rows
        self.text_emb = text_emb[keep_indices].to(self.device)

        # remap image indices to new row numbering
        old_to_new = {old: new for new, old in enumerate(keep_indices)}
        keep_set = set(keep_indices)
        new_image_indices = []
        new_image_emb_rows = []
        for i, orig_idx in enumerate(image_indices_all.tolist()):
            if orig_idx in keep_set:
                new_image_indices.append(old_to_new[orig_idx])
                new_image_emb_rows.append(i)

        full_image_emb = torch.zeros_like(self.text_emb)
        if new_image_indices:
            idx_tensor = torch.tensor(new_image_indices, dtype=torch.long)
            full_image_emb[idx_tensor] = image_emb_all[new_image_emb_rows].to(self.device)
        self.image_emb = full_image_emb

        # mean-center embeddings to close the modality gap
        # text and image embeddings cluster in different regions of SigLIP's shared space.
        # centering each modality removes the systematic offset so cross-modal
        # similarities are on the same scale as within-modal similarities.
        self.text_mean = self.text_emb.mean(dim=0)
        self.image_mean = self.image_emb.mean(dim=0)

        self.text_emb = self.text_emb - self.text_mean
        self.text_emb = self.text_emb / self.text_emb.norm(dim=-1, keepdim=True)

        self.image_emb = self.image_emb - self.image_mean
        self.image_emb = self.image_emb / self.image_emb.norm(dim=-1, keepdim=True)

        # pre-compute fused embeddings (from centered versions)
        self.fused_emb = self.text_emb + self.image_emb
        self.fused_emb = self.fused_emb / self.fused_emb.norm(dim=-1, keepdim=True)

        logger.info(f"embeddings: {tuple(self.text_emb.shape)} on {self.device}")

        # BM25 index (load from disk or build + save)
        self.stemmer = Stemmer.Stemmer("english")
        bm25_dir = catalog_path / "bm25_index"

        if bm25_dir.exists() and not rebuild_bm25:
            logger.info(f"loading BM25 index from {bm25_dir} (mmap={bm25_mmap})...")
            self.bm25 = bm25s.BM25.load(str(bm25_dir), mmap=bm25_mmap)
            n_bm25 = self.bm25.scores["num_docs"]
            if n_bm25 != len(self.df):
                logger.warning(f"BM25 index has {n_bm25} docs but catalog has {len(self.df)} rows — rebuilding")  # fmt: skip
                rebuild_bm25 = True
            else:
                logger.info(f"BM25 index: {n_bm25:,} documents (loaded from disk)")

        if not hasattr(self, "bm25") or rebuild_bm25:
            logger.info("building BM25 index...")
            corpus = self._build_bm25_corpus()
            corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=self.stemmer)
            self.bm25 = bm25s.BM25()
            self.bm25.index(corpus_tokens)
            logger.info(f"BM25 index: {len(corpus):,} documents")

            # save for fast loading next time
            bm25_dir.mkdir(parents=True, exist_ok=True)
            self.bm25.save(str(bm25_dir))
            logger.info(f"BM25 index saved to {bm25_dir}")

        logger.info(f"ready (device={self.device}, embed_mode={embed_mode}, alpha={alpha})")

    def _build_bm25_corpus(self) -> list[str]:
        """Internal method to build BM25 corpus from title + all features + key details + review data.
        No word limit — BM25 benefits from more keywords.

        :return: corpus as a list of strings
        """
        corpus = []
        titles = self.df["title"].to_list()
        features_col = self.df["features"].to_list()
        details_col = self.df["details_json"].to_list()

        # review text (optional — may not exist if prepared without --reviews-path)
        has_reviews = "review_text" in self.df.columns
        if has_reviews:
            review_col = self.df["review_text"].to_list()
            logger.info("including review text in BM25 corpus")
        else:
            review_col = ["" for _ in titles]

        detail_keys = {
            "Department",
            "Material",
            "Color",
            "Style",
            "Fabric Type",
            "Closure Type",
            "Pattern",
            "Brand",
        }

        for title, features, details_json, review_text in zip(
            titles, features_col, details_col, review_col
        ):
            parts = [title]

            # all feature bullets
            if features:
                for f in features:
                    if isinstance(f, str) and f.strip():
                        parts.append(f.strip())

            # key detail fields
            try:
                details = json.loads(details_json) if isinstance(details_json, str) else {}
            except (json.JSONDecodeError, TypeError):
                details = {}
            for key in detail_keys:
                val = details.get(key)
                if val and isinstance(val, str) and val.strip():
                    parts.append(f"{key}: {val.strip()}")

            # raw review text — BM25 handles term weighting natively
            if review_text:
                parts.append(review_text)

            corpus.append(". ".join(parts))

        return corpus

    def _get_embeddings(self, mode: str = None) -> torch.Tensor:
        """Internal method to get the embedding tensor for the specified mode

        :param mode: one of ("text", "image", "fused", "adaptive"), defaults to None
        :raises ValueError: "unknown embed_mode" if not one of the correct mode options above.
        :return: torch tensor representing embedding for the given modality selection
        """
        mode = mode or self.embed_mode
        if mode == "text":
            return self.text_emb
        elif mode == "image":
            return self.image_emb
        elif mode in ("fused", "adaptive"):
            return self.fused_emb  # adaptive uses fused for dedup
        raise ValueError(f"unknown embed_mode: {mode}")

    def _encode_query(self, text: str, model, tokenizer) -> torch.Tensor:
        """Internal method to encode a query string with FashionSigLIP model.

        :param text: input query string
        :param model: instance of FashionSigLIP model
        :param tokenizer: isntance of FashionSigLIP tokenizer
        :return: torch tensor embedding output from FashionSigLIP model
        """
        tokens = tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features  # (1, 768)

    def _apply_filters(self, filters: dict | None) -> pl.DataFrame:
        """Internal method to apply pre-filter catalog using Polars

        :param filters: filters dictionary {field: value}
        :return: filtered polars DataFrame
        """
        if not filters:
            return self.df

        mask = pl.lit(True)

        if "category" in filters:
            cats = filters["category"]
            if isinstance(cats, str):
                cats = [cats]
            mask = mask & pl.col("clip_category").is_in(cats)

        if "gender" in filters:
            genders = filters["gender"]
            if isinstance(genders, str):
                # expand to include the appropriate unisex variant
                if genders in ("men", "women"):
                    genders = [genders, "unisex_adults"]
                elif genders in ("boys", "girls"):
                    genders = [genders, "unisex_kids"]
                elif genders == "unisex_adults":
                    genders = ["men", "women", "unisex_adults"]
                elif genders == "unisex_kids":
                    genders = ["boys", "girls", "unisex_kids"]
                else:
                    genders = [genders]
            mask = mask & pl.col("clip_gender").is_in(genders)

        if "season" in filters:
            mask = mask & (
                (pl.col("clip_season") == filters["season"])
                | pl.col("clip_season").is_null()  # non-clothing passes through
            )

        if "formality" in filters:
            formalities = filters["formality"]
            if isinstance(formalities, str):
                formalities = [formalities]
            mask = mask & (
                pl.col("clip_formality").is_in(formalities) | pl.col("clip_formality").is_null()
            )

        if "color" in filters:
            colors = filters["color"]
            if isinstance(colors, str):
                colors = [colors]

            # rely on image features for "color" instead of text
            mask = mask & pl.col("clip_color_image").is_in(colors)

        if "min_rating" in filters:
            mask = mask & (pl.col("average_rating") >= filters["min_rating"])

        if "min_reviews" in filters:
            mask = mask & (pl.col("rating_number") >= filters["min_reviews"])

        return self.df.filter(mask)

    def query(
        self,
        text: str,
        query_emb: torch.Tensor = None,
        filters: dict = None,
        top_k: int = 20,
        mode: str = "hybrid",
        embed_mode: str = None,
        alpha: float = None,
        beta: float = None,
        min_dense_ratio: float = 0.0,
        min_dense_score: float = 0.0,
    ) -> pl.DataFrame:
        """Top-level method to search the catalog

        :param text: query string (used for BM25 and/or dense encoding)
        :param query_emb: optional pre-computed query embedding (skips encoding), defaults to None
        :param filters: optional filters dictionary of attribute filters (category, gender, season, etc.), defaults to None
        :param top_k: number of top results to return, defaults to 20
        :param mode: retrieval mode - "dense" (kNN only), "sparse" (BM25 only), or "hybrid", defaults to "hybrid"
        :param embed_mode: override default embedding mode for this query ("text", "image", "fused", "adaptive"),
            defaults to None
        :param alpha: override default alpha for this query (dense vs sparse weight), defaults to None
        :param beta: text vs image weight for adaptive mode (0.0=image only, 1.0=text only), defaults to None
        :param min_dense_ratio: drop results with dense_score < ratio * max (relative), defaults to 0.0
        :param min_dense_score: absolute minimum dense score floor, defaults to 0.0
        :raises ValueError: unknwown retrieval mode string passed in
        :return: polars pl.DataFrame with top_k results and scores
        """
        alpha = alpha if alpha is not None else self.alpha
        beta = beta if beta is not None else getattr(self, "beta", 0.5)
        em = embed_mode or self.embed_mode

        # pre-filter
        filtered = self._apply_filters(filters)
        if len(filtered) == 0:
            return filtered.head(0)

        indices = torch.tensor(
            filtered["_row_idx"].to_numpy(),
            dtype=torch.long,
            device=self.device,
        )
        n_filtered = len(indices)
        k = min(top_k, n_filtered)

        # dense scores
        dense_scores = None
        text_scores_debug = None
        image_scores_debug = None
        if mode in ("dense", "hybrid") and query_emb is not None:
            # center query embedding (text modality) to match centered catalog embeddings
            query_centered = query_emb - self.text_mean
            query_centered = query_centered / query_centered.norm(dim=-1, keepdim=True)

            if em == "adaptive":
                # query-adaptive: separate text and image scoring
                text_subset = self.text_emb[indices]
                image_subset = self.image_emb[indices]
                text_scores_debug = (query_centered @ text_subset.T).squeeze(0)
                image_scores_debug = (query_centered @ image_subset.T).squeeze(0)
                dense_scores = beta * text_scores_debug + (1 - beta) * image_scores_debug
            else:
                embeddings = self._get_embeddings(embed_mode)
                subset_emb = embeddings[indices]
                dense_scores = (query_centered @ subset_emb.T).squeeze(0)  # (M,) on device

        # sparse scores (BM25)
        sparse_scores = None
        if mode in ("sparse", "hybrid") and text:
            query_tokens = bm25s.tokenize(text, stopwords="en", stemmer=self.stemmer)
            # retrieve top matches from full corpus
            bm25_k = min(len(self.df), max(n_filtered * 2, 1000))
            result_ids, result_scores = self.bm25.retrieve(query_tokens, k=bm25_k)

            # build score lookup: catalog row → BM25 score (vectorized)
            score_lookup = torch.zeros(len(self.df), dtype=torch.float32, device=self.device)
            bm25_idx = torch.tensor(result_ids[0], dtype=torch.long, device=self.device)
            bm25_vals = torch.tensor(result_scores[0], dtype=torch.float32, device=self.device)
            score_lookup.scatter_(0, bm25_idx, bm25_vals)
            sparse_scores = score_lookup[indices]  # select filtered subset

        # combine and rank
        if mode == "dense":
            final_scores = dense_scores
        elif mode == "sparse":
            final_scores = sparse_scores
        elif mode == "hybrid":
            if dense_scores is not None and sparse_scores is not None:
                final_scores = _weighted_rrf(
                    dense_scores,
                    sparse_scores,
                    rrf_k=60,
                    dense_weight=alpha,
                    sparse_weight=1 - alpha,
                )
            else:
                final_scores = dense_scores if dense_scores is not None else sparse_scores
        else:
            raise ValueError(f"unknown mode: {mode}")

        if final_scores is None:
            return filtered.head(0)

        # top-k with diversity
        # retrieve extra candidates for dedup headroom
        candidate_k = min(k * 10, n_filtered)
        top_scores, top_idx = final_scores.topk(candidate_k)

        candidate_df = filtered[top_idx.cpu().tolist()]
        candidate_df = candidate_df.with_columns(pl.Series("score", top_scores.cpu().tolist()))

        # add component scores for debugging
        if mode == "hybrid":
            if dense_scores is not None:
                candidate_df = candidate_df.with_columns(
                    pl.Series("dense_score", dense_scores[top_idx].cpu().tolist())
                )
            if sparse_scores is not None:
                candidate_df = candidate_df.with_columns(
                    pl.Series("sparse_score", sparse_scores[top_idx].cpu().tolist())
                )
            if text_scores_debug is not None:
                candidate_df = candidate_df.with_columns(
                    pl.Series("text_score", text_scores_debug[top_idx].cpu().tolist())
                )
            if image_scores_debug is not None:
                candidate_df = candidate_df.with_columns(
                    pl.Series("image_score", image_scores_debug[top_idx].cpu().tolist())
                )

        # deduplicate near-identical products (title + embedding similarity)
        embeddings = self._get_embeddings(embed_mode)
        # (candidate_k, D)
        candidate_emb = embeddings[indices[top_idx]]

        # apply dense score floor — drop results far below the best dense match
        has_floor = min_dense_ratio > 0 or min_dense_score > 0
        if has_floor and dense_scores is not None and "dense_score" in candidate_df.columns:
            max_dense = candidate_df["dense_score"].max()
            relative_floor = max_dense * min_dense_ratio if min_dense_ratio > 0 else 0.0
            floor = max(min_dense_score, relative_floor)
            floor_mask = candidate_df["dense_score"] >= floor
            candidate_df = candidate_df.filter(floor_mask)
            keep_mask = floor_mask.to_numpy()
            candidate_emb = candidate_emb[keep_mask]

        result = _deduplicate(candidate_df, top_k=k, embeddings=candidate_emb)

        return result


def _normalize_title(title: str) -> str:
    """Internal helper function to normalize title for deduplication — strip size, color, variant suffixes.

    :param title: input product title string
    :return: normalized title string
    """

    # remove parenthetical content: (Large, Black), (XL), (Size 10), etc.
    t = re.sub(r"\([^)]*\)", "", title)
    # remove trailing size/color patterns: "... Black XL", "... Size 10"
    t = re.sub(r"\s+(?:X{0,3}[SML]|XX?L|\d+[.]\d+|\d{1,2})$", "", t.strip())
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def _deduplicate(
    df: pl.DataFrame,
    top_k: int,
    embeddings: torch.Tensor = None,
    sim_threshold: float = 0.95,
    title_overlap_threshold: float = 0.6,
) -> pl.DataFrame:
    """Internal helper function to remove near-duplicate products using title normalization + embedding similarity.

    Two products are considered duplicates if:
    1. their normalized titles match exactly, OR
    2. their title word overlap (Jaccard) exceeds title_overlap_threshold
        AND their embedding cosine similarity exceeds sim_threshold

    :param df: input polars dataframe representing product metadata
    :param top_k: number of top k results
    :param embeddings: product torch embeddings, defaults to None
    :param sim_threshold: similarity threshold, defaults to 0.95
    :param title_overlap_threshold: threshold for jaccard similarity of word overlap, defaults to 0.6
    :return: updated product dataframe with only unique items kept (duplicates removed)
    """
    if len(df) == 0:
        return df

    seen_titles = set()
    keep_indices = []
    kept_embeddings = []
    kept_word_sets = []

    titles = df["title"].to_list()
    for i, title in enumerate(titles):
        # title-based dedup (exact normalized match)
        norm = _normalize_title(title)
        if norm in seen_titles:
            continue

        # text overlap + embedding dedup
        words_i = set(norm.split())
        is_dup = False
        if embeddings is not None and kept_word_sets:
            emb = embeddings[i]
            for j, kept_words in enumerate(kept_word_sets):
                # check title word overlap first (cheap)
                intersection = len(words_i & kept_words)
                union = len(words_i | kept_words)
                jaccard = intersection / union if union > 0 else 0
                if jaccard < title_overlap_threshold:
                    continue
                # high text overlap — now check embedding similarity (expensive)
                sim = (emb @ kept_embeddings[j]).item()
                if sim > sim_threshold:
                    is_dup = True
                    break

        if is_dup:
            continue

        seen_titles.add(norm)
        keep_indices.append(i)
        if embeddings is not None:
            kept_embeddings.append(embeddings[i])
        kept_word_sets.append(words_i)
        if len(keep_indices) >= top_k:
            break

    return df[keep_indices]


def _weighted_rrf(
    dense_scores: torch.Tensor,
    sparse_scores: torch.Tensor,
    rrf_k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> torch.Tensor:
    """Internal helper function to run Weighted Reciprocal Rank Fusion.

    RRF_score(doc) = w_dense / (rrf_k + rank_dense(doc)) + w_sparse / (rrf_k + rank_sparse(doc))

    Uses ranks instead of raw scores — robust to different score distributions.

    :param dense_scores: raw dense similarity scores (M,) on device
    :param sparse_scores: raw BM25 scores (M,) on device
    :param rrf_k: RRF constant (default 60, standard in literature), defaults to 60
    :param dense_weight: weight for dense retrieval, defaults to 0.7
    :param sparse_weight: weight for sparse retrieval, defaults to 0.3
    :return: RRF scores torch tensor (M,) on same device
    """
    n = len(dense_scores)

    # compute ranks (0-indexed, lower rank = higher score)
    dense_ranks = torch.empty(n, device=dense_scores.device)
    dense_ranks[dense_scores.argsort(descending=True)] = torch.arange(
        n,
        device=dense_scores.device,
        dtype=torch.float32,
    )

    sparse_ranks = torch.empty(n, device=sparse_scores.device)
    sparse_ranks[sparse_scores.argsort(descending=True)] = torch.arange(
        n,
        device=sparse_scores.device,
        dtype=torch.float32,
    )

    # RRF scores
    rrf_scores = dense_weight / (rrf_k + dense_ranks) + sparse_weight / (rrf_k + sparse_ranks)

    return rrf_scores
