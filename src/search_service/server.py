import os
import time
from concurrent import futures

import grpc
import numpy as np
import polars as pl
import torch

from fit_kit.utils.log_utils import get_custom_logger
from search_service import catalog_pb2, catalog_pb2_grpc

logger = get_custom_logger("search")


CATALOG_DIR = os.getenv("CATALOG_DIR", "data/catalog")
DEVICE = os.getenv("DEVICE", "cpu")
DEFAULT_ALPHA = float(os.getenv("ALPHA", "0.6"))
PORT = os.getenv("GRPC_PORT", "50051")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))


class CatalogSearchService(catalog_pb2_grpc.CatalogSearchServicer):
    """gRPC service wrapping CatalogSearch for hybrid retrieval.

    gRPC server for catalog search.

    Loads the catalog, embeddings, and BM25 index once at startup.
    Serves hybrid search requests from the worker over gRPC.
    """

    def __init__(self) -> None:
        from fit_kit.core.schemas import Category
        from fit_kit.core.search import CatalogSearch

        logger.info(f"loading catalog from {CATALOG_DIR} ...")
        self.search = CatalogSearch(
            CATALOG_DIR,
            device=DEVICE,
            embed_mode="adaptive",
            alpha=DEFAULT_ALPHA,
        )
        self.categories = [c.value for c in Category]
        self.genders = ["men", "women", "boys", "girls", "unisex_adults", "unisex_kids"]
        logger.info(f"search service ready ({len(self.search.df)} products)")

    def Search(self, request, context):
        """Hybrid search with progressive filter fallback."""
        t0 = time.perf_counter()

        query_emb = (
            torch.tensor(np.array(request.embedding, dtype=np.float32))
            .unsqueeze(0)
            .to(self.search.device)
        )
        text = request.text
        top_k = request.top_k or 8
        alpha = request.alpha or DEFAULT_ALPHA
        beta = request.beta or 0.5
        min_dense_ratio = request.min_dense_ratio or 0.0
        min_dense_score = request.min_dense_score or 0.0

        # build filters dict
        filters = {}
        if request.filters.category:
            filters["category"] = list(request.filters.category)
        if request.filters.gender:
            filters["gender"] = request.filters.gender
        if request.filters.season:
            filters["season"] = request.filters.season
        if request.filters.formality:
            filters["formality"] = list(request.filters.formality)
        if request.filters.color:
            filters["color"] = list(request.filters.color)

        # progressive filter fallback — accumulate results from strict → relaxed
        products_df, filters_used = self._search_with_fallback(
            text=text,
            query_emb=query_emb,
            filters=filters,
            top_k=top_k,
            alpha=alpha,
            beta=beta,
            min_dense_ratio=min_dense_ratio,
            min_dense_score=min_dense_score,
        )

        # build response
        products = []
        include_emb = request.include_embeddings

        for row in products_df.iter_rows(named=True):
            p = catalog_pb2.Product(
                asin=row.get("asin", ""),
                title=row.get("title", ""),
                score=float(row.get("score", 0)),
                average_rating=float(row.get("average_rating", 0)),
                clip_color=row.get("clip_color", ""),
                clip_category=row.get("clip_category", ""),
                clip_formality=row.get("clip_formality", "") or "",
                clip_season=row.get("clip_season", "") or "",
                clip_gender=row.get("clip_gender", "") or "",
                image_url=row.get("image_url", ""),
                dense_score=float(row.get("dense_score", 0)),
                sparse_score=float(row.get("sparse_score", 0)),
                text_score=float(row.get("text_score", 0)),
                image_score=float(row.get("image_score", 0)),
                # heavy fields (features, details_json, review_text) omitted
                # — fetched on-demand via GetProductDetail
            )
            if include_emb:
                idx = row.get("_row_idx")
                if idx is not None and self.search.image_emb is not None:
                    img_emb = self.search.image_emb[idx].cpu().numpy()
                    p.image_embedding.extend(img_emb.tolist())
                if idx is not None and self.search.text_emb is not None:
                    txt_emb = self.search.text_emb[idx].cpu().numpy()
                    p.text_embedding.extend(txt_emb.tolist())

            products.append(p)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return catalog_pb2.SearchResponse(
            products=products,
            filters_used={k: str(v) for k, v in filters_used.items()},
            elapsed_ms=round(elapsed_ms, 2),
        )

    def GetCatalogInfo(self, request, context):
        """Return catalog metadata."""
        return catalog_pb2.CatalogInfo(
            product_count=len(self.search.df),
            categories=self.categories,
            genders=self.genders,
        )

    def GetEmbeddings(self, request, context):
        """Return image + text embeddings for a list of ASINs."""
        import polars as pl

        asin_to_idx = {}
        for asin in request.asins:
            rows = self.search.df.filter(pl.col("asin") == asin)
            if len(rows) > 0:
                asin_to_idx[asin] = rows["_row_idx"][0]

        products = []
        for asin in request.asins:
            idx = asin_to_idx.get(asin)
            if idx is None:
                continue
            pe = catalog_pb2.ProductEmbedding(asin=asin)
            pe.image_embedding.extend(self.search.image_emb[idx].cpu().numpy().tolist())
            pe.text_embedding.extend(self.search.text_emb[idx].cpu().numpy().tolist())
            products.append(pe)

        return catalog_pb2.EmbeddingResponse(products=products)

    def GetProductDetail(self, request, context):
        """Return full product details for modal display."""
        import polars as pl

        rows = self.search.df.filter(pl.col("asin") == request.asin)
        if len(rows) == 0:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Product {request.asin} not found")
            return catalog_pb2.ProductDetailResponse()

        row = rows.row(0, named=True)
        return catalog_pb2.ProductDetailResponse(
            asin=row.get("asin", ""),
            title=row.get("title", ""),
            features=row.get("features", []) or [],
            details_json=row.get("details_json", "") or "",
            reviews_json=row.get("reviews_json", "") or row.get("review_text", "") or "",
            average_rating=float(row.get("average_rating", 0)),
            clip_category=row.get("clip_category", "") or "",
            clip_color=row.get("clip_color", "") or "",
            clip_formality=row.get("clip_formality", "") or "",
            clip_season=row.get("clip_season", "") or "",
            clip_gender=row.get("clip_gender", "") or "",
        )

    def _search_with_fallback(
        self,
        text,
        query_emb,
        filters,
        top_k,
        alpha,
        beta=0.5,
        min_dense_ratio=0.0,
        min_dense_score=0.0,
    ):
        """Progressive filter relaxation: drop color → formality → season. Keep category + gender."""

        accumulated = []
        seen_asins = set()
        filters_used = dict(filters)
        relaxable = ["color", "formality", "season"]
        search_kwargs = dict(
            beta=beta,
            min_dense_ratio=min_dense_ratio,
            min_dense_score=min_dense_score,
        )

        # tier 1: all filters
        results = self.search.query(
            text=text,
            query_emb=query_emb,
            filters=filters,
            top_k=top_k,
            alpha=alpha,
            **search_kwargs,
        )
        new = results.filter(~pl.col("asin").is_in(seen_asins))
        accumulated.append(new)
        seen_asins.update(new["asin"].to_list())

        if len(seen_asins) >= top_k:
            return pl.concat(accumulated).head(top_k), filters_used

        # progressive relaxation
        for drop_key in relaxable:
            if len(seen_asins) >= top_k:
                break
            if drop_key not in filters_used:
                continue

            relaxed = {k: v for k, v in filters_used.items() if k != drop_key}
            filters_used = relaxed
            results = self.search.query(
                text=text,
                query_emb=query_emb,
                filters=relaxed,
                top_k=top_k,
                alpha=alpha,
                **search_kwargs,
            )
            new = results.filter(~pl.col("asin").is_in(seen_asins))
            accumulated.append(new)
            seen_asins.update(new["asin"].to_list())

        return pl.concat(accumulated).head(top_k), filters_used


def serve():
    """Start the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    service = CatalogSearchService()
    catalog_pb2_grpc.add_CatalogSearchServicer_to_server(service, server)

    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    logger.info(f"search service listening on port {PORT}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
