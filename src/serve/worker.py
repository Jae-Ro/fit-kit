import asyncio
import json
import os
import time

import redis.asyncio as aioredis
from taskiq import Context, TaskiqDepends, TaskiqEvents, TaskiqState
from taskiq_redis import RedisStreamBroker

from fit_kit.utils.log_utils import get_custom_logger
from serve.schemas import EventType, SSEEvent, catalog_count_key, task_channel, worker_status_key

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SEARCH_SERVICE_URL = os.getenv("SEARCH_SERVICE_URL", "localhost:50051")

broker = RedisStreamBroker(REDIS_URL)

logger = get_custom_logger("worker")


# ------------------------------------------------------------------------------
# TaskIQ Worker
# Worker and task definitions

# Shared between backend (calls .kiq()) and worker (executes tasks).
# ML imports are deferred to startup/task body so the backend stays lightweight.

# The worker calls the search service via gRPC for catalog retrieval,
# keeping the catalog data out of the worker process.
# ------------------------------------------------------------------------------


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def startup(state: TaskiqState, **kwargs):
    """Load ML models and connect to search service."""
    # set thread pool for asyncio.to_thread()
    import concurrent.futures

    import grpc.aio
    import torch

    from fit_kit.core.planner import create_planner_agent
    from fit_kit.core.recommender import OutfitScorer, QueryEncoder, QueryRouter
    from search_service import catalog_pb2, catalog_pb2_grpc

    ot_checkpoint = os.getenv("OT_CHECKPOINT", "checkpoints/cp_best.pt")
    device = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    planner_model = os.getenv("PLANNER_MODEL", "openai:gpt-5.5.")
    image_weight = float(os.getenv("OT_IMAGE_WEIGHT", "10.0"))

    # search params (configurable via env)
    state.search_params = {
        "alpha": float(os.getenv("SEARCH_ALPHA", "0.6")),
        "beta": float(os.getenv("SEARCH_BETA", "0.2")),
        "min_dense_ratio": float(os.getenv("SEARCH_MIN_DENSE_RATIO", "0.7")),
        "min_dense_score": float(os.getenv("SEARCH_MIN_DENSE_SCORE", "0.3")),
        "top_k": int(os.getenv("SEARCH_TOP_K", "8")),
    }

    logger.info("loading FashionSigLIP encoder...")
    state.encoder = QueryEncoder(device)

    logger.info("building query router...")
    state.router = QueryRouter(state.encoder)

    if os.path.exists(ot_checkpoint):
        logger.info(f"loading OutfitTransformer from {ot_checkpoint}...")
        state.scorer = OutfitScorer(ot_checkpoint, device, image_weight)
    else:
        logger.warning(f"no OT checkpoint at {ot_checkpoint}, scoring disabled")
        state.scorer = None

    logger.info(f"creating planner agent (model={planner_model})...")
    state.planner_agent = create_planner_agent(model=planner_model)

    # connect to search service via async gRPC
    logger.info(f"connecting to search service at {SEARCH_SERVICE_URL}...")

    state.grpc_channel = grpc.aio.insecure_channel(
        SEARCH_SERVICE_URL,
        options=[("grpc.max_receive_message_length", 50 * 1024 * 1024)],
    )
    state.search_stub = catalog_pb2_grpc.CatalogSearchStub(state.grpc_channel)

    # fetch catalog info and store in Redis
    state.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        info = await state.search_stub.GetCatalogInfo(catalog_pb2.Empty())
        await state.redis.set(catalog_count_key(), str(info.product_count))
        logger.info(f"catalog: {info.product_count} products")
    except Exception as e:
        logger.warning(f"could not fetch catalog info: {e}")

    max_workers = int(os.getenv("THREAD_POOL_SIZE", os.cpu_count() or 4))
    loop = asyncio.get_event_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=max_workers))

    await state.redis.set(worker_status_key(), "ready")
    logger.info(f"worker ready (thread_pool={max_workers})")


@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def shutdown(state: TaskiqState, **kwargs):
    """Clean up connections."""
    if hasattr(state, "redis"):
        await state.redis.set(worker_status_key(), "offline")
        await state.redis.close()
    if hasattr(state, "grpc_channel"):
        await state.grpc_channel.close()


async def publish_event(
    redis_conn: aioredis.Redis,
    task_id: str,
    event_type: EventType,
    data: dict | None = None,
):
    """Publish an SSE-formatted event to the task's Redis pub/sub channel."""
    evt = SSEEvent(event=event_type, data=data or {})
    await redis_conn.publish(task_channel(task_id), evt.to_sse())


def _products_to_dicts(products) -> list[dict]:
    """Convert gRPC Product messages to JSON-serializable dicts.
    Only includes display-critical fields — full details fetched on modal open
    """
    from urllib.parse import quote

    return [
        {
            "asin": p.asin,
            "title": p.title,
            "score": round(p.score, 4),
            "average_rating": round(p.average_rating, 1),
            "clip_color": p.clip_color,
            "clip_category": p.clip_category,
            "clip_formality": p.clip_formality,
            "clip_season": p.clip_season,
            "clip_gender": p.clip_gender,
            "image_url": (
                f"/api/images/{p.asin}.jpg?fallback={quote(p.image_url, safe='')}"
                if p.asin and p.image_url
                else f"/api/images/{p.asin}.jpg"
                if p.asin
                else ""
            ),
            "fallback_image_url": p.image_url,
            "dense_score": round(p.dense_score, 4),
            "sparse_score": round(p.sparse_score, 4),
            "text_score": round(p.text_score, 4),
            "image_score": round(p.image_score, 4),
        }
        for p in products
    ]


@broker.task(retry_on_error=True, max_retries=2)
async def recommend_task(
    task_id: str,
    query: str,
    user_context: dict | None,
    top_k: int = 5,
    search_overrides: dict | None = None,
    context: Context = TaskiqDepends(),
) -> None:
    """Run the full recommendation pipeline.

    Async architecture:
      1. LLM planner: native async (await agent.run)
      2. Search: async gRPC calls to search service
      3. Compute: asyncio.to_thread for encode/OT scoring
    """
    from fit_kit.core.planner import aplan
    from search_service import catalog_pb2

    state = context.state
    r = state.redis

    # merge user overrides with env defaults
    sp = {**state.search_params}
    if search_overrides:
        if "alpha" in search_overrides:
            sp["alpha"] = search_overrides["alpha"]
        if "beta" in search_overrides:
            sp["beta"] = search_overrides["beta"]
        if "filters" in search_overrides:
            sp["user_filters"] = search_overrides["filters"]
    sp["top_k"] = top_k

    try:
        # encode & route (CPU)
        query_emb = state.encoder.encode(query)
        intent = state.router.route(query_emb)

        await publish_event(
            r,
            task_id,
            EventType.ROUTING,
            {
                "intent": intent,
                "query": query,
            },
        )

        # single-item path
        if intent == "single_item":
            gender = (user_context or {}).get("gender", "")
            filters = catalog_pb2.Filters(gender=gender)

            # apply user filters from advanced search
            user_filters = sp.get("user_filters", {})
            if user_filters.get("category"):
                filters.category.extend(user_filters["category"])
            if user_filters.get("season"):
                filters.season = (
                    user_filters["season"][0]
                    if isinstance(user_filters["season"], list)
                    else user_filters["season"]
                )
            if user_filters.get("formality"):
                filters.formality.extend(user_filters["formality"])
            if user_filters.get("color"):
                filters.color.extend(user_filters["color"])

            response = await state.search_stub.Search(
                catalog_pb2.SearchRequest(
                    embedding=query_emb.squeeze(0).cpu().numpy().tolist(),
                    text=query,
                    filters=filters,
                    top_k=sp["top_k"],
                    alpha=sp["alpha"],
                    beta=sp["beta"],
                    min_dense_ratio=sp["min_dense_ratio"],
                    min_dense_score=sp["min_dense_score"],
                    include_embeddings=False,
                )
            )

            # cache full details for on-demand modal fetch

            await publish_event(
                r,
                task_id,
                EventType.SLOT_RESULT,
                {
                    "slot_index": 0,
                    "category": "search results",
                    "query": query,
                    "products": _products_to_dicts(response.products),
                    "elapsed_ms": round(response.elapsed_ms, 1),
                },
            )
            await publish_event(
                r,
                task_id,
                EventType.COMPLETE,
                {
                    "intent": "single_item",
                    "total_ms": round(response.elapsed_ms, 1),
                },
            )
            return

        # outfit path: plan
        # outfit slots use fewer candidates (OT scoring is O(anchors x k × slots))
        outfit_top_k = min(sp["top_k"], int(os.getenv("OUTFIT_TOP_K", "8")))
        t_plan = time.perf_counter()
        slot_plan = await aplan(query, user_context=user_context, agent=state.planner_agent)
        plan_ms = (time.perf_counter() - t_plan) * 1000

        await publish_event(
            r,
            task_id,
            EventType.PLAN_COMPLETE,
            {
                "occasion": slot_plan.occasion,
                "constraints": slot_plan.constraints.model_dump(mode="json", exclude_none=True),
                "slots": [
                    {
                        "category": [c.value for c in sq.category],
                        "query": sq.query,
                        "formality": [f.value for f in sq.formality] if sq.formality else None,
                        "color": [c.value for c in sq.color] if sq.color else None,
                    }
                    for sq in slot_plan.slot_queries
                ],
                "elapsed_ms": round(plan_ms, 1),
            },
        )

        # search each slot (async gRPC)
        slot_results = []
        total_search_ms = 0.0

        for i, sq in enumerate(slot_plan.slot_queries):
            # encode slot query locally
            slot_emb = state.encoder.encode(sq.query)

            # build gRPC filters
            filters = catalog_pb2.Filters(
                category=[c.value for c in sq.category],
            )
            if slot_plan.constraints.gender:
                filters.gender = slot_plan.constraints.gender.value
            if slot_plan.constraints.season:
                filters.season = slot_plan.constraints.season.value
            if sq.formality:
                filters.formality.extend([f.value for f in sq.formality])
            if sq.color:
                filters.color.extend([c.value for c in sq.color])

            # async gRPC call to search service
            response = await state.search_stub.Search(
                catalog_pb2.SearchRequest(
                    embedding=slot_emb.squeeze(0).cpu().numpy().tolist(),
                    text=sq.query,
                    filters=filters,
                    top_k=outfit_top_k,
                    alpha=sp["alpha"],
                    beta=sp["beta"],
                    min_dense_ratio=sp["min_dense_ratio"],
                    min_dense_score=sp["min_dense_score"],
                    include_embeddings=True,
                )
            )

            total_search_ms += response.elapsed_ms

            # store Amazon URLs in Redis for the backend image proxy

            slot_results.append(
                {
                    "slot_query": sq,
                    "products": response.products,
                    "filters_used": dict(response.filters_used),
                    "elapsed_ms": response.elapsed_ms,
                }
            )

            cat_label = "+".join(c.value for c in sq.category)
            await publish_event(
                r,
                task_id,
                EventType.SLOT_RESULT,
                {
                    "slot_index": i,
                    "category": cat_label,
                    "query": sq.query,
                    "filters": dict(response.filters_used),
                    "products": _products_to_dicts(response.products),
                    "elapsed_ms": round(response.elapsed_ms, 1),
                },
            )

        # OT scoring (CPU — offload to thread)
        ot_ms = 0.0
        if state.scorer and len(slot_results) > 1:
            await publish_event(r, task_id, EventType.OT_SCORING, {})

            t_ot = time.perf_counter()

            # trim to top 6 per slot for scoring (but display keeps all results)
            # reduces computation by half
            scoring_results = [
                {
                    **slot,
                    "products": slot["products"][:6],
                }
                for slot in slot_results
            ]

            ot_result = await asyncio.to_thread(
                _score_outfits,
                state.scorer,
                scoring_results,
            )
            ot_ms = (time.perf_counter() - t_ot) * 1000

            await publish_event(r, task_id, EventType.OT_RESULT, ot_result)

        # done
        total_ms = plan_ms + total_search_ms + ot_ms
        await publish_event(
            r,
            task_id,
            EventType.COMPLETE,
            {
                "intent": "outfit",
                "plan_ms": round(plan_ms, 1),
                "search_ms": round(total_search_ms, 1),
                "ot_ms": round(ot_ms, 1),
                "total_ms": round(total_ms, 1),
            },
        )

    except Exception as e:
        logger.exception(f"task {task_id} failed")
        await publish_event(r, task_id, EventType.ERROR, {"message": str(e)})


def _score_outfits(scorer, slot_results):
    """Run OT greedy scoring on gRPC search results. Runs in thread pool."""
    import numpy as np
    import torch

    # convert gRPC products to the format OT scorer expects
    # scorer.greedy_select_from_embeddings takes per-slot embedding arrays
    slot_embeddings = []
    slot_products = []

    for sr in slot_results:
        products = sr["products"]
        img_embs = []
        txt_embs = []
        prod_info = []

        for p in products:
            img_embs.append(np.array(p.image_embedding, dtype=np.float32))
            txt_embs.append(np.array(p.text_embedding, dtype=np.float32))
            prod_info.append(
                {
                    "asin": p.asin,
                    "title": p.title,
                }
            )

        slot_embeddings.append(
            {
                "image": torch.tensor(np.stack(img_embs)),
                "text": torch.tensor(np.stack(txt_embs)),
            }
        )
        slot_products.append(prod_info)

    # run greedy CP scoring
    outfits = scorer.greedy_select_from_embeddings(slot_embeddings, slot_products)

    return {
        "outfits": [
            {
                "items": [
                    {
                        "asin": item["asin"],
                        "title": item["title"],
                        "cp_score": round(item["cp_score"], 3),
                    }
                    for item in outfit["items"]
                ],
                "outfit_cp": round(outfit["outfit_cp"], 3),
            }
            for outfit in outfits
        ],
        "elapsed_ms": 0,  # filled by caller
    }


@broker.task()
async def score_outfit_task(
    score_id: str,
    asins: list[str],
    context: Context = TaskiqDepends(),
) -> None:
    """Score a user-assembled outfit. Stores result in Redis for the backend to read."""

    import numpy as np
    import torch

    from search_service import catalog_pb2

    state = context.state
    r = state.redis
    key = f"fit-kit:score:{score_id}"

    try:
        if not state.scorer or len(asins) < 2:
            await r.set(key, json.dumps({"outfit_cp": 0.0}), ex=30)
            return

        # fetch embeddings from search service
        response = await state.search_stub.GetEmbeddings(catalog_pb2.EmbeddingRequest(asins=asins))

        if len(response.products) < 2:
            await r.set(key, json.dumps({"outfit_cp": 0.0, "error": "Products not found"}), ex=30)
            return

        # build embedding tensor
        embs = []
        for pe in response.products:
            img = torch.tensor(np.array(pe.image_embedding, dtype=np.float32))
            txt = torch.tensor(np.array(pe.text_embedding, dtype=np.float32))
            embs.append(torch.cat([img, txt], dim=-1))

        outfit_emb = torch.stack(embs).to(state.scorer.device)
        with torch.no_grad():
            cp = state.scorer.model.predict_compatibility([outfit_emb]).item()

        await r.set(key, json.dumps({"outfit_cp": round(cp, 3)}), ex=30)

    except Exception as e:
        logger.exception(f"score_outfit failed: {e}")
        await r.set(key, json.dumps({"outfit_cp": 0.0, "error": str(e)}), ex=30)
