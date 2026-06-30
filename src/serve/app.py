import asyncio
import json
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import grpc.aio
import redis.asyncio as aioredis
from litestar import Litestar, MediaType, get, post
from litestar.config.cors import CORSConfig
from litestar.response import Response, Stream

from fit_kit.core.prompts import OUTFIT_IMAGE_GEN_PROMPT
from fit_kit.utils.log_utils import get_custom_logger
from search_service import catalog_pb2, catalog_pb2_grpc
from serve.schemas import (
    CatalogInfoResponse,
    EventType,
    HealthResponse,
    RecommendRequest,
    SSEEvent,
    catalog_count_key,
    task_channel,
    worker_status_key,
)
from serve.worker import recommend_task, score_outfit_task

logger = get_custom_logger("backend")


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", "data/catalog/images"))
SEARCH_SERVICE_URL = os.getenv("SEARCH_SERVICE_URL", "localhost:50051")

MODELS_DIR = Path(os.getenv("MODELS_DIR", "data/catalog/models"))

_GENDER_TO_MODEL = {
    "men": "Men.jpg",
    "women": "Women.jpg",
    "boys": "Boys.jpg",
    "girls": "Girls.jpg",
    "unisex_adults": "Men.jpg",
    "unisex_kids": "Boys.jpg",
}

ALLOWED_CDN_HOSTS = {
    "m.media-amazon.com",
    "images-na.ssl-images-amazon.com",
}

IMAGE_GEN_MODEL = os.getenv("IMAGE_GEN_MODEL", "gpt-4o")

_redis: aioredis.Redis | None = None
_search_stub: catalog_pb2_grpc.CatalogSearchStub | None = None
_grpc_channel: grpc.aio.Channel | None = None


def _is_safe_url(url: str) -> bool:

    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in ALLOWED_CDN_HOSTS


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


async def stream_task_events(task_id: str, timeout: float = 120.0) -> AsyncGenerator[str, None]:
    """Subscribe to Redis pub/sub and yield SSE events for a task."""
    r = await get_redis()
    pubsub = r.pubsub()
    channel = task_channel(task_id)

    await pubsub.subscribe(channel)

    yield SSEEvent(
        event=EventType.ACCEPTED,
        data={"task_id": task_id},
    ).to_sse()

    try:
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg is None:
                yield ": keepalive\n\n"
                continue

            if msg["type"] != "message":
                continue

            raw = msg["data"]
            yield raw if raw.startswith("event:") else f"data: {raw}\n\n"

            # check for terminal events
            event_line = raw.split("\n")[0] if raw.startswith("event:") else ""
            event_type = event_line.replace("event: ", "")
            if event_type in (EventType.COMPLETE.value, EventType.ERROR.value):
                break
        else:
            yield SSEEvent(
                event=EventType.ERROR,
                data={"message": "Timeout waiting for results"},
            ).to_sse()

    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.close()


@post("/api/recommend", media_type=MediaType.TEXT)
async def recommend(data: RecommendRequest) -> Stream:
    """Enqueue a recommendation task and stream results via SSE."""
    task_id = str(uuid.uuid4())

    user_context = {}
    if data.gender:
        user_context["gender"] = data.gender
    if data.season:
        user_context["season"] = data.season

    # build search overrides from advanced options
    search_overrides = {}
    if data.alpha is not None:
        search_overrides["alpha"] = data.alpha
    if data.beta is not None:
        search_overrides["beta"] = data.beta
    if data.filters:
        search_overrides["filters"] = data.filters

    # enqueue via Taskiq — the worker picks this up
    await recommend_task.kiq(
        task_id=task_id,
        query=data.query,
        user_context=user_context or None,
        top_k=data.top_k,
        search_overrides=search_overrides or None,
    )

    return Stream(
        stream_task_events(task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@get("/api/health")
async def health() -> HealthResponse:
    """Check backend and worker health."""
    try:
        r = await get_redis()
        status = await r.get(worker_status_key())
        return HealthResponse(status="ok", worker_ready=status == "ready")
    except Exception:
        return HealthResponse(status="degraded", worker_ready=False)


@get("/api/catalog/info")
async def catalog_info() -> CatalogInfoResponse:
    """Return catalog metadata (product count)."""
    try:
        r = await get_redis()
        count = await r.get(catalog_count_key())
        return CatalogInfoResponse(product_count=int(count) if count else 0)
    except Exception:
        return CatalogInfoResponse(product_count=0)


@get("/api/images/{filename:str}")
async def serve_image(filename: str, fallback: str | None = None) -> Response:
    """Serve product images. On miss: redirect to CDN, download in background."""
    path = IMAGE_DIR / filename

    # cache hit → serve from disk
    if path.exists():
        return Response(
            content=path.read_bytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    # cache miss with fallback CDN URL → redirect + background download
    if fallback:
        if not _is_safe_url(fallback):
            return Response(content="Invalid fallback URL", status_code=400)

        asyncio.create_task(_fetch_and_save(path, fallback))
        return Response(
            content="",
            status_code=307,
            headers={"Location": fallback},
        )

    return Response(content="Not found", status_code=404)


async def _fetch_and_save(path: Path, url: str) -> None:
    """Download an image from CDN and save to disk. Fire-and-forget."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10.0, follow_redirects=True)
            if resp.status_code == 200 and len(resp.content) > 100:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(resp.content)
    except Exception as e:
        logger.error(f"Error attempting to fetch image: {url}. {e}")


@asynccontextmanager
async def lifespan(app: Litestar):
    """Startup/shutdown lifecycle."""
    global _redis, _grpc_channel, _search_stub
    _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    _grpc_channel = grpc.aio.insecure_channel(
        SEARCH_SERVICE_URL,
        options=[("grpc.max_receive_message_length", 50 * 1024 * 1024)],
    )
    _search_stub = catalog_pb2_grpc.CatalogSearchStub(_grpc_channel)
    yield
    if _grpc_channel:
        await _grpc_channel.close()
    if _redis:
        await _redis.close()


@post("/api/score-outfit", media_type=MediaType.JSON)
async def score_outfit(data: dict) -> dict:
    """Score a user-assembled outfit. Enqueues task, waits for result via Redis."""
    asins: list[str] = data.get("asins", [])
    if len(asins) < 2:
        return {"outfit_cp": 0.0, "error": "Need at least 2 items"}

    score_id = str(uuid.uuid4())
    await score_outfit_task.kiq(score_id, asins)

    # wait for result via Redis polling
    r = await get_redis()
    key = f"fit-kit:score:{score_id}"
    for _ in range(50):  # 5s timeout
        result = await r.get(key)
        if result:
            await r.delete(key)
            return json.loads(result)
        await asyncio.sleep(0.1)

    return {"outfit_cp": 0.0, "error": "Scoring timeout"}


@get("/api/products/{asin:str}")
async def get_product_detail(asin: str) -> dict:
    """Fetch full product details for modal display."""
    r = await get_redis()

    # check Redis cache first
    cache_key = f"fit-kit:product:{asin}"
    cached = await r.get(cache_key)
    if cached:
        return json.loads(cached)

    # cache miss — fetch from search service via gRPC
    try:
        resp = await _search_stub.GetProductDetail(catalog_pb2.ProductDetailRequest(asin=asin))

        # parse reviews
        reviews = []
        if resp.reviews_json:
            try:
                parsed = json.loads(resp.reviews_json)
                if isinstance(parsed, list):
                    for item in parsed[:10]:
                        if isinstance(item, dict):
                            reviews.append(item)
                        elif isinstance(item, str):
                            reviews.append({"text": item})
            except (json.JSONDecodeError, TypeError):
                if resp.reviews_json:
                    reviews = [{"text": resp.reviews_json[:500]}]

        # parse details
        details = {}
        if resp.details_json:
            try:
                details = json.loads(resp.details_json)
            except (json.JSONDecodeError, TypeError):
                pass

        result = {
            "asin": resp.asin,
            "title": resp.title,
            "features": list(resp.features),
            "details": details,
            "reviews": reviews,
        }

        # cache for 1 hour
        await r.set(cache_key, json.dumps(result), ex=3600)
        return result

    except Exception:
        return {"asin": asin, "features": [], "details": {}, "reviews": []}


@post("/api/visualize", media_type=MediaType.JSON)
async def visualize_outfit(data: dict) -> dict:
    """Generate an AI image of the outfit on a model using OpenAI's image API."""
    import base64

    from openai import AsyncOpenAI

    asins: list[str] = data.get("asins", [])
    query: str = data.get("query", "")
    occasion: str = data.get("occasion", "")
    gender: str = data.get("gender", "women").lower()

    if not asins:
        return {"error": "No items provided"}

    # fetch product titles
    r = await get_redis()
    items = []
    product_images_b64 = []

    for asin in asins:
        cached = await r.get(f"fit-kit:product:{asin}")
        if cached:
            items.append(json.loads(cached).get("title", asin))
        else:
            try:
                resp = await _search_stub.GetProductDetail(
                    catalog_pb2.ProductDetailRequest(asin=asin)
                )
                items.append(resp.title or asin)
            except Exception:
                items.append(asin)

        # load product image
        img_path = IMAGE_DIR / f"{asin}.jpg"
        if img_path.exists():
            product_images_b64.append(base64.b64encode(img_path.read_bytes()).decode())

    # load model image based on gender
    model_filename = _GENDER_TO_MODEL.get(gender, "female.jpg")
    model_path = MODELS_DIR / model_filename
    model_image_b64 = None
    if model_path.exists():
        model_image_b64 = base64.b64encode(model_path.read_bytes()).decode()

    # build prompt
    item_list = "\n".join(f"- {item}" for item in items)
    gender_label = "male" if gender in ("men", "boys") else "female"

    prompt = OUTFIT_IMAGE_GEN_PROMPT.format(
        gender_label=gender_label,
        item_list=item_list,
        occasion=f"Occasion: {occasion}" if occasion else "",
        style_context=f"Style context: {query}" if query else "",
    )

    # call OpenAI Responses API (supports image inputs + image generation)
    client = AsyncOpenAI()

    try:
        input_content = []

        # model image first (the base to dress)
        if model_image_b64:
            input_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{model_image_b64}",
                }
            )

        # product images as reference
        for img_b64 in product_images_b64:
            input_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{img_b64}",
                }
            )

        # text prompt
        input_content.append(
            {
                "type": "input_text",
                "text": prompt,
            }
        )

        response = await client.responses.create(
            model=IMAGE_GEN_MODEL,
            input=[
                {
                    "role": "user",
                    "content": input_content,
                }
            ],
            tools=[
                {
                    "type": "image_generation",
                    "quality": "low",
                    "size": "1024x1536",
                }
            ],
        )

        # extract the generated image from the response
        for output in response.output:
            if output.type == "image_generation_call":
                return {"image": output.result}

        return {"error": "No image generated in response"}

    except Exception as e:
        logger.exception(e)
        return {"error": str(e)}


# ------------------------------------------------------------------------------
# Litestar App
# API with SSE streaming.

# Receives recommendation requests, enqueues them via Taskiq,
# and streams progress events back to the frontend via Server-Sent Events.
# ------------------------------------------------------------------------------

app = Litestar(
    route_handlers=[
        recommend,
        health,
        catalog_info,
        serve_image,
        score_outfit,
        get_product_detail,
        visualize_outfit,
    ],
    cors_config=CORSConfig(
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    ),
    lifespan=[lifespan],
    debug=True,
)
