import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    query: str | None = Field(
        default=None,
        min_length=2,
        max_length=500,
        description="Natural language query",
    )
    image: str | None = Field(
        default=None,
        description="Base64-encoded reference image for multimodal search",
    )
    gender: str | None = Field(
        default=None,
        description="men, women, boys, girls, unisex_adults, unisex_kids",
    )
    season: str | None = Field(default=None)
    top_k: int = Field(
        default=24,
        ge=1,
        le=48,
    )
    image_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Blend weight for image vs text (1.0 = image only)",
    )
    alpha: float | None = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Dense vs sparse weight",
    )
    beta: float | None = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Text vs image weight",
    )
    filters: dict | None = Field(
        default=None,
        description="Attribute filters (category, season, formality, color)",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    worker_ready: bool = False


class CatalogInfoResponse(BaseModel):
    product_count: int = 0


class EventType(str, Enum):
    """Events streamed to the frontend as outfit builds progressively."""

    ACCEPTED = "accepted"  # task created, task_id assigned
    ROUTING = "routing"  # query encoded, intent classified
    PLAN_COMPLETE = "plan_complete"  # planner finished, slots defined
    SLOT_RESULT = "slot_result"  # one slot's search results ready
    OT_SCORING = "ot_scoring"  # OutfitTransformer scoring started
    OT_RESULT = "ot_result"  # scored outfits ready
    COMPLETE = "complete"  # pipeline finished
    ERROR = "error"  # something went wrong


class SSEEvent(BaseModel):
    """A single server-sent event."""

    event: EventType
    data: dict[str, Any] = Field(default_factory=dict)

    def to_sse(self) -> str:
        """Format as SSE wire format."""

        return f"event: {self.event.value}\ndata: {json.dumps(self.data)}\n\n"


def task_channel(task_id: str) -> str:
    """Redis pub/sub channel name for a task."""
    return f"fit-kit:task:{task_id}"


def worker_status_key() -> str:
    """Redis key for worker health status."""
    return "fit-kit:worker:status"


def catalog_count_key() -> str:
    """Redis key for catalog product count."""
    return "fit-kit:catalog:count"
