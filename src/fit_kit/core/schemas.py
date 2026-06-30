from dataclasses import dataclass
from enum import Enum
from typing import Optional

from polars import DataFrame
from pydantic import BaseModel, Field


class Category(str, Enum):
    """Product categories aligned with the catalog taxonomy."""

    tops = "tops"
    dresses = "dresses"
    sweaters = "sweaters"
    pants = "pants"
    skirts = "skirts"
    shorts = "shorts"
    activewear = "activewear"
    swimwear = "swimwear"
    outerwear = "outerwear"
    sleepwear = "sleepwear"
    underwear = "underwear"
    socks = "socks"
    shoes = "shoes"
    bags = "bags"
    jewelry = "jewelry"
    watches = "watches"
    sunglasses = "sunglasses"
    belts = "belts"
    scarves = "scarves"
    hats = "hats"
    accessories = "accessories"


class Gender(str, Enum):
    """Target demographic labels from CLIP + regex classification."""

    women = "women"
    men = "men"
    girls = "girls"
    boys = "boys"
    unisex_adults = "unisex_adults"
    unisex_kids = "unisex_kids"


class Season(str, Enum):
    """Seasonality labels from CLIP classification."""

    summer = "summer"
    winter = "winter"
    spring_fall = "spring_fall"
    all_season = "all_season"


class Formality(str, Enum):
    """Formality levels from CLIP classification."""

    casual = "casual"
    business_casual = "business_casual"
    formal = "formal"
    athletic = "athletic"
    loungewear = "loungewear"


class Color(str, Enum):
    """Primary color labels from CLIP classification."""

    black = "black"
    white = "white"
    blue = "blue"
    red = "red"
    pink = "pink"
    green = "green"
    yellow = "yellow"
    orange = "orange"
    purple = "purple"
    brown = "brown"
    grey = "grey"
    navy = "navy"
    beige = "beige"
    gold = "gold"
    multicolor = "multicolor"


class SlotQuery(BaseModel):
    """One outfit slot with its retrieval query and optional filters."""

    category: list[Category] = Field(
        ...,
        min_length=1,
        max_length=3,
        description=(
            "Product category or categories for this slot. "
            "Use multiple when categories genuinely overlap, "
            "e.g. ['tops', 'sweaters'] for hoodies, ['shorts', 'pants'] for capris. "
            "Keep to 1 category when the item is unambiguous."
        ),
    )
    query: str = Field(
        ...,
        min_length=3,
        description=(
            "3-6 word retrieval query optimized for product search. "
            "Include material, style, color, pattern when relevant. "
            "ALWAYS preserve brand names, proper nouns, and specific entity names "
            "from the user's query (e.g. 'Nike', 'Taylor Swift', 'Disney'). "
            "Avoid synonym stacking (not 'soft warm cozy fuzzy', "
            "instead 'fleece pullover hoodie men'). "
            "Be specific: 'strappy heeled sandals gold' not 'nice shoes'."
        ),
    )
    formality: Optional[list[Formality]] = Field(
        default=None,
        description=(
            "Formality filter(s) for this slot. "
            "Use multiple values when a category spans formalities, "
            "e.g. ['casual', 'athletic'] for golf attire. "
            "Omit if formality is not relevant (e.g. jewelry, bags)."
        ),
    )
    color: Optional[list[Color]] = Field(
        default=None,
        description=(
            "Color filter(s) for this slot. Only specify when the user explicitly "
            "mentions a color or the occasion strongly implies one "
            "(e.g. 'little black dress' → ['black'], 'all white outfit' → ['white']). "
            "Leave null when any color works — semantic search already captures "
            "color intent from the query text."
        ),
    )


class Constraints(BaseModel):
    """Global constraints that apply to the entire outfit."""

    season: Optional[Season] = Field(
        default=None,
        description="Season for the outfit. Omit for year-round items.",
    )
    gender: Optional[Gender] = Field(
        default=None,
        description="Target demographic for the outfit.",
    )
    max_price: Optional[float] = Field(default=None, description="Maximum price per item in USD")
    exclude: Optional[list[str]] = Field(
        default=None,
        description="Explicit exclusions, e.g. ['strapless', 'mini dress', 'heels']",
    )


class SlotPlan(BaseModel):
    """Complete outfit plan produced by the planner."""

    occasion: Optional[str] = Field(
        default=None,
        description="The occasion or context, e.g. 'beach wedding', 'job interview', 'hiking'",
    )
    slot_queries: list[SlotQuery] = Field(
        ...,
        min_length=1,
        max_length=8,
        description="One entry per outfit category needed, each with a retrieval-optimized query",
    )
    constraints: Constraints = Field(
        default_factory=Constraints,
        description="Global constraints that apply to all slots",
    )


@dataclass
class SlotSearchResult:
    """Search results for a single outfit slot."""

    slot_query: SlotQuery
    products: list[DataFrame]
    filters_used: dict
    elapsed_ms: float = 0.0


@dataclass
class OutfitItem:
    """A single item selected by OutfitTransformer scoring."""

    asin: str
    title: str
    cp_score: float  # compatibility score when this item was added (1.0 for anchor)


@dataclass
class ScoredOutfit:
    """A complete outfit scored by OutfitTransformer."""

    items: list[OutfitItem]
    outfit_cp: float  # full-outfit compatibility score


@dataclass
class OutfitScoringResult:
    """Result of OutfitTransformer greedy scoring."""

    outfits: list[ScoredOutfit]
    slot_categories: list[str]  # category label per slot, e.g. ["tops", "pants+shorts"]
    anchor_slot: int = 0  # which slot was used as anchor (shoes-first heuristic)


@dataclass
class OutfitSearchSet:
    """Complete outfit recommendation."""

    query: str
    user_context: dict | None
    plan: SlotPlan
    slot_results: list[SlotSearchResult]
    plan_elapsed_s: float = 0.0
    search_elapsed_ms: float = 0.0
    ot_result: OutfitScoringResult | None = None
