// SSE event types — mirrors serve.schemas.EventType

export type EventType =
  | "accepted"
  | "routing"
  | "plan_complete"
  | "slot_result"
  | "ot_scoring"
  | "ot_result"
  | "complete"
  | "error";

export interface Product {
  asin: string;
  title: string;
  score: number;
  average_rating: number;
  clip_color: string;
  clip_category: string;
  clip_formality: string;
  clip_season: string;
  clip_gender: string;
  image_url: string;
  fallback_image_url: string;
  dense_score: number;
  sparse_score: number;
  text_score: number;
  image_score: number;
}

export interface SlotResult {
  slot_index: number;
  category: string;
  query: string;
  filters: Record<string, string>;
  products: Product[];
  elapsed_ms: number;
}

export interface PlanSlot {
  category: string[];
  query: string;
  formality: string[] | null;
}

export interface OutfitItem {
  asin: string;
  title: string;
  cp_score: number;
}

export interface Outfit {
  items: OutfitItem[];
  outfit_cp: number;
}

export interface OTResult {
  outfits: Outfit[];
  slot_categories: string[];
  anchor_slot: string;
  elapsed_ms: number;
}

export interface Timings {
  plan_ms?: number;
  search_ms?: number;
  ot_ms?: number;
  total_ms?: number;
}

export interface CatalogInfo {
  product_count: number;
}

// App-level state

export type Phase = "landing" | "streaming" | "complete";

export interface AppState {
  phase: Phase;
  intent: "outfit" | "single_item" | null;
  occasion: string | null;
  constraints: Record<string, unknown> | null;
  planSlots: PlanSlot[];
  slots: SlotResult[];
  otResult: OTResult | null;
  selectedItems: Record<number, number>; // slotIndex → itemIndex
  timings: Timings;
  statusMessage: string;
  isScoring: boolean;
}