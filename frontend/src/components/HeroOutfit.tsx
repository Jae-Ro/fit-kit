import { useState } from "react";
import { createPortal } from "react-dom";
import { Star, Check, X, Sparkles } from "lucide-react";
import type { SlotResult, OTResult, Product } from "../types/api";

type HeroState = "skeleton" | "shimmer" | "filled" | "generated";

interface Props {
  state: HeroState;
  slots: SlotResult[];
  otResult: OTResult | null;
  selectedItems: Record<number, number>;
  isCustom: boolean;
  userCp: number | null;
  scoringPending: boolean;
  generatedImage: string | null;
  generating: boolean;
  onVisualize?: () => void;
}

export function HeroOutfit({
  state,
  slots,
  otResult,
  selectedItems,
  isCustom,
  userCp,
  scoringPending,
  generatedImage,
  generating,
  onVisualize,
}: Props) {
  const borderColor = isCustom ? "#85B7EB" : state === "generated" ? "#AFA9EC" : "#85B7EB";
  const isFilled = state === "filled" || state === "generated";
  const [showFullImage, setShowFullImage] = useState(false);

  // get selected products for hero display
  const heroProducts: { category: string; product: Product | null }[] = slots.map(
    (slot, i) => ({
      category: slot.category,
      product: isFilled ? slot.products[selectedItems[i] ?? 0] ?? null : null,
    })
  );

  // if no slots yet, show placeholder categories
  const categories = slots.length > 0
    ? slots.map((s) => s.category)
    : ["TOPS", "PANTS", "SHOES", "BELTS", "SUNGLASSES"];

  // determine which CP to show
  const cpValue = isCustom ? userCp : otResult?.outfits?.[0]?.outfit_cp ?? null;

  return (
    <div
      className={`hero-outfit ${isFilled ? "fade-in" : ""}`}
      style={{ borderColor }}
    >
      <div className="hero-header">
        <Star size={13} color="var(--tt)" />
        <span
          className="hero-label"
          style={{ color: isCustom ? "var(--bt)" : "var(--tt)" }}
        >
          {isCustom ? "Your outfit" : "Recommended Fit"}
        </span>

        {isFilled && cpValue !== null && (
          <span className={`hero-cp-badge ${isCustom ? "hero-cp-custom" : ""}`}>
            {scoringPending ? "scoring…" : `CP ${cpValue.toFixed(3)}`}
          </span>
        )}

        {isFilled && (
          <div className="hero-visualize-btn" onClick={onVisualize}>
            {state === "generated" ? (
              <>
                <Check size={11} color="var(--pt)" />
                <span>Generated</span>
              </>
            ) : (
              <>
                <Sparkles size={11} color="var(--pt)" />
                <span>Visualize with AI</span>
              </>
            )}
          </div>
        )}
      </div>

      <div className="hero-items">
        {isFilled
          ? heroProducts.map(({ category, product }, i) => (
              <div className="hero-item" key={category}>
                <div
                  className={`hero-item-img ${
                    selectedItems[i] !== undefined &&
                    selectedItems[i] !== (otResult?.outfits?.[0]?.items?.[i] ? i : 0)
                      ? "hero-item-custom"
                      : ""
                  }`}
                >
                  {product?.image_url ? (
                    <img
                      src={product.image_url}
                      alt={product.title}
                      onError={(e) => {
                        const img = e.target as HTMLImageElement;
                        if (product.fallback_image_url && img.src !== product.fallback_image_url) {
                          img.src = product.fallback_image_url;
                        } else {
                          img.style.display = "none";
                        }
                      }}
                    />
                  ) : (
                    <span>{category.charAt(0)}</span>
                  )}
                </div>
                <div className="hero-item-category">{category.toUpperCase()}</div>
                <div className="hero-item-title">
                  {product?.title ?? ""}
                </div>
              </div>
            ))
          : categories.map((cat) => (
              <div className="hero-item" key={cat}>
                <div
                  className={`hero-item-img skeleton-pulse ${
                    state === "shimmer" ? "shimmer-fast" : ""
                  }`}
                />
                <div className="hero-item-category">{cat.toUpperCase()}</div>
              </div>
            ))}

        {(state === "generated" || generating) && (
          <div className="hero-generated-preview">
            {generating ? (
              <>
                <div className="hero-generating-spinner" />
                <div className="sil-label">Generating...</div>
              </>
            ) : generatedImage ? (
              <img
                src={generatedImage}
                alt="AI generated outfit preview"
                className="hero-generated-img"
                onClick={() => setShowFullImage(true)}
                style={{ cursor: "pointer" }}
              />
            ) : (
              <>
                <div className="hero-silhouette">
                  <div className="sil-head" />
                  <div className="sil-torso" />
                  <div className="sil-legs" />
                  <div className="sil-feet">
                    <div className="sil-shoe" />
                    <div className="sil-shoe" />
                  </div>
                </div>
                <div className="sil-label">AI GENERATED</div>
              </>
            )}
          </div>
        )}
      </div>

      {showFullImage && generatedImage && createPortal(
        <div className="generated-modal-overlay" onClick={() => setShowFullImage(false)}>
          <div className="generated-modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setShowFullImage(false)}>
              <X size={18} />
            </button>

            <div className="generated-modal-layout">
              {/* Left: product images */}
              <div className="generated-modal-items">
                {heroProducts.map(({ product }) => (
                  product && (
                    <div className="generated-item-img" key={product.asin}>
                      <img
                        src={product.image_url}
                        alt={product.title}
                        onError={(e) => {
                          const img = e.target as HTMLImageElement;
                          if (product.fallback_image_url && img.src !== product.fallback_image_url) {
                            img.src = product.fallback_image_url;
                          }
                        }}
                      />
                    </div>
                  )
                ))}
              </div>

              {/* Right: full-size generated image */}
              <div className="generated-modal-preview">
                <img src={generatedImage} alt="AI generated outfit" />
                <div className="generated-modal-title">AI Generated Outfit</div>

              </div>
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}