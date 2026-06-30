import { Check } from "lucide-react";
import type { Product } from "../types/api";

interface Props {
  product: Product;
  selected?: boolean;
  onSelect?: () => void;
  onClick?: () => void;
}

export function ProductCard({ product, selected, onSelect, onClick }: Props) {
  return (
    <div
      className={`product-card ${selected ? "selected" : ""}`}
      onClick={onClick}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      {onSelect && (
        <div
          className={`card-checkbox ${selected ? "checked" : ""}`}
          onClick={(e) => {
            e.stopPropagation();
            onSelect();
          }}
        >
          {selected && <Check size={10} color="var(--bg)" />}
        </div>
      )}
      <div className="card-image">
        {product.image_url ? (
          <img
            src={product.image_url}
            alt={product.title}
            loading="lazy"
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
          <span className="card-placeholder">
            {product.clip_category?.charAt(0).toUpperCase() ?? "?"}
          </span>
        )}
      </div>
      <div className="card-title">{product.title}</div>
      <div className="card-meta">
        ★ {product.average_rating} · {product.clip_color}
      </div>
    </div>
  );
}