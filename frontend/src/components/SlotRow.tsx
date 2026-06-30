import { Info } from "lucide-react";
import type { SlotResult, Product } from "../types/api";
import { ProductCard } from "./ProductCard";
import { SkeletonCard } from "./SkeletonCard";

interface Props {
  slot?: SlotResult;
  category: string;
  filled: boolean;
  showSelections: boolean;
  selectedIndex?: number;
  onSelectItem?: (slotIndex: number, itemIndex: number) => void;
  onClickProduct?: (product: Product) => void;
}

export function SlotRow({
  slot,
  category,
  filled,
  showSelections,
  selectedIndex,
  onSelectItem,
  onClickProduct,
}: Props) {
  return (
    <div className="slot-row">
      <div className="slot-header">
        <span className="slot-category">{category.toUpperCase()}</span>
        {slot && <span className="slot-count">{slot.products.length}</span>}
        {slot && (
          <div className="info-btn">
            <Info size={10} color="var(--t4)" />
            <div className="info-tip">
              <div style={{ fontWeight: 500, marginBottom: 3, color: "var(--tx)" }}>
                Search query
              </div>
              "{slot.query}"
              <div
                style={{
                  marginTop: 4,
                  paddingTop: 4,
                  borderTop: "1px solid var(--bd)",
                  fontSize: 8,
                  color: "var(--t3)",
                }}
              >
                {Object.entries(slot.filters || {})
                  .map(([k, v]) => `${k}: ${v}`)
                  .join(" · ")}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="card-row">
        {filled && slot
          ? slot.products.map((product, ii) => (
              <ProductCard
                key={product.asin}
                product={product}
                selected={showSelections && ii === selectedIndex}
                onSelect={
                  showSelections && onSelectItem
                    ? () => onSelectItem(slot.slot_index, ii)
                    : undefined
                }
                onClick={
                  onClickProduct
                    ? () => onClickProduct(product)
                    : undefined
                }
              />
            ))
          : Array.from({ length: 5 }, (_, i) => <SkeletonCard key={i} />)}
      </div>
    </div>
  );
}