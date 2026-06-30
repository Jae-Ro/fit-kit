import { useEffect, useState } from "react";
import { X } from "lucide-react";
import type { Product } from "../types/api";

interface ProductDetail {
  features: string[];
  details: Record<string, string>;
  reviews: { title?: string; text: string; rating?: number; helpful_votes?: number }[];
}

interface Props {
  product: Product;
  onClose: () => void;
}

export function ProductModal({ product, onClose }: Props) {
  const [detail, setDetail] = useState<ProductDetail | null>(null);

  useEffect(() => {
    fetch(`/api/products/${product.asin}`)
      .then((r) => r.json())
      .then(setDetail)
      .catch(() => setDetail({ features: [], details: {}, reviews: [] }));
  }, [product.asin]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          <X size={18} />
        </button>

        <div className="modal-layout">
          {/* Image */}
          <div className="modal-image">
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

          {/* Details */}
          <div className="modal-details">
            <h2 className="modal-title">{product.title}</h2>

            <div className="modal-tags">
              <span className="modal-tag">{product.clip_category}</span>
              <span className="modal-tag">{product.clip_color}</span>
              {product.clip_formality && <span className="modal-tag">{product.clip_formality}</span>}
              {product.clip_season && <span className="modal-tag">{product.clip_season}</span>}
              {product.clip_gender && <span className="modal-tag">{product.clip_gender}</span>}
              <span className="modal-tag">★ {product.average_rating}</span>
            </div>

            {/* Retrieval scores */}
            <div className="modal-section">
              <h3 className="modal-section-title">Retrieval Scores</h3>
              <div className="modal-scores">
                <div className="score-item">
                  <span className="score-label">Hybrid (RRF)</span>
                  <span className="score-value">{product.score.toFixed(4)}</span>
                </div>
                <div className="score-item">
                  <span className="score-label">Dense</span>
                  <span className="score-value">{product.dense_score.toFixed(4)}</span>
                </div>
                <div className="score-item">
                  <span className="score-label">Sparse (BM25)</span>
                  <span className="score-value">{product.sparse_score.toFixed(4)}</span>
                </div>
                <div className="score-item">
                  <span className="score-label">Text similarity</span>
                  <span className="score-value">{product.text_score.toFixed(4)}</span>
                </div>
                <div className="score-item">
                  <span className="score-label">Image similarity</span>
                  <span className="score-value">{product.image_score.toFixed(4)}</span>
                </div>
              </div>
            </div>

            {!detail && (
              <div className="modal-section">
                <span style={{ color: "var(--t3)", fontSize: 12 }}>Loading details…</span>
              </div>
            )}

            {/* Features */}
            {detail?.features && detail.features.length > 0 && (
              <div className="modal-section">
                <h3 className="modal-section-title">Features</h3>
                <ul className="modal-features">
                  {detail.features.map((f, i) => (
                    <li key={i}>{f}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Details */}
            {detail?.details && Object.keys(detail.details).length > 0 && (
              <div className="modal-section">
                <h3 className="modal-section-title">Details</h3>
                <div className="modal-detail-grid">
                  {Object.entries(detail.details).map(([k, v]) => (
                    <div key={k} className="detail-row">
                      <span className="detail-key">{k}</span>
                      <span className="detail-val">{String(v)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Reviews */}
            {detail?.reviews && detail.reviews.length > 0 && (
              <div className="modal-section">
                <h3 className="modal-section-title">Reviews ({detail.reviews.length})</h3>
                <div className="modal-reviews-list">
                  {detail.reviews.map((review, i) => (
                    <div key={i} className="modal-review-item">
                      <div className="review-header">
                        {review.rating && (
                          <span className="review-rating">
                            {"★".repeat(review.rating)}{"☆".repeat(5 - review.rating)}
                          </span>
                        )}
                        {review.title && (
                          <span className="review-title">{review.title}</span>
                        )}
                        {review.helpful_votes !== undefined && review.helpful_votes > 0 && (
                          <span className="review-helpful">
                            {review.helpful_votes} helpful
                          </span>
                        )}
                      </div>
                      {review.text && <p className="review-body">{review.text}</p>}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}