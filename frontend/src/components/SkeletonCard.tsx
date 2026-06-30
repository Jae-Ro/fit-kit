export function SkeletonCard() {
  return (
    <div className="product-card skeleton-card">
      <div className="card-image skeleton-pulse" />
      <div className="skeleton-pulse" style={{ width: "70%", height: 8, margin: "0 auto 3px" }} />
      <div className="skeleton-pulse" style={{ width: "50%", height: 7, margin: "0 auto" }} />
    </div>
  );
}
