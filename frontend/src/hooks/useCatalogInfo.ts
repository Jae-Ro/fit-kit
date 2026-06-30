import { useEffect, useState } from "react";
import type { CatalogInfo } from "../types/api";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export function useCatalogInfo() {
  const [info, setInfo] = useState<CatalogInfo | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/catalog/info`)
      .then((r) => r.json())
      .then(setInfo)
      .catch(() => setInfo(null));
  }, []);

  const label = info
    ? `${Math.round(info.product_count / 1000)}k products`
    : "";

  return { info, label };
}
