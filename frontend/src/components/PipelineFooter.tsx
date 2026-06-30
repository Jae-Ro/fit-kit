import type { Timings } from "../types/api";

interface Props {
  timings: Timings;
  statusText?: string;
}

export function PipelineFooter({ timings, statusText }: Props) {
  const parts: string[] = [];
  if (timings.plan_ms) parts.push(`plan: ${(timings.plan_ms / 1000).toFixed(1)}s`);
  if (timings.search_ms) parts.push(`search: ${Math.round(timings.search_ms)}ms`);
  if (timings.ot_ms) parts.push(`score: ${Math.round(timings.ot_ms)}ms`);
  if (timings.total_ms && !statusText) parts.push(`total: ${(timings.total_ms / 1000).toFixed(1)}s`);

  return (
    <div className="results-footer">
      <span>{statusText ?? parts.join(" · ")}</span>
      <span>Amazon Fashion</span>
    </div>
  );
}
