import { useCallback, useRef } from "react";
import type { EventType } from "../types/api";

interface SSECallbacks {
  onEvent: (eventType: EventType, data: Record<string, unknown>) => void;
  onError: (error: string) => void;
  onDone: () => void;
}

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export function useRecommend() {
  const abortRef = useRef<AbortController | null>(null);

  const recommend = useCallback(
    async (
      params: {
        query?: string;
        image?: string;
        gender?: string;
        season?: string;
        top_k?: number;
        image_weight?: number;
      },
      callbacks: SSECallbacks
    ) => {
      // abort any existing request
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await fetch(`${API_BASE}/api/recommend`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          callbacks.onError(`HTTP ${res.status}`);
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let currentEvent = "";
        let currentData = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              currentData = line.slice(6).trim();
            } else if (line === "" && currentEvent && currentData) {
              try {
                const data = JSON.parse(currentData);
                callbacks.onEvent(currentEvent as EventType, data);

                if (currentEvent === "complete") {
                  callbacks.onDone();
                  return;
                }
                if (currentEvent === "error") {
                  callbacks.onError(data.message ?? "Unknown error");
                  return;
                }
              } catch {
                // skip malformed JSON
              }
              currentEvent = "";
              currentData = "";
            } else if (line.startsWith(":")) {
              // keepalive comment, ignore
            }
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          callbacks.onError((err as Error).message);
        }
      }
    },
    []
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { recommend, cancel };
}