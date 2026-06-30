import { useCallback, useEffect, useState } from "react";
import { Shirt, Check, Sun, Moon } from "lucide-react";
import { useRecommend } from "./hooks/useRecommend";
import { useCatalogInfo } from "./hooks/useCatalogInfo";
import { Landing } from "./components/Landing";
import { SearchBar } from "./components/SearchBar";
import type { SearchOptions } from "./components/SearchBar";
import { HeroOutfit } from "./components/HeroOutfit";
import { SlotRow } from "./components/SlotRow";
import { ProductModal } from "./components/ProductModal";
import { PipelineFooter } from "./components/PipelineFooter";
import type { AppState, EventType, SlotResult, OTResult, PlanSlot, Product } from "./types/api";
import "./App.css";

function useTheme() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  }, [dark]);

  const toggle = useCallback(() => setDark((d) => !d), []);
  return { dark, toggle };
}

const INITIAL_STATE: AppState = {
  phase: "landing",
  intent: null,
  occasion: null,
  constraints: null,
  planSlots: [],
  slots: [],
  otResult: null,
  selectedItems: {},
  timings: {},
  statusMessage: "",
  isScoring: false,
};

function App() {
  const [state, setState] = useState<AppState>(INITIAL_STATE);
  const [currentQuery, setCurrentQuery] = useState("");
  const [currentGender, setCurrentGender] = useState("Men");
  const [currentOptions, setCurrentOptions] = useState<SearchOptions | undefined>();
  const [visualized, setVisualized] = useState(false);
  const { recommend, cancel } = useRecommend();
  const { label: catalogLabel } = useCatalogInfo();
  const { dark, toggle: toggleTheme } = useTheme();
  const [modalProduct, setModalProduct] = useState<Product | null>(null);
  const [userCp, setUserCp] = useState<number | null>(null);
  const [scoringPending, setScoringPending] = useState(false);

  const handleHome = useCallback(() => {
    cancel();
    setState(INITIAL_STATE);
    setVisualized(false);
  }, [cancel]);

  const handleSubmit = useCallback(
    (query: string, image: string | null, gender: string, options?: SearchOptions) => {
      setCurrentQuery(query);
      setCurrentGender(gender);
      setCurrentOptions(options);
      setVisualized(false);
      setUserCp(null);
      setState({
        ...INITIAL_STATE,
        phase: "streaming",
        statusMessage: "Analyzing your request...",
      });

      recommend(
        {
          query: query || undefined,
          image: image || undefined,
          gender: gender.toLowerCase(),
          ...(options && {
            top_k: options.top_k,
            alpha: options.alpha,
            beta: options.beta,
            filters: options.filters,
          }),
        },
        {
          onEvent: (eventType: EventType, data: Record<string, unknown>) => {
            setState((prev) => {
              switch (eventType) {
                case "routing":
                  return {
                    ...prev,
                    intent: data.intent as "outfit" | "single_item",
                    statusMessage:
                      data.intent === "outfit"
                        ? "Planning outfit..."
                        : "Searching...",
                  };

                case "plan_complete":
                  return {
                    ...prev,
                    occasion: data.occasion as string,
                    constraints: data.constraints as Record<string, unknown>,
                    planSlots: data.slots as PlanSlot[],
                    timings: { ...prev.timings, plan_ms: data.elapsed_ms as number },
                    statusMessage: `${(data.slots as PlanSlot[]).length} slots planned`,
                  };

                case "slot_result": {
                  const slot = data as unknown as SlotResult;
                  if (!slot.filters) slot.filters = {};
                  const newSlots = [...prev.slots];
                  newSlots[slot.slot_index] = slot;
                  const filled = newSlots.filter(Boolean).length;
                  const total = prev.planSlots.length || 1;
                  return {
                    ...prev,
                    slots: newSlots,
                    timings: {
                      ...prev.timings,
                      search_ms: (prev.timings.search_ms ?? 0) + (slot.elapsed_ms ?? 0),
                    },
                    statusMessage:
                      filled < total
                        ? `Searching products... ${filled}/${total} slots`
                        : "All slots filled",
                  };
                }

                case "ot_scoring":
                  return {
                    ...prev,
                    isScoring: true,
                    statusMessage: "Scoring outfit combinations...",
                  };

                case "ot_result": {
                  const result = data as unknown as OTResult;
                  // set default selections from best outfit
                  const selections: Record<number, number> = {};
                  if (result.outfits?.[0]) {
                    const bestOutfit = result.outfits[0];
                    bestOutfit.items.forEach((item, slotIdx) => {
                      // find which product index matches the asin
                      const slot = prev.slots[slotIdx];
                      if (slot) {
                        const idx = slot.products.findIndex(
                          (p) => p.asin === item.asin
                        );
                        if (idx >= 0) selections[slotIdx] = idx;
                      }
                    });
                  }
                  return {
                    ...prev,
                    otResult: result,
                    selectedItems: selections,
                    isScoring: false,
                    timings: {
                      ...prev.timings,
                      ot_ms: result.elapsed_ms,
                    },
                  };
                }

                case "complete": {
                  // if no OT result (single slot), default to first items
                  const selections = { ...prev.selectedItems };
                  if (!prev.otResult) {
                    prev.slots.forEach((slot, i) => {
                      if (slot && !(i in selections)) {
                        selections[i] = 0;
                      }
                    });
                  }
                  return {
                    ...prev,
                    phase: "complete",
                    statusMessage: "",
                    selectedItems: selections,
                    timings: {
                      ...prev.timings,
                      total_ms: data.total_ms as number,
                    },
                  };
                }

                case "error":
                  return {
                    ...prev,
                    phase: "complete",
                    statusMessage: `Error: ${data.message}`,
                  };

                default:
                  return prev;
              }
            });
          },
          onError: (err) => {
            setState((prev) => ({
              ...prev,
              phase: "complete",
              statusMessage: `Error: ${err}`,
            }));
          },
          onDone: () => {},
        }
      );
    },
    [recommend]
  );

  const handleSelectItem = useCallback((slotIndex: number, itemIndex: number) => {
    setVisualized(false);
    setState((prev) => {
      const newSelected = { ...prev.selectedItems, [slotIndex]: itemIndex };

      // trigger live scoring for the user-assembled outfit
      const asins: string[] = [];
      for (let i = 0; i < prev.slots.length; i++) {
        const slot = prev.slots[i];
        if (!slot) continue;
        const idx = newSelected[i] ?? 0;
        const product = slot.products[idx];
        if (product) asins.push(product.asin);
      }

      if (asins.length >= 2) {
        setScoringPending(true);
        fetch("/api/score-outfit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ asins }),
        })
          .then((r) => r.json())
          .then((data) => {
            setUserCp(data.outfit_cp ?? null);
            setScoringPending(false);
          })
          .catch(() => setScoringPending(false));
      }

      return { ...prev, selectedItems: newSelected };
    });
  }, []);

  const handleClickProduct = useCallback((product: Product) => {
    setModalProduct(product);
  }, []);

  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);

  const handleVisualize = useCallback(() => {
    // collect selected ASINs
    const asins: string[] = [];
    for (let i = 0; i < state.slots.length; i++) {
      const slot = state.slots[i];
      if (!slot) continue;
      const idx = state.selectedItems[i] ?? 0;
      const product = slot.products[idx];
      if (product) asins.push(product.asin);
    }

    if (asins.length === 0) return;

    setGenerating(true);
    setVisualized(true);

    fetch("/api/visualize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        asins,
        query: currentQuery,
        occasion: state.occasion ?? "",
        gender: currentGender.toLowerCase(),
      }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.image) {
          setGeneratedImage(`data:image/png;base64,${data.image}`);
        }
        setGenerating(false);
      })
      .catch(() => setGenerating(false));
  }, [state.slots, state.selectedItems, state.occasion, currentQuery, currentGender]);

  // determine hero state
  const isCustom = state.otResult
    ? Object.entries(state.selectedItems).some(([slotIdx, itemIdx]) => {
        const bestOutfit = state.otResult!.outfits?.[0];
        if (!bestOutfit) return false;
        const slot = state.slots[Number(slotIdx)];
        if (!slot) return false;
        const bestAsin = bestOutfit.items[Number(slotIdx)]?.asin;
        return slot.products[itemIdx]?.asin !== bestAsin;
      })
    : false;

  const heroState = visualized
    ? "generated"
    : state.otResult || (state.phase === "complete" && state.slots.length > 0)
    ? "filled"
    : state.isScoring
    ? "shimmer"
    : state.planSlots.length > 0
    ? "skeleton"
    : "skeleton";

  const showHero = state.phase !== "landing" && state.intent === "outfit" && (state.planSlots.length > 0 || state.slots.length > 0);
  const showSelections = state.intent === "outfit" && state.phase === "complete" && state.slots.length > 0;

  // -- Landing --
  if (state.phase === "landing") {
    return (
      <div className="app-frame">
        <Landing
          catalogLabel={catalogLabel}
          onSubmit={handleSubmit}
          dark={dark}
          onToggleTheme={toggleTheme}
        />
      </div>
    );
  }

  // -- Streaming / Complete --
  const slotCategories =
    state.planSlots.length > 0
      ? state.planSlots.map((s) => s.category.join("+"))
      : state.slots.map((s) => s?.category ?? "");

  return (
    <div className="app-frame">
      <div className="app-content">
        {/* Compact header */}
        <div className="results-header">
          <div className="results-logo home-btn" onClick={handleHome}>
            <Shirt size={16} color="var(--bt)" />
            <span className="logo-text">FIT-KIT</span>
          </div>
          <SearchBar
            compact
            defaultQuery={currentQuery}
            defaultGender={currentGender}
            defaultOptions={currentOptions}
            onSubmit={handleSubmit}
          />
          <button className="theme-toggle" onClick={toggleTheme}>
            {dark ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>

        {/* Status bar */}
        {state.statusMessage && (
          <div className="status-bar">
            {state.phase === "streaming" && !state.statusMessage.startsWith("All") ? (
              <div className="pulse-dot" style={{
                background: state.isScoring ? "var(--at)" : "var(--bt)"
              }} />
            ) : state.statusMessage.startsWith("All") || state.statusMessage.includes("planned") ? (
              <Check size={12} color="var(--tt)" />
            ) : null}
            <span
              style={{
                color: state.isScoring
                  ? "var(--at)"
                  : state.statusMessage.startsWith("All") || state.statusMessage.includes("planned")
                  ? "var(--tt)"
                  : "var(--bt)",
              }}
            >
              {state.statusMessage}
            </span>
            {state.intent && state.slots.length === 0 && (
              <span className="intent-badge">{state.intent.replace("_", " ")}</span>
            )}
            {state.occasion && state.slots.length > 0 && (
              <span className="occasion-badge">{state.occasion}</span>
            )}
          </div>
        )}

        {/* Hero outfit card */}
        {showHero && (
          <HeroOutfit
            state={heroState as "skeleton" | "shimmer" | "filled" | "generated"}
            slots={state.slots.filter(Boolean)}
            otResult={state.otResult}
            selectedItems={state.selectedItems}
            isCustom={isCustom}
            userCp={userCp}
            scoringPending={scoringPending}
            generatedImage={generatedImage}
            generating={generating}
            onVisualize={handleVisualize}
          />
        )}

        {/* Slot rows */}
        {slotCategories.map((cat, i) => (
          <SlotRow
            key={cat || i}
            slot={state.slots[i]}
            category={cat}
            filled={!!state.slots[i]}
            showSelections={showSelections}
            selectedIndex={state.selectedItems[i]}
            onSelectItem={showSelections ? handleSelectItem : undefined}
            onClickProduct={handleClickProduct}
          />
        ))}
      </div>

      <PipelineFooter
        timings={state.timings}
        statusText={state.phase === "streaming" ? state.statusMessage : undefined}
      />

      {/* Product detail modal */}
      {modalProduct && (
        <ProductModal
          product={modalProduct}
          onClose={() => setModalProduct(null)}
        />
      )}
    </div>
  );
}

export default App;