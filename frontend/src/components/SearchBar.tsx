import { useState } from "react";
import { Search, Camera, ArrowRight, SlidersHorizontal, X, ChevronDown, ChevronUp } from "lucide-react";

export interface SearchFilters {
  category?: string[];
  season?: string[];
  formality?: string[];
  color?: string[];
}

export interface SearchOptions {
  top_k: number;
  alpha: number;
  beta: number;
  filters: SearchFilters;
}

const DEFAULT_OPTIONS: SearchOptions = {
  top_k: 24,
  alpha: 0.8,
  beta: 0.4,
  filters: {},
};

interface Props {
  compact?: boolean;
  defaultQuery?: string;
  defaultGender?: string;
  defaultOptions?: SearchOptions;
  onSubmit: (query: string, image: string | null, gender: string, options?: SearchOptions) => void;
  onGenderChange?: (gender: string) => void;
}

const GENDERS = ["Men", "Women", "Boys", "Girls"];

const CATEGORIES = [
  "tops", "dresses", "sweaters", "pants", "skirts", "shorts",
  "activewear", "swimwear", "outerwear", "sleepwear", "underwear", "socks",
  "shoes", "bags", "jewelry", "watches", "sunglasses", "belts",
  "scarves", "hats", "accessories",
];

const SEASONS = ["summer", "winter", "spring_fall", "all_season"];
const FORMALITIES = ["casual", "business_casual", "formal", "athletic", "loungewear"];
const COLORS = [
  "black", "white", "blue", "red", "pink", "green", "yellow", "orange",
  "purple", "brown", "grey", "navy", "beige", "gold", "multicolor",
];

export function SearchBar({ compact, defaultQuery, defaultGender, defaultOptions, onSubmit, onGenderChange }: Props) {
  const [query, setQuery] = useState(defaultQuery ?? "");
  const [gender, setGender] = useState(defaultGender ?? "Women");
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [options, setOptions] = useState<SearchOptions>(defaultOptions ?? DEFAULT_OPTIONS);

  const handleSubmit = () => {
    if (!query?.trim() && !imageBase64) return;
    const hasNonDefault =
      options.top_k !== DEFAULT_OPTIONS.top_k ||
      options.alpha !== DEFAULT_OPTIONS.alpha ||
      options.beta !== DEFAULT_OPTIONS.beta ||
      Object.keys(options.filters).length > 0;
    onSubmit(query.trim(), imageBase64, gender, hasNonDefault ? options : undefined);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setImageBase64(reader.result as string);
    reader.readAsDataURL(file);
  };

  const toggleFilter = (key: keyof SearchFilters, value: string) => {
    setOptions((prev) => {
      const current = prev.filters[key] ?? [];
      const next = current.includes(value)
        ? current.filter((v) => v !== value)
        : [...current, value];
      return {
        ...prev,
        filters: {
          ...prev.filters,
          [key]: next.length > 0 ? next : undefined,
        },
      };
    });
  };

  const activeFilterCount = Object.values(options.filters).reduce(
    (n, arr) => n + (arr?.length ?? 0), 0
  );
  const hasNonDefaultSliders =
    options.top_k !== DEFAULT_OPTIONS.top_k ||
    options.alpha !== DEFAULT_OPTIONS.alpha ||
    options.beta !== DEFAULT_OPTIONS.beta;
  const badgeCount = activeFilterCount + (hasNonDefaultSliders ? 1 : 0);

  return (
    <div className={`search-wrapper ${compact ? "search-wrapper-compact" : ""}`}>
      <div className={`search-bar ${compact ? "search-bar-compact" : ""}`}>
        <label className="search-upload-btn">
          <Camera size={compact ? 12 : 14} color="var(--t3)" />
          <input type="file" accept="image/*" hidden onChange={handleImageUpload} />
        </label>

        {imageBase64 && (
          <div className="search-image-preview">
            <img src={imageBase64} alt="upload" />
            <button className="search-image-remove" onClick={() => setImageBase64(null)}>
              <X size={8} />
            </button>
          </div>
        )}

        <Search size={compact ? 12 : 14} color="var(--t3)" />
        <input
          className="search-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          placeholder="what are you looking for?"
        />

        <select
          className="search-gender"
          value={gender}
          onChange={(e) => {
            setGender(e.target.value);
            onGenderChange?.(e.target.value);
          }}
        >
          {GENDERS.map((g) => (
            <option key={g} value={g}>{g}</option>
          ))}
        </select>

        <button
          className={`search-advanced-toggle ${showAdvanced ? "active" : ""}`}
          onClick={() => setShowAdvanced((s) => !s)}
          title="Advanced search options"
        >
          <SlidersHorizontal size={compact ? 12 : 14} />
          {badgeCount > 0 && (
            <span className="advanced-badge">{badgeCount}</span>
          )}
        </button>

        <button className="search-submit" onClick={handleSubmit}>
          <ArrowRight size={compact ? 12 : 16} color="var(--bg)" />
        </button>
      </div>

      {showAdvanced && (
        <div className="advanced-panel">
          {/* Sliders */}
          <div className="advanced-sliders">
            <SliderControl
              label="Results per slot"
              value={options.top_k}
              min={1} max={48} step={1}
              display={String(options.top_k)}
              onChange={(v) => setOptions((p) => ({ ...p, top_k: v }))}
            />
            <SliderControl
              label="Semantic ↔ Keyword"
              value={options.alpha}
              min={0} max={1} step={0.05}
              displayLeft="BM25" displayRight="Dense"
              display={options.alpha.toFixed(2)}
              onChange={(v) => setOptions((p) => ({ ...p, alpha: v }))}
            />
            <SliderControl
              label="Text ↔ Image"
              value={options.beta}
              min={0} max={1} step={0.05}
              displayLeft="Image" displayRight="Text"
              display={options.beta.toFixed(2)}
              onChange={(v) => setOptions((p) => ({ ...p, beta: v }))}
            />
          </div>

          {/* Filters */}
          <FilterSection
            label="Category"
            options={CATEGORIES}
            selected={options.filters.category ?? []}
            onToggle={(v) => toggleFilter("category", v)}
          />
          <FilterSection
            label="Season"
            options={SEASONS}
            selected={options.filters.season ?? []}
            onToggle={(v) => toggleFilter("season", v)}
          />
          <FilterSection
            label="Formality"
            options={FORMALITIES}
            selected={options.filters.formality ?? []}
            onToggle={(v) => toggleFilter("formality", v)}
          />
          <FilterSection
            label="Color"
            options={COLORS}
            selected={options.filters.color ?? []}
            onToggle={(v) => toggleFilter("color", v)}
          />

          <button
            className="advanced-reset"
            onClick={() => setOptions(DEFAULT_OPTIONS)}
          >
            Reset to defaults
          </button>
        </div>
      )}
    </div>
  );
}

/* -- Sub-components ------------------------------ */

function SliderControl({
  label, value, min, max, step, display, displayLeft, displayRight, onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  displayLeft?: string;
  displayRight?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="slider-control">
      <div className="slider-header">
        <span className="slider-label">{label}</span>
        <span className="slider-value">{display}</span>
      </div>
      <div className="slider-row">
        {displayLeft && <span className="slider-end-label">{displayLeft}</span>}
        <input
          type="range"
          className="slider-input"
          min={min} max={max} step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
        />
        {displayRight && <span className="slider-end-label">{displayRight}</span>}
      </div>
    </div>
  );
}

function FilterSection({
  label, options, selected, onToggle,
}: {
  label: string;
  options: string[];
  selected: string[];
  onToggle: (value: string) => void;
}) {
  const [expanded, setExpanded] = useState(selected.length > 0);

  return (
    <div className="filter-section">
      <button className="filter-section-header" onClick={() => setExpanded((e) => !e)}>
        <span className="filter-section-label">{label}</span>
        {selected.length > 0 && (
          <span className="filter-section-count">{selected.length}</span>
        )}
        {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>
      {expanded && (
        <div className="filter-chips">
          {options.map((opt) => (
            <button
              key={opt}
              className={`filter-chip ${selected.includes(opt) ? "filter-chip-active" : ""}`}
              onClick={() => onToggle(opt)}
            >
              {opt.replace("_", " ")}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}