import { useState } from "react";
import { Sparkles, Star, Image, Brain, Search, ChevronRight, Sun, Moon } from "lucide-react";
import { SearchBar } from "./SearchBar";
import type { SearchOptions } from "./SearchBar";

interface Props {
  catalogLabel: string;
  onSubmit: (query: string, image: string | null, gender: string, options?: SearchOptions) => void;
  dark: boolean;
  onToggleTheme: () => void;
}

const CHIPS = [
  "Golf outfit with white shoes", 
  "Hiking outfit",
  "Chelsea boots",
  "I need a flowy outfit to attend a wedding in Italy this summer as a guest", 
];

const PIPELINE = [
  { icon: Brain, label: "Plan", color: "bl" },
  { icon: Search, label: "Search", color: "tl" },
  { icon: Sparkles, label: "Score", color: "al" },
  { icon: Image, label: "Visualize", color: "pl", dim: true },
];

export function Landing({ catalogLabel, onSubmit, dark, onToggleTheme }: Props) {
  const [gender, setGender] = useState("Women");
  const handleChip = (text: string) => onSubmit(text, null, gender);

  return (
    <div className="landing">
      <div className="landing-header">
        <div className="results-logo">
          <Sparkles size={16} color="var(--bt)" />
          <span className="logo-text">FIT-KIT</span>
        </div>
        <div className="landing-header-right">
          {catalogLabel && (
            <>
              <span className="catalog-label">{catalogLabel}</span>
              <span className="status-dot" />
            </>
          )}
          <button className="theme-toggle" onClick={onToggleTheme}>
            {dark ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </div>
      <div className="landing-content">
        <div className="landing-left">
          <div className="landing-badge">
            <Sparkles size={11} />
            Semantic outfit search
          </div>

          <h1 className="landing-headline">
            Find anything
            <br />
            to wear.
          </h1>

          <p className="landing-sub">Search by text, image, or both.</p>

          <SearchBar onSubmit={onSubmit} onGenderChange={setGender} />

          <div className="chip-row">
            {CHIPS.map((c) => (
              <button key={c} className="chip" onClick={() => handleChip(c)}>
                {c}
              </button>
            ))}
          </div>
        </div>

        <div className="landing-right">
          <div className="preview-card">
            <div className="preview-header">
              <Star size={14} color="var(--tt)" />
              <span>Example result</span>
            </div>
            {["👔", "👖", "👞"].map((emoji) => (
              <div className="preview-row" key={emoji}>
                {[1, 0.4, 0.2].map((op, j) => (
                  <div key={j} className="preview-cell" style={{ opacity: op }}>
                    {emoji}
                  </div>
                ))}
              </div>
            ))}
          </div>

          <div className="visualize-hint">
            <div className="visualize-hint-icon">
              <Image size={18} color="var(--pt)" />
            </div>
            <div>
              <div className="visualize-hint-title">Visualize the look</div>
              <div className="visualize-hint-sub">Generate a preview</div>
            </div>
          </div>
        </div>
      </div>

      <div className="pipeline-footer">
        {PIPELINE.map((step, i) => (
          <div key={step.label} className="pipeline-step-wrapper">
            {i > 0 && <ChevronRight size={10} color="var(--t4)" />}
            <div
              className="pipeline-step"
              style={{ opacity: step.dim ? 0.55 : 1 }}
            >
              <div className={`pipeline-icon bg-${step.color}`}>
                <step.icon size={10} />
              </div>
              <span>{step.label}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}