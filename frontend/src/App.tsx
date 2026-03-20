import { useState, useMemo, useEffect, useRef, useCallback } from "react";

// ── Types ──────────────────────────────────────────────────
interface ScoringBreakdown {
  pattern_confidence: number;
  feature_score: number;
  strategy_score: number;
  regime_alignment: number;
  backtest_edge: number;
  volume_confirm: number;
}

interface Setup {
  pattern_name: string;
  category: string;
  symbol: string;
  bias: string;
  timeframe_detected: string;
  multi_tf: boolean;
  entry_price: number;
  stop_loss: number;
  target_price: number;
  risk_reward_ratio: number;
  confidence: number;
  composite_score: number;
  scoring: ScoringBreakdown;
  detected_at: string;
  description: string;
  win_rate: number;
  max_attempts: number;
  exit_strategy: string;
  key_levels: Record<string, number>;
  ideal_time: string;
  regime: string;
}

interface ChartBar {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface RegimeData {
  regime: string;
  atr_ratio: number;
}

interface HotStrategy {
  name: string;
  strategy_type: string;
  win_rate: number;
  profit_factor: number;
  expectancy: number;
  hot_score: number;
  total_signals: number;
}

const API = "http://localhost:8000";

// ── Bean Logo ──────────────────────────────────────────────
function BeanLogo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="none">
      <ellipse cx="50" cy="54" rx="36" ry="40" fill="#8B5E3C" />
      <ellipse cx="50" cy="52" rx="33" ry="37" fill="#A0714F" />
      <path d="M50 22C50 22 42 42 42 54C42 66 50 86 50 86" stroke="#6B4226" strokeWidth="3" strokeLinecap="round" opacity="0.5" />
      <circle cx="38" cy="42" r="4" fill="#2D1B0E" />
      <circle cx="62" cy="42" r="4" fill="#2D1B0E" />
      <circle cx="39" cy="41" r="1.2" fill="white" />
      <circle cx="63" cy="41" r="1.2" fill="white" />
      <path d="M42 56Q50 64 58 56" stroke="#2D1B0E" strokeWidth="2.5" strokeLinecap="round" fill="none" />
      <path d="M26 30L20 20" stroke="#A0714F" strokeWidth="2" strokeLinecap="round" opacity="0.6" />
      <path d="M74 30L80 20" stroke="#A0714F" strokeWidth="2" strokeLinecap="round" opacity="0.6" />
      <path d="M18 22L22 16" stroke="#4CAF50" strokeWidth="2" strokeLinecap="round" />
      <path d="M82 22L78 16" stroke="#EF5350" strokeWidth="2" strokeLinecap="round" />
      <path d="M84 24L88 20" stroke="#EF5350" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

// ── Regime Banner ──────────────────────────────────────────
const REGIME_CONFIG: Record<string, { label: string; emoji: string; bg: string; color: string; desc: string }> = {
  trending_bull: { label: "TRENDING BULL", emoji: "🟢", bg: "#e8f5e9", color: "#2e7d32", desc: "Favor momentum & breakout" },
  trending_bear: { label: "TRENDING BEAR", emoji: "🔴", bg: "#ffebee", color: "#c62828", desc: "Favor momentum short & breakout" },
  high_volatility: { label: "HIGH VOLATILITY", emoji: "🟡", bg: "#fff8e1", color: "#e65100", desc: "Favor scalps & mean reversion" },
  mean_reverting: { label: "MEAN REVERTING", emoji: "⚪", bg: "#f3e5f5", color: "#6a1b9a", desc: "Favor mean reversion & scalps" },
};

function RegimeBanner({ regime }: { regime: RegimeData | null }) {
  if (!regime) return null;
  const cfg = REGIME_CONFIG[regime.regime] || { label: regime.regime, emoji: "❓", bg: "#f5f5f5", color: "#666", desc: "" };
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 12, padding: "8px 16px",
      background: cfg.bg, borderRadius: 10, marginBottom: 16,
      border: `1px solid ${cfg.color}22`,
    }}>
      <span style={{ fontSize: 18 }}>{cfg.emoji}</span>
      <div>
        <span style={{ fontSize: 12, fontWeight: 800, color: cfg.color, letterSpacing: 0.5 }}>{cfg.label}</span>
        <span style={{ fontSize: 11, color: cfg.color, opacity: 0.7, marginLeft: 8 }}>ATR Ratio: {regime.atr_ratio.toFixed(3)}</span>
      </div>
      <span style={{ fontSize: 11, color: cfg.color, opacity: 0.6, marginLeft: "auto" }}>{cfg.desc}</span>
    </div>
  );
}

// ── Hot Strategies Panel ───────────────────────────────────
function HotStrategiesPanel({ strategies }: { strategies: HotStrategy[] }) {
  if (!strategies || strategies.length === 0) return null;
  return (
    <div style={{
      padding: "12px 16px", background: "#faf8f5", borderRadius: 10,
      border: "1px solid #f0ebe3", marginBottom: 16,
    }}>
      <div style={{ fontSize: 11, fontWeight: 800, color: "#a08060", letterSpacing: 0.5, marginBottom: 8 }}>
        🔥 HOT STRATEGIES
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {strategies.map(s => (
          <div key={s.name} style={{
            padding: "6px 12px", background: "#fff", borderRadius: 8,
            border: "1px solid #e8dfd0", fontSize: 11,
          }}>
            <span style={{ fontWeight: 800, color: "#3d2b1f" }}>{s.name}</span>
            <span style={{ color: "#a08060", marginLeft: 6 }}>
              {(s.win_rate * 100).toFixed(0)}% WR
            </span>
            <span style={{ color: "#2e7d32", marginLeft: 4 }}>
              PF {s.profit_factor.toFixed(1)}
            </span>
            <span style={{
              marginLeft: 6, padding: "1px 6px", borderRadius: 4, fontWeight: 800, fontSize: 10,
              background: s.hot_score >= 70 ? "#e8f5e9" : "#fff3e0",
              color: s.hot_score >= 70 ? "#2e7d32" : "#e65100",
            }}>
              {s.hot_score.toFixed(0)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Category Badge ─────────────────────────────────────────
const CAT_STYLE: Record<string, { bg: string; color: string; label: string }> = {
  classical: { bg: "#e3f2fd", color: "#1565c0", label: "Classical" },
  candlestick: { bg: "#fff3e0", color: "#e65100", label: "Candle" },
  smb_scalp: { bg: "#f3e5f5", color: "#6a1b9a", label: "SMB" },
  quant: { bg: "#e8f5e9", color: "#2e7d32", label: "Quant" },
};

function CatBadge({ category }: { category: string }) {
  const cfg = CAT_STYLE[category] || { bg: "#f5f5f5", color: "#666", label: category };
  return (
    <span style={{
      fontSize: 9, fontWeight: 800, padding: "2px 6px", borderRadius: 3,
      background: cfg.bg, color: cfg.color, textTransform: "uppercase", letterSpacing: 0.3,
    }}>
      {cfg.label}
    </span>
  );
}

// ── Score Bar ──────────────────────────────────────────────
function ScoreBar({ score, label }: { score: number; label: string }) {
  const color = score >= 70 ? "#2e7d32" : score >= 50 ? "#e65100" : "#c62828";
  return (
    <div style={{ width: 48, textAlign: "center" }}>
      <div style={{ fontSize: 9, color: "#a08060", fontWeight: 700 }}>{label}</div>
      <div style={{
        fontFamily: "monospace", fontSize: 14, fontWeight: 900, color,
        background: score >= 70 ? "#e8f5e9" : score >= 50 ? "#fff3e0" : "#ffebee",
        borderRadius: 4, padding: "1px 4px",
      }}>
        {score.toFixed(0)}
      </div>
    </div>
  );
}

// ── Scoring Tooltip ────────────────────────────────────────
function ScoringTooltip({ scoring }: { scoring: ScoringBreakdown }) {
  if (!scoring) return null;
  const factors = [
    { label: "Pattern", value: scoring.pattern_confidence, weight: "20%" },
    { label: "Features", value: scoring.feature_score, weight: "25%" },
    { label: "Strategy", value: scoring.strategy_score, weight: "20%" },
    { label: "Regime", value: scoring.regime_alignment, weight: "15%" },
    { label: "Backtest", value: scoring.backtest_edge, weight: "10%" },
    { label: "Volume", value: scoring.volume_confirm, weight: "10%" },
  ];
  return (
    <div style={{
      display: "flex", gap: 6, flexWrap: "wrap", padding: "8px 12px",
      background: "#faf8f5", borderRadius: 8, border: "1px solid #f0ebe3",
      marginTop: 8,
    }}>
      {factors.map(f => {
        const c = f.value >= 70 ? "#2e7d32" : f.value >= 40 ? "#e65100" : "#c62828";
        return (
          <div key={f.label} style={{ textAlign: "center", minWidth: 52 }}>
            <div style={{ fontSize: 9, color: "#a08060" }}>{f.label} ({f.weight})</div>
            <div style={{ fontFamily: "monospace", fontSize: 12, fontWeight: 800, color: c }}>
              {f.value.toFixed(0)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── TF Badge ───────────────────────────────────────────────
function TfBadge({ tf }: { tf: string }) {
  const isMulti = tf.includes("&");
  return (
    <span style={{
      fontSize: 10, fontWeight: 800, padding: "2px 7px", borderRadius: 4,
      background: isMulti ? "#e8eaf6" : "#f0ebe3",
      color: isMulti ? "#283593" : "#8B5E3C",
      border: isMulti ? "1px solid #c5cae9" : "none",
    }}>
      {isMulti ? "★ " : ""}{tf}
    </span>
  );
}

// ── Trade Chart ────────────────────────────────────────────
function TradeChart({ setup, onClose }: { setup: Setup; onClose: () => void }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!containerRef.current) return;
    let chart: ReturnType<typeof import("lightweight-charts").createChart> | null = null;

    const loadChart = async () => {
      try {
        const lc = await import("lightweight-charts");
        const tf = setup.timeframe_detected.includes("15m") ? "15min" : "5min";
        const res = await fetch(`${API}/api/chart/${setup.symbol}?timeframe=${tf}&days_back=5`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        const bars: ChartBar[] = data.bars;
        if (!bars || bars.length === 0) throw new Error("No chart data");

        if (containerRef.current) containerRef.current.innerHTML = "";

        chart = lc.createChart(containerRef.current!, {
          width: containerRef.current!.clientWidth, height: 380,
          layout: { background: { color: "#fdf8f0" } as { color: string }, textColor: "#5d4037", fontFamily: "Nunito, sans-serif" },
          grid: { vertLines: { color: "#f0ebe3" }, horzLines: { color: "#f0ebe3" } },
          crosshair: { mode: lc.CrosshairMode.Normal },
          rightPriceScale: { borderColor: "#e8dfd0" },
          timeScale: { borderColor: "#e8dfd0", timeVisible: true },
        });

        const series = chart.addCandlestickSeries({
          upColor: "#2e7d32", downColor: "#c62828",
          borderUpColor: "#2e7d32", borderDownColor: "#c62828",
          wickUpColor: "#2e7d32", wickDownColor: "#c62828",
        });
        series.setData(bars as any);

        series.createPriceLine({ price: setup.entry_price, color: "#1565c0", lineWidth: 2, lineStyle: lc.LineStyle.Dashed, axisLabelVisible: true, title: "ENTRY" });
        series.createPriceLine({ price: setup.stop_loss, color: "#c62828", lineWidth: 2, lineStyle: lc.LineStyle.Dashed, axisLabelVisible: true, title: "STOP" });
        series.createPriceLine({ price: setup.target_price, color: "#2e7d32", lineWidth: 2, lineStyle: lc.LineStyle.Dashed, axisLabelVisible: true, title: "TARGET" });

        Object.entries(setup.key_levels || {}).forEach(([name, price]) => {
          if (typeof price === "number" && price > 0) {
            series.createPriceLine({ price, color: "#a08060", lineWidth: 1, lineStyle: lc.LineStyle.Dotted, axisLabelVisible: false, title: name });
          }
        });

        chart.timeScale().fitContent();
        setLoading(false);

        const ro = new ResizeObserver(() => {
          if (containerRef.current && chart) chart.applyOptions({ width: containerRef.current.clientWidth });
        });
        ro.observe(containerRef.current!);
        return () => ro.disconnect();
      } catch (e: unknown) { setError(e instanceof Error ? e.message : "Chart failed"); setLoading(false); }
    };
    loadChart();
    return () => { if (chart) chart.remove(); };
  }, [setup]);

  return (
    <div style={{ background: "#fdf8f0", border: "1.5px solid #e8dfd0", borderRadius: 12, padding: 16, marginTop: 8, marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <div style={{ fontSize: 13, fontWeight: 800, color: "#3d2b1f" }}>
          {setup.symbol} — {setup.pattern_name} ({setup.timeframe_detected})
        </div>
        <div style={{ display: "flex", gap: 12, fontSize: 11, fontWeight: 700, alignItems: "center" }}>
          <span style={{ color: "#1565c0" }}>Entry ${setup.entry_price.toFixed(2)}</span>
          <span style={{ color: "#c62828" }}>Stop ${setup.stop_loss.toFixed(2)}</span>
          <span style={{ color: "#2e7d32" }}>Target ${setup.target_price.toFixed(2)}</span>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 16, color: "#a08060", padding: "0 4px" }}>✕</button>
        </div>
      </div>
      <ScoringTooltip scoring={setup.scoring} />
      {loading && <div style={{ textAlign: "center", padding: 40, color: "#a08060" }}>Loading chart...</div>}
      {error && <div style={{ textAlign: "center", padding: 20, color: "#c62828", fontSize: 12 }}>{error}</div>}
      <div ref={containerRef} style={{ width: "100%", minHeight: loading ? 0 : 380, marginTop: 8 }} />
    </div>
  );
}

// ── Setup Row ──────────────────────────────────────────────
function SetupRow({ s, isChartOpen, onToggleChart }: { s: Setup; isChartOpen: boolean; onToggleChart: () => void }) {
  const isLong = s.bias === "long";
  const detected = new Date(s.detected_at);
  const timeStr = detected.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const dateStr = detected.toLocaleDateString([], { month: "short", day: "numeric" });

  return (
    <div>
      <div onClick={onToggleChart} style={{
        display: "flex", alignItems: "center", gap: 8, padding: "10px 0",
        borderBottom: "1px solid #f0ebe3", cursor: "pointer",
        background: isChartOpen ? "#faf8f5" : "transparent", transition: "background 0.15s",
      }}>
        {/* Symbol + Bias */}
        <div style={{ width: 68 }}>
          <div style={{ fontSize: 14, fontWeight: 800, color: "#3d2b1f" }}>{s.symbol}</div>
          <span style={{
            fontSize: 10, fontWeight: 700, padding: "1px 7px", borderRadius: 100,
            background: isLong ? "#e8f5e9" : "#ffebee", color: isLong ? "#2e7d32" : "#c62828",
          }}>{s.bias.toUpperCase()}</span>
        </div>

        {/* Pattern + Category + TF + Time */}
        <div style={{ flex: 1, minWidth: 130 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 5, flexWrap: "wrap" }}>
            <span style={{ fontSize: 12, fontWeight: 700, color: "#3d2b1f" }}>{s.pattern_name}</span>
            <CatBadge category={s.category} />
            <TfBadge tf={s.timeframe_detected} />
          </div>
          <div style={{ fontSize: 10, color: "#a08060", marginTop: 1 }}>{dateStr} {timeStr}</div>
        </div>

        {/* Prices */}
        <div style={{ display: "flex", gap: 10 }}>
          {[
            { l: "ENTRY", v: s.entry_price, c: "#3d2b1f" },
            { l: "STOP", v: s.stop_loss, c: "#c62828" },
            { l: "TARGET", v: s.target_price, c: "#2e7d32" },
          ].map(({ l, v, c }) => (
            <div key={l} style={{ textAlign: "right" }}>
              <div style={{ fontSize: 9, color: "#a08060", fontWeight: 600 }}>{l}</div>
              <div style={{ fontFamily: "monospace", fontSize: 12, fontWeight: 700, color: c }}>${v.toFixed(2)}</div>
            </div>
          ))}
        </div>

        {/* R:R */}
        <div style={{ width: 40, textAlign: "center" }}>
          <div style={{ fontSize: 9, color: "#a08060" }}>R:R</div>
          <div style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 800, color: s.risk_reward_ratio >= 2 ? "#2e7d32" : "#3d2b1f" }}>
            {s.risk_reward_ratio.toFixed(1)}
          </div>
        </div>

        {/* Composite Score */}
        <ScoreBar score={s.composite_score} label="SCORE" />

        {/* Chart toggle */}
        <div style={{ width: 18, textAlign: "center", fontSize: 13, color: "#c4a882" }}>
          {isChartOpen ? "▴" : "▾"}
        </div>
      </div>

      {isChartOpen && <TradeChart setup={s} onClose={onToggleChart} />}
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────
function App() {
  const [symbol, setSymbol] = useState("AAPL");
  const [setups, setSetups] = useState<Setup[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [chartIdx, setChartIdx] = useState<number | null>(null);
  const [filterBias, setFilterBias] = useState("ALL");
  const [filterCat, setFilterCat] = useState("ALL");
  const [hasScanned, setHasScanned] = useState(false);
  const [sortBy, setSortBy] = useState<"score" | "rr" | "confidence">("score");
  const [mode, setMode] = useState<"today" | "active">("today");
  const [regime, setRegime] = useState<RegimeData | null>(null);
  const [hotStrategies, setHotStrategies] = useState<HotStrategy[]>([]);
  const [patternCount, setPatternCount] = useState(47);

  // Load regime + hot strategies on mount
  useEffect(() => {
    fetch(`${API}/api/regime`).then(r => r.json()).then(d => {
      if (d.regime) setRegime(d);
    }).catch(() => {});
    fetch(`${API}/api/hot-strategies?top_n=5`).then(r => r.json()).then(d => {
      if (d.strategies) setHotStrategies(d.strategies);
    }).catch(() => {});
    fetch(`${API}/api/health`).then(r => r.json()).then(d => {
      if (d.patterns) setPatternCount(d.patterns);
    }).catch(() => {});
  }, []);

  const handleScan = useCallback(async () => {
    setLoading(true); setError(""); setSetups([]); setHasScanned(true); setChartIdx(null);
    try {
      const res = await fetch(`${API}/api/scan?symbol=${symbol}&mode=${mode}`);
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setSetups(data.setups);
    } catch (e: unknown) { setError(e instanceof Error ? e.message : "Unknown error"); }
    finally { setLoading(false); }
  }, [symbol, mode]);

  const filtered = useMemo(() => {
    let result = setups;
    if (filterBias !== "ALL") result = result.filter(s => s.bias === filterBias.toLowerCase());
    if (filterCat !== "ALL") result = result.filter(s => s.category === filterCat);
    return [...result].sort((a, b) => {
      if (sortBy === "rr") return b.risk_reward_ratio - a.risk_reward_ratio;
      if (sortBy === "confidence") return b.confidence - a.confidence;
      return (b.composite_score || 0) - (a.composite_score || 0);
    });
  }, [setups, filterBias, filterCat, sortBy]);

  const longs = setups.filter(s => s.bias === "long").length;
  const shorts = setups.filter(s => s.bias === "short").length;
  const multiTf = setups.filter(s => s.multi_tf).length;

  const pill = (active: boolean, small = false) => ({
    fontSize: small ? 11 : 12, fontWeight: active ? 800 : 600 as const,
    padding: small ? "4px 10px" : "6px 14px", borderRadius: 100,
    cursor: "pointer" as const, border: "none",
    background: active ? "#3d2b1f" : "transparent",
    color: active ? "#fff" : "#a08060", fontFamily: "'Nunito', sans-serif",
  });

  return (
    <div style={{ background: "#fdf8f0", minHeight: "100vh", fontFamily: "'Nunito', sans-serif", color: "#3d2b1f" }}>
      {/* Header */}
      <div style={{
        padding: "12px 32px", borderBottom: "1px solid #e8dfd0",
        display: "flex", justifyContent: "space-between", alignItems: "center", background: "#faf5ea",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <BeanLogo size={32} />
          <span style={{ fontSize: 22, fontWeight: 900, letterSpacing: -0.5 }}>AlphaBean</span>
          <span style={{ fontSize: 10, color: "#a08060", fontWeight: 600 }}>v3.0</span>
          <span style={{ fontSize: 9, color: "#c4a882", fontWeight: 600, marginLeft: 4 }}>
            {patternCount} patterns • 6-factor scoring
          </span>
        </div>
        <div style={{ display: "flex", gap: 12, fontSize: 12, fontWeight: 700, alignItems: "center" }}>
          {setups.length > 0 && (
            <>
              <span style={{ color: "#2e7d32" }}>{longs} Long</span>
              <span style={{ color: "#c62828" }}>{shorts} Short</span>
              {multiTf > 0 && <span style={{ color: "#283593" }}>★ {multiTf} Multi-TF</span>}
              <span style={{ color: "#a08060" }}>{setups.length} Total</span>
            </>
          )}
        </div>
      </div>

      <div style={{ maxWidth: 1020, margin: "0 auto", padding: "20px 32px" }}>
        {/* Regime Banner */}
        <RegimeBanner regime={regime} />

        {/* Hot Strategies */}
        <HotStrategiesPanel strategies={hotStrategies} />

        {/* Scan Controls */}
        <div style={{ display: "flex", gap: 10, alignItems: "end", marginBottom: 18, flexWrap: "wrap" }}>
          <div>
            <label style={{ fontSize: 11, color: "#a08060", display: "block", marginBottom: 4, fontWeight: 700 }}>SYMBOL</label>
            <input type="text" value={symbol}
              onChange={e => setSymbol(e.target.value.toUpperCase())}
              onKeyDown={e => e.key === "Enter" && handleScan()}
              placeholder="AAPL"
              style={{
                fontSize: 14, padding: "8px 14px", borderRadius: 10,
                border: "1.5px solid #e8dfd0", width: 110, fontWeight: 800, background: "#fff",
              }}
            />
          </div>

          <div>
            <label style={{ fontSize: 11, color: "#a08060", display: "block", marginBottom: 4, fontWeight: 700 }}>MODE</label>
            <div style={{ display: "flex", gap: 3, background: "#f0ebe3", borderRadius: 100, padding: 3 }}>
              <button style={pill(mode === "today")} onClick={() => setMode("today")}>Today</button>
              <button style={pill(mode === "active")} onClick={() => setMode("active")}>Active</button>
            </div>
          </div>

          <button onClick={handleScan} disabled={loading || !symbol} style={{
            fontSize: 14, fontWeight: 800, padding: "9px 28px", borderRadius: 10, border: "none",
            background: loading ? "#c4a882" : "#3d2b1f", color: "#fff",
            cursor: loading ? "wait" : "pointer",
          }}>
            {loading ? "Scanning..." : "Scan"}
          </button>

          <div style={{ fontSize: 11, color: "#a08060", padding: "8px 0", fontWeight: 600 }}>
            Auto-scans 5min + 15min • {patternCount} patterns
          </div>
        </div>

        {/* Error */}
        {error && (
          <div style={{ padding: "12px 16px", borderRadius: 10, background: "#ffebee", color: "#c62828", fontSize: 13, marginBottom: 16 }}>
            <strong>Error:</strong> {error}
            <div style={{ marginTop: 4, fontSize: 11, color: "#a08060" }}>
              Backend: <code style={{ background: "#fff", padding: "1px 6px", borderRadius: 4 }}>uvicorn backend.main:app --reload --port 8000</code>
            </div>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div style={{ textAlign: "center", padding: 48, color: "#a08060" }}>
            <BeanLogo size={48} />
            <div style={{ fontSize: 14, marginTop: 12, fontWeight: 700 }}>Scanning 5min + 15min bars...</div>
            <div style={{ fontSize: 12, marginTop: 4 }}>Running {patternCount} detectors × 2 timeframes</div>
          </div>
        )}

        {/* Results */}
        {!loading && setups.length > 0 && (
          <>
            {/* Filters */}
            <div style={{ display: "flex", gap: 10, marginBottom: 14, flexWrap: "wrap", alignItems: "center" }}>
              <div style={{ display: "flex", gap: 3, background: "#f0ebe3", borderRadius: 100, padding: 3 }}>
                {["ALL", "LONG", "SHORT"].map(b => (
                  <button key={b} style={pill(filterBias === b)} onClick={() => setFilterBias(b)}>{b}</button>
                ))}
              </div>
              <div style={{ display: "flex", gap: 3, background: "#f0ebe3", borderRadius: 100, padding: 3 }}>
                {[
                  { v: "ALL", l: "All" },
                  { v: "classical", l: "Classical" },
                  { v: "candlestick", l: "Candle" },
                  { v: "smb_scalp", l: "SMB" },
                  { v: "quant", l: "Quant" },
                ].map(c => (
                  <button key={c.v} style={pill(filterCat === c.v, true)} onClick={() => setFilterCat(c.v)}>{c.l}</button>
                ))}
              </div>
              <div style={{ marginLeft: "auto", display: "flex", gap: 4, alignItems: "center", fontSize: 11, color: "#a08060" }}>
                <span style={{ fontWeight: 700 }}>Sort:</span>
                {([["score", "Score"], ["rr", "R:R"], ["confidence", "Conf"]] as const).map(([key, label]) => (
                  <button key={key} onClick={() => setSortBy(key as any)} style={pill(sortBy === key, true)}>{label}</button>
                ))}
              </div>
            </div>

            {/* Column headers */}
            <div style={{
              display: "flex", alignItems: "center", gap: 8, padding: "8px 0",
              borderBottom: "2px solid #3d2b1f", fontSize: 9, fontWeight: 800,
              color: "#a08060", letterSpacing: 0.6, textTransform: "uppercase",
            }}>
              <div style={{ width: 68 }}>Symbol</div>
              <div style={{ flex: 1 }}>Pattern / Category / TF</div>
              <div style={{ display: "flex", gap: 10 }}>
                <div style={{ width: 52, textAlign: "right" }}>Entry</div>
                <div style={{ width: 52, textAlign: "right" }}>Stop</div>
                <div style={{ width: 52, textAlign: "right" }}>Target</div>
              </div>
              <div style={{ width: 40, textAlign: "center" }}>R:R</div>
              <div style={{ width: 48, textAlign: "center" }}>Score</div>
              <div style={{ width: 18 }} />
            </div>

            {filtered.map((s, i) => (
              <SetupRow
                key={`${s.symbol}-${s.pattern_name}-${s.timeframe_detected}-${i}`}
                s={s} isChartOpen={chartIdx === i}
                onToggleChart={() => setChartIdx(chartIdx === i ? null : i)}
              />
            ))}

            {filtered.length === 0 && (
              <div style={{ textAlign: "center", padding: 32, color: "#a08060", fontSize: 13 }}>
                No setups match your filters.
              </div>
            )}
          </>
        )}

        {/* Empty state */}
        {!loading && hasScanned && setups.length === 0 && !error && (
          <div style={{ textAlign: "center", padding: 48, color: "#a08060" }}>
            <BeanLogo size={48} />
            <div style={{ fontSize: 16, fontWeight: 700, marginTop: 12 }}>No patterns detected</div>
            <div style={{ fontSize: 12, marginTop: 4 }}>
              <strong>{symbol}</strong> has no active setups on 5min or 15min right now. Try "Active" mode or a different ticker.
            </div>
          </div>
        )}

        {/* Welcome */}
        {!loading && !hasScanned && (
          <div style={{ textAlign: "center", padding: 48, color: "#a08060" }}>
            <BeanLogo size={56} />
            <div style={{ fontSize: 18, fontWeight: 800, marginTop: 12, color: "#3d2b1f" }}>Welcome to AlphaBean v3.0</div>
            <div style={{ fontSize: 13, marginTop: 6, maxWidth: 480, margin: "6px auto 0" }}>
              Type a ticker and hit Scan. AlphaBean runs {patternCount} pattern detectors across 4 categories
              (Classical, Candlestick, SMB Scalps, Quant) on both 5min and 15min bars simultaneously.
              Each setup is scored using a 6-factor composite: pattern confidence, statistical features,
              strategy performance, regime alignment, backtest edge, and volume confirmation.
              Click any row to see the candlestick chart with entry, stop, and target levels drawn.
            </div>
          </div>
        )}

        <div style={{ textAlign: "center", marginTop: 40, padding: "16px 0", borderTop: "1px solid #f0ebe3" }}>
          <span style={{ fontSize: 11, color: "#c4a882" }}>AlphaBean v3.0 — {patternCount} Detectors — 6-Factor Scoring — 5m + 15m Auto-Scan</span>
        </div>
      </div>
    </div>
  );
}

export default App;