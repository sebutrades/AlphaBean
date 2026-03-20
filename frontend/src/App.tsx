import { useState, useMemo, useEffect, useRef, useCallback } from "react";

// ── Types ──────────────────────────────────────────────────
interface ScoringBreakdown {
  pattern_confidence: number; feature_score: number; strategy_score: number;
  regime_alignment: number; backtest_edge: number; volume_confirm: number;
}
interface AIVerdict {
  verdict: string; confidence: number; reasoning: string;
  news_sentiment: string; key_factors: string[]; processing_time: number;
}
interface InPlayStock {
  symbol: string; reason: string; catalyst: string;
  expected_direction: string; priority: number;
}
interface Setup {
  pattern_name: string; category: string; symbol: string; bias: string;
  timeframe_detected: string; multi_tf: boolean;
  entry_price: number; stop_loss: number; target_price: number;
  risk_reward_ratio: number; confidence: number; composite_score: number;
  scoring: ScoringBreakdown; detected_at: string; description: string;
  win_rate: number; key_levels: Record<string, number>;
  regime: string; ai_verdict?: AIVerdict; in_play_info?: InPlayStock;
}
interface RegimeData { regime: string; atr_ratio: number; }
interface HotStrategy {
  name: string; strategy_type: string; win_rate: number;
  profit_factor: number; expectancy: number; hot_score: number;
}

const API = "http://localhost:8000";

// ── Theme System ───────────────────────────────────────────
const DARK = {
  bg: "#080b12", bgCard: "#0f1520", bgHover: "#141c2b",
  border: "#1a2332", borderLight: "#253045",
  text: "#e2e8f0", textDim: "#64748b", textMuted: "#475569",
  accent: "#3b82f6", long: "#10b981", longBg: "rgba(16,185,129,0.12)",
  short: "#ef4444", shortBg: "rgba(239,68,68,0.12)",
  gold: "#f59e0b", goldBg: "rgba(245,158,11,0.12)",
  chartBg: "#080b12", chartGrid: "#111827", chartText: "#64748b",
};
const LIGHT = {
  bg: "#f8f9fc", bgCard: "#ffffff", bgHover: "#f1f5f9",
  border: "#e2e8f0", borderLight: "#cbd5e1",
  text: "#1e293b", textDim: "#64748b", textMuted: "#94a3b8",
  accent: "#3b82f6", long: "#16a34a", longBg: "rgba(22,163,74,0.08)",
  short: "#dc2626", shortBg: "rgba(220,38,38,0.08)",
  gold: "#d97706", goldBg: "rgba(217,119,6,0.08)",
  chartBg: "#ffffff", chartGrid: "#f1f5f9", chartText: "#64748b",
};
type Theme = typeof DARK;

// ── Bean Logo ──────────────────────────────────────────────
function BeanLogo({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="none">
      <ellipse cx="50" cy="54" rx="36" ry="40" fill="#8B5E3C" />
      <ellipse cx="50" cy="52" rx="33" ry="37" fill="#A0714F" />
      <ellipse cx="50" cy="50" rx="28" ry="32" fill="#B8865A" opacity="0.3" />
      <path d="M50 20C50 20 41 42 41 54C41 66 50 88 50 88" stroke="#6B4226" strokeWidth="2.5" strokeLinecap="round" opacity="0.4" />
      {/* Eyes */}
      <ellipse cx="38" cy="43" rx="4.5" ry="5" fill="#2D1B0E" />
      <ellipse cx="62" cy="43" rx="4.5" ry="5" fill="#2D1B0E" />
      <circle cx="36" cy="41" r="1.8" fill="white" />
      <circle cx="60" cy="41" r="1.8" fill="white" />
      <circle cx="40" cy="44" r="0.8" fill="white" opacity="0.5" />
      <circle cx="64" cy="44" r="0.8" fill="white" opacity="0.5" />
      {/* Blush */}
      <ellipse cx="30" cy="52" rx="5" ry="3" fill="#E8A0A0" opacity="0.35" />
      <ellipse cx="70" cy="52" rx="5" ry="3" fill="#E8A0A0" opacity="0.35" />
      {/* Smile */}
      <path d="M42 57Q50 66 58 57" stroke="#2D1B0E" strokeWidth="2.5" strokeLinecap="round" fill="none" />
      {/* Tiny arms */}
      <path d="M17 48C14 44 12 38 16 36" stroke="#A0714F" strokeWidth="3" strokeLinecap="round" />
      <path d="M83 48C86 44 88 38 84 36" stroke="#A0714F" strokeWidth="3" strokeLinecap="round" />
      {/* Trading antennas */}
      <path d="M36 16L32 8" stroke="#10b981" strokeWidth="2.5" strokeLinecap="round" />
      <circle cx="32" cy="7" r="2.5" fill="#10b981" />
      <path d="M64 16L68 8" stroke="#ef4444" strokeWidth="2.5" strokeLinecap="round" />
      <circle cx="68" cy="7" r="2.5" fill="#ef4444" />
      {/* Tiny chart on belly */}
      <path d="M41 64L45 60L49 63L53 58L57 61" stroke="#3b82f6" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.6" />
    </svg>
  );
}

// ── Verdict Badge ──────────────────────────────────────────
function VerdictBadge({ verdict, t }: { verdict?: AIVerdict; t: Theme }) {
  if (!verdict || verdict.verdict === "PENDING") return null;
  const cfg: Record<string, { bg: string; color: string; icon: string }> = {
    CONFIRMED: { bg: `${t.long}20`, color: t.long, icon: "✓" },
    CAUTION: { bg: `${t.gold}20`, color: t.gold, icon: "!" },
    DENIED: { bg: `${t.short}20`, color: t.short, icon: "✗" },
  };
  const c = cfg[verdict.verdict] || cfg.CAUTION;
  return (
    <span title={verdict.reasoning} style={{
      fontSize: 9, fontWeight: 800, padding: "2px 7px", borderRadius: 4,
      background: c.bg, color: c.color, border: `1px solid ${c.color}30`,
      letterSpacing: 0.3, cursor: "help",
    }}>{c.icon} {verdict.verdict}</span>
  );
}

// ── Regime Pill ────────────────────────────────────────────
const REGIME_CFG: Record<string, { label: string; icon: string; colorKey: keyof Theme }> = {
  trending_bull: { label: "BULL", icon: "▲", colorKey: "long" },
  trending_bear: { label: "BEAR", icon: "▼", colorKey: "short" },
  high_volatility: { label: "HI-VOL", icon: "◆", colorKey: "gold" },
  mean_reverting: { label: "RANGE", icon: "◎", colorKey: "accent" },
};
function RegimePill({ regime, t }: { regime: RegimeData | null; t: Theme }) {
  if (!regime) return null;
  const cfg = REGIME_CFG[regime.regime];
  if (!cfg) return null;
  const color = t[cfg.colorKey] as string;
  return (
    <span style={{
      fontSize: 9, fontWeight: 700, padding: "2px 8px", borderRadius: 5,
      background: `${color}15`, color, border: `1px solid ${color}30`,
      letterSpacing: 0.4,
    }}>{cfg.icon} {cfg.label}</span>
  );
}

// ── Category + TF Badges ───────────────────────────────────
const CAT_CLR: Record<string, string> = { classical: "#3b82f6", candlestick: "#f59e0b", smb_scalp: "#a855f7", quant: "#10b981" };
function CatBadge({ category }: { category: string }) {
  const color = CAT_CLR[category] || "#888";
  const label = { classical: "Classical", candlestick: "Candle", smb_scalp: "SMB", quant: "Quant" }[category] || category;
  return <span style={{ fontSize: 8, fontWeight: 700, padding: "1px 5px", borderRadius: 3, background: `${color}18`, color, textTransform: "uppercase", letterSpacing: 0.3 }}>{label}</span>;
}
function TfBadge({ tf, t }: { tf: string; t: Theme }) {
  const multi = tf.includes("&");
  return <span style={{ fontSize: 8, fontWeight: 700, padding: "1px 5px", borderRadius: 3, background: multi ? `${t.accent}20` : `${t.textDim}15`, color: multi ? t.accent : t.textDim }}>{multi ? "★ " : ""}{tf}</span>;
}

// ── Score Cell ─────────────────────────────────────────────
function ScoreCell({ score, t }: { score: number; t: Theme }) {
  const color = score >= 65 ? t.long : score >= 45 ? t.gold : t.short;
  return (
    <span style={{
      fontFamily: "'JetBrains Mono', monospace", fontSize: 12, fontWeight: 800,
      color, padding: "2px 5px", borderRadius: 4, background: `${color}15`,
      display: "inline-block", minWidth: 34, textAlign: "center",
    }}>{score.toFixed(0)}</span>
  );
}

// ── Trade Timing ───────────────────────────────────────────
function TradeTiming({ setup, t }: { setup: Setup; t: Theme }) {
  const detected = new Date(setup.detected_at);
  const isToday = new Date().toDateString() === detected.toDateString();
  const timeStr = detected.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const dateStr = isToday ? "Today" : detected.toLocaleDateString([], { month: "short", day: "numeric" });
  const isLong = setup.bias === "long";
  const currentRelEntry = isLong
    ? `Buy above $${setup.entry_price.toFixed(2)}`
    : `Short below $${setup.entry_price.toFixed(2)}`;

  return (
    <div style={{ fontSize: 10, color: t.textDim, display: "flex", gap: 8, alignItems: "center", marginTop: 1 }}>
      <span>{dateStr} {timeStr}</span>
      <span style={{ color: t.accent, fontWeight: 600 }}>{currentRelEntry}</span>
      <span style={{ color: t.textMuted }}>→ Target ${setup.target_price.toFixed(2)}</span>
    </div>
  );
}

// ── Scoring Breakdown ──────────────────────────────────────
function ScoringPanel({ scoring, verdict, t }: { scoring: ScoringBreakdown; verdict?: AIVerdict; t: Theme }) {
  const factors = [
    { label: "Pattern", value: scoring.pattern_confidence, weight: "20%" },
    { label: "Features", value: scoring.feature_score, weight: "25%" },
    { label: "Strategy", value: scoring.strategy_score, weight: "20%" },
    { label: "Regime", value: scoring.regime_alignment, weight: "15%" },
    { label: "Backtest", value: scoring.backtest_edge, weight: "10%" },
    { label: "Volume", value: scoring.volume_confirm, weight: "10%" },
  ];
  return (
    <div style={{ padding: "10px 14px", background: t.bgCard, borderRadius: 8, border: `1px solid ${t.border}`, marginTop: 8 }}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: verdict?.reasoning ? 10 : 0 }}>
        {factors.map(f => {
          const c = f.value >= 65 ? t.long : f.value >= 40 ? t.gold : t.short;
          return (
            <div key={f.label} style={{ textAlign: "center", minWidth: 52 }}>
              <div style={{ fontSize: 8, color: t.textMuted }}>{f.label} ({f.weight})</div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, fontWeight: 700, color: c }}>{f.value.toFixed(0)}</div>
            </div>
          );
        })}
      </div>
      {verdict && verdict.reasoning && verdict.verdict !== "PENDING" && (
        <div style={{ fontSize: 11, color: t.textDim, borderTop: `1px solid ${t.border}`, paddingTop: 8, marginTop: 4 }}>
          <span style={{ fontWeight: 700, color: t.text }}>AI: </span>{verdict.reasoning}
          {verdict.key_factors?.length > 0 && (
            <div style={{ marginTop: 4, display: "flex", gap: 4 }}>
              {verdict.key_factors.map((f, i) => (
                <span key={i} style={{ fontSize: 8, padding: "1px 5px", borderRadius: 3, background: `${t.accent}12`, color: t.accent }}>{f}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Chart (Lightweight Charts v4 compatible) ───────────────
function TradeChart({ setup, onClose, t }: { setup: Setup; onClose: () => void; t: Theme }) {
  const ref = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!ref.current) return;
    let chart: any = null;
    const load = async () => {
      try {
        const lc = await import("lightweight-charts");
        const tf = setup.timeframe_detected.includes("15m") ? "15min" : "5min";
        const res = await fetch(`${API}/api/chart/${setup.symbol}?timeframe=${tf}&days_back=5`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        if (!data.bars?.length) throw new Error("No data");
        if (ref.current) ref.current.innerHTML = "";

        chart = lc.createChart(ref.current!, {
          width: ref.current!.clientWidth, height: 360,
          layout: { background: { color: t.chartBg } as any, textColor: t.chartText, fontFamily: "JetBrains Mono, monospace" },
          grid: { vertLines: { color: t.chartGrid }, horzLines: { color: t.chartGrid } },
          crosshair: { mode: lc.CrosshairMode.Normal },
          rightPriceScale: { borderColor: t.border }, timeScale: { borderColor: t.border, timeVisible: true },
        });

        // v4 compatible: try new API first, fallback to v3
        let series: any;
        if ((lc as any).CandlestickSeries) {
          series = chart.addSeries((lc as any).CandlestickSeries, {
            upColor: t.long, downColor: t.short, borderUpColor: t.long, borderDownColor: t.short,
            wickUpColor: t.long, wickDownColor: t.short,
          });
        } else if (chart.addCandlestickSeries) {
          series = chart.addCandlestickSeries({
            upColor: t.long, downColor: t.short, borderUpColor: t.long, borderDownColor: t.short,
            wickUpColor: t.long, wickDownColor: t.short,
          });
        } else {
          throw new Error("Unsupported lightweight-charts version");
        }

        series.setData(data.bars);

        const addLine = (price: number, color: string, title: string) => {
          series.createPriceLine({ price, color, lineWidth: 2, lineStyle: lc.LineStyle.Dashed, axisLabelVisible: true, title });
        };
        addLine(setup.entry_price, t.accent, "ENTRY");
        addLine(setup.stop_loss, t.short, "STOP");
        addLine(setup.target_price, t.long, "TARGET");

        Object.entries(setup.key_levels || {}).forEach(([n, p]) => {
          if (typeof p === "number" && p > 0) series.createPriceLine({ price: p, color: t.textMuted, lineWidth: 1, lineStyle: lc.LineStyle.Dotted, axisLabelVisible: false, title: n });
        });

        chart.timeScale().fitContent();
        setLoading(false);
        const ro = new ResizeObserver(() => { if (ref.current && chart) chart.applyOptions({ width: ref.current.clientWidth }); });
        ro.observe(ref.current!);
        return () => ro.disconnect();
      } catch (e: any) { setError(e.message || "Chart error"); setLoading(false); }
    };
    load();
    return () => { if (chart) chart.remove(); };
  }, [setup, t]);

  return (
    <div style={{ background: t.bg, border: `1px solid ${t.border}`, borderRadius: 10, padding: 14, margin: "6px 0" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: t.text }}>{setup.symbol} — {setup.pattern_name} ({setup.timeframe_detected})</span>
        <div style={{ display: "flex", gap: 10, fontSize: 10, fontWeight: 700, alignItems: "center" }}>
          <span style={{ color: t.accent }}>Entry ${setup.entry_price.toFixed(2)}</span>
          <span style={{ color: t.short }}>Stop ${setup.stop_loss.toFixed(2)}</span>
          <span style={{ color: t.long }}>Target ${setup.target_price.toFixed(2)}</span>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: t.textDim, fontSize: 14 }}>✕</button>
        </div>
      </div>
      <ScoringPanel scoring={setup.scoring} verdict={setup.ai_verdict} t={t} />
      {loading && <div style={{ textAlign: "center", padding: 30, color: t.textDim }}>Loading chart...</div>}
      {error && <div style={{ textAlign: "center", padding: 14, color: t.short, fontSize: 11 }}>Chart: {error}</div>}
      <div ref={ref} style={{ width: "100%", minHeight: loading ? 0 : 360, marginTop: 8 }} />
    </div>
  );
}

// ── Setup Row ──────────────────────────────────────────────
function SetupRow({ s, open, toggle, t }: { s: Setup; open: boolean; toggle: () => void; t: Theme }) {
  const isLong = s.bias === "long";
  return (
    <div>
      <div onClick={toggle} style={{
        display: "flex", alignItems: "center", gap: 8, padding: "9px 10px",
        borderBottom: `1px solid ${t.border}`, cursor: "pointer",
        background: open ? t.bgHover : "transparent", transition: "background 0.15s",
      }}>
        <div style={{ width: 58 }}>
          <div style={{ fontSize: 13, fontWeight: 800, color: t.text }}>{s.symbol}</div>
          <span style={{ fontSize: 9, fontWeight: 700, padding: "1px 5px", borderRadius: 3, background: isLong ? t.longBg : t.shortBg, color: isLong ? t.long : t.short }}>{s.bias.toUpperCase()}</span>
        </div>
        <div style={{ flex: 1, minWidth: 120 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap" }}>
            <span style={{ fontSize: 12, fontWeight: 700, color: t.text }}>{s.pattern_name}</span>
            <CatBadge category={s.category} /> <TfBadge tf={s.timeframe_detected} t={t} />
            <VerdictBadge verdict={s.ai_verdict} t={t} />
          </div>
          <TradeTiming setup={s} t={t} />
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {[{ l: "ENTRY", v: s.entry_price, c: t.text }, { l: "STOP", v: s.stop_loss, c: t.short }, { l: "TGT", v: s.target_price, c: t.long }].map(({ l, v, c }) => (
            <div key={l} style={{ textAlign: "right", minWidth: 48 }}>
              <div style={{ fontSize: 8, color: t.textMuted, fontWeight: 600, letterSpacing: 0.4 }}>{l}</div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, fontWeight: 700, color: c }}>${v.toFixed(2)}</div>
            </div>
          ))}
        </div>
        <div style={{ width: 32, textAlign: "center" }}>
          <div style={{ fontSize: 8, color: t.textMuted }}>R:R</div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, fontWeight: 800, color: s.risk_reward_ratio >= 2 ? t.long : t.text }}>{s.risk_reward_ratio.toFixed(1)}</div>
        </div>
        <ScoreCell score={s.composite_score} t={t} />
        <div style={{ width: 12, textAlign: "center", fontSize: 10, color: t.textDim }}>{open ? "▴" : "▾"}</div>
      </div>
      {open && <TradeChart setup={s} onClose={toggle} t={t} />}
    </div>
  );
}

// ── In-Play Card ───────────────────────────────────────────
function InPlayCard({ stock, onClick, t }: { stock: InPlayStock; onClick: () => void; t: Theme }) {
  const dc = stock.expected_direction === "bullish" ? t.long : stock.expected_direction === "bearish" ? t.short : t.gold;
  return (
    <div onClick={onClick} style={{
      padding: "9px 12px", background: t.bgCard, borderRadius: 8, border: `1px solid ${t.border}`,
      cursor: "pointer", minWidth: 170, flex: "0 0 auto", transition: "border-color 0.15s",
    }} onMouseEnter={e => (e.currentTarget.style.borderColor = t.accent)} onMouseLeave={e => (e.currentTarget.style.borderColor = t.border)}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
        <span style={{ fontSize: 14, fontWeight: 800, color: t.text }}>{stock.symbol}</span>
        <span style={{ fontSize: 8, fontWeight: 700, padding: "1px 5px", borderRadius: 3, background: `${dc}18`, color: dc, textTransform: "uppercase" }}>{stock.expected_direction}</span>
      </div>
      <div style={{ fontSize: 10, color: t.textDim, lineHeight: 1.4 }}>{stock.reason}</div>
    </div>
  );
}

// ── Pill ───────────────────────────────────────────────────
function Pill({ active, children, onClick, t, small }: { active: boolean; children: React.ReactNode; onClick: () => void; t: Theme; small?: boolean }) {
  return (
    <button onClick={onClick} style={{
      fontSize: small ? 9 : 10, fontWeight: active ? 700 : 500,
      padding: small ? "3px 7px" : "4px 10px", borderRadius: 5,
      cursor: "pointer", border: "none",
      background: active ? t.accent : "transparent",
      color: active ? "#fff" : t.textDim,
      fontFamily: "'Outfit', sans-serif", transition: "all 0.15s",
    }}>{children}</button>
  );
}

// ── Main App ───────────────────────────────────────────────
function App() {
  const [dark, setDark] = useState(true);
  const t = dark ? DARK : LIGHT;

  const [view, setView] = useState<"opportunities" | "scan">("opportunities");
  const [symbol, setSymbol] = useState("AAPL");
  const [scanSetups, setScanSetups] = useState<Setup[]>([]);
  const [topSetups, setTopSetups] = useState<Setup[]>([]);
  const [inPlay, setInPlay] = useState<InPlayStock[]>([]);
  const [marketSummary, setMarketSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const [topLoading, setTopLoading] = useState(true);
  const [error, setError] = useState("");
  const [chartIdx, setChartIdx] = useState<number | null>(null);
  const [filterBias, setFilterBias] = useState("ALL");
  const [filterCat, setFilterCat] = useState("ALL");
  const [sortBy, setSortBy] = useState<"score" | "rr">("score");
  const [mode, setMode] = useState<"today" | "active">("today");
  const [regime, setRegime] = useState<RegimeData | null>(null);
  const [hotStrategies, setHotStrategies] = useState<HotStrategy[]>([]);
  const [patternCount, setPatternCount] = useState(47);

  useEffect(() => {
    fetch(`${API}/api/health`).then(r => r.json()).then(d => { if (d.patterns) setPatternCount(d.patterns); }).catch(() => {});
    fetch(`${API}/api/regime`).then(r => r.json()).then(d => { if (d.regime) setRegime(d); }).catch(() => {});
    fetch(`${API}/api/hot-strategies?top_n=5`).then(r => r.json()).then(d => { if (d.strategies) setHotStrategies(d.strategies); }).catch(() => {});
    setTopLoading(true);
    fetch(`${API}/api/top-opportunities`)
      .then(r => r.json())
      .then(d => {
        if (d.setups) setTopSetups(d.setups);
        if (d.in_play?.stocks) setInPlay(d.in_play.stocks);
        if (d.market_summary) setMarketSummary(d.market_summary);
      })
      .catch(e => console.error("Top opportunities:", e))
      .finally(() => setTopLoading(false));
  }, []);

  const handleScan = useCallback(async () => {
    setLoading(true); setError(""); setScanSetups([]); setChartIdx(null);
    try {
      const res = await fetch(`${API}/api/scan?symbol=${symbol}&mode=${mode}&ai=true`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setScanSetups(data.setups);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  }, [symbol, mode]);

  const handleInPlayClick = (stock: InPlayStock) => {
    setSymbol(stock.symbol); setView("scan");
    setTimeout(() => {
      setLoading(true); setError(""); setScanSetups([]); setChartIdx(null);
      fetch(`${API}/api/scan?symbol=${stock.symbol}&mode=active&ai=true`)
        .then(r => r.json()).then(d => { if (!d.error) setScanSetups(d.setups); })
        .catch(e => setError(e.message)).finally(() => setLoading(false));
    }, 50);
  };

  const activeSetups = view === "opportunities" ? topSetups : scanSetups;
  const filtered = useMemo(() => {
    let r = activeSetups;
    if (filterBias !== "ALL") r = r.filter(s => s.bias === filterBias.toLowerCase());
    if (filterCat !== "ALL") r = r.filter(s => s.category === filterCat);
    return [...r].sort((a, b) => sortBy === "rr" ? b.risk_reward_ratio - a.risk_reward_ratio : (b.composite_score || 0) - (a.composite_score || 0));
  }, [activeSetups, filterBias, filterCat, sortBy]);

  const longs = activeSetups.filter(s => s.bias === "long").length;
  const shorts = activeSetups.filter(s => s.bias === "short").length;

  return (
    <div style={{ background: t.bg, minHeight: "100vh", fontFamily: "'Outfit', sans-serif", color: t.text, transition: "background 0.3s, color 0.3s" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Outfit:wght@400;500;600;700;800;900&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${t.border}; border-radius: 3px; }
        input:focus, button:focus { outline: none; }
      `}</style>

      {/* Header */}
      <div style={{
        padding: "8px 24px", borderBottom: `1px solid ${t.border}`,
        display: "flex", justifyContent: "space-between", alignItems: "center",
        background: t.bgCard,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <BeanLogo size={30} />
          <span style={{ fontSize: 18, fontWeight: 900, letterSpacing: -0.5 }}>AlphaBean</span>
          <span style={{ fontSize: 8, color: t.textMuted, fontWeight: 600 }}>v3.2</span>
          <div style={{ display: "flex", gap: 2, marginLeft: 10, background: t.border, borderRadius: 5, padding: 2 }}>
            <Pill active={view === "opportunities"} onClick={() => setView("opportunities")} t={t}>Opportunities</Pill>
            <Pill active={view === "scan"} onClick={() => setView("scan")} t={t}>Scan</Pill>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 10, fontWeight: 600 }}>
          <RegimePill regime={regime} t={t} />
          {activeSetups.length > 0 && <>
            <span style={{ color: t.long }}>{longs}L</span>
            <span style={{ color: t.short }}>{shorts}S</span>
          </>}
          {/* Theme Toggle */}
          <button onClick={() => setDark(!dark)} style={{
            background: t.border, border: "none", borderRadius: 5, padding: "3px 8px",
            cursor: "pointer", fontSize: 12, color: t.text, display: "flex", alignItems: "center", gap: 3,
          }}>
            {dark ? "☀️" : "🌙"}
          </button>
        </div>
      </div>

      <div style={{ maxWidth: 1060, margin: "0 auto", padding: "14px 24px" }}>

        {/* ═══ OPPORTUNITIES ═══ */}
        {view === "opportunities" && <>
          {marketSummary && (
            <div style={{ padding: "9px 14px", background: t.bgCard, borderRadius: 8, border: `1px solid ${t.border}`, marginBottom: 12, fontSize: 11, color: t.textDim }}>
              <span style={{ fontWeight: 700, color: t.text }}>Market: </span>{marketSummary}
            </div>
          )}
          {inPlay.length > 0 && (
            <div style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: t.textMuted, marginBottom: 6, letterSpacing: 0.4 }}>IN-PLAY TODAY</div>
              <div style={{ display: "flex", gap: 6, overflowX: "auto", paddingBottom: 4 }}>
                {inPlay.map(s => <InPlayCard key={s.symbol} stock={s} onClick={() => handleInPlayClick(s)} t={t} />)}
              </div>
            </div>
          )}
          {hotStrategies.length > 0 && (
            <div style={{ display: "flex", gap: 5, marginBottom: 12, flexWrap: "wrap", alignItems: "center" }}>
              <span style={{ fontSize: 9, fontWeight: 700, color: t.textMuted }}>🔥 HOT:</span>
              {hotStrategies.map(s => (
                <span key={s.name} style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, background: t.bgCard, border: `1px solid ${t.border}`, color: t.textDim }}>
                  <span style={{ fontWeight: 700, color: t.text }}>{s.name}</span>
                  <span style={{ marginLeft: 3, color: t.long }}>{(s.win_rate*100).toFixed(0)}%</span>
                </span>
              ))}
            </div>
          )}
          {topLoading && (
            <div style={{ textAlign: "center", padding: 50, color: t.textDim }}>
              <BeanLogo size={44} />
              <div style={{ fontSize: 12, fontWeight: 700, marginTop: 14 }}>Finding today's opportunities...</div>
              <div style={{ fontSize: 10, marginTop: 4, color: t.textMuted }}>Scraping news → Ollama extraction → {patternCount} patterns → AI evaluation</div>
            </div>
          )}
        </>}

        {/* ═══ SCAN ═══ */}
        {view === "scan" && (
          <div style={{ display: "flex", gap: 8, alignItems: "end", marginBottom: 14, flexWrap: "wrap" }}>
            <div>
              <label style={{ fontSize: 8, color: t.textMuted, display: "block", marginBottom: 2, fontWeight: 700, letterSpacing: 0.4 }}>SYMBOL</label>
              <input type="text" value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())}
                onKeyDown={e => e.key === "Enter" && handleScan()} placeholder="AAPL"
                style={{
                  fontSize: 13, padding: "6px 10px", borderRadius: 6, width: 90,
                  border: `1.5px solid ${t.border}`, fontWeight: 800,
                  background: t.bgCard, color: t.text, fontFamily: "'Outfit', sans-serif",
                }}
              />
            </div>
            <div style={{ display: "flex", gap: 2, background: t.border, borderRadius: 5, padding: 2 }}>
              <Pill active={mode === "today"} onClick={() => setMode("today")} t={t} small>Today</Pill>
              <Pill active={mode === "active"} onClick={() => setMode("active")} t={t} small>Active</Pill>
            </div>
            <button onClick={handleScan} disabled={loading || !symbol} style={{
              fontSize: 11, fontWeight: 700, padding: "7px 18px", borderRadius: 6, border: "none",
              background: loading ? t.border : t.accent, color: "#fff",
              cursor: loading ? "wait" : "pointer", fontFamily: "'Outfit', sans-serif",
            }}>{loading ? "Scanning..." : "Scan"}</button>
            <span style={{ fontSize: 9, color: t.textMuted }}>5m+15m • {patternCount} patterns • AI</span>
          </div>
        )}

        {error && <div style={{ padding: "8px 12px", borderRadius: 6, background: t.shortBg, color: t.short, fontSize: 11, marginBottom: 12, border: `1px solid ${t.short}30` }}>{error}</div>}
        {loading && view === "scan" && <div style={{ textAlign: "center", padding: 40, color: t.textDim }}><BeanLogo size={36} /><div style={{ fontSize: 12, fontWeight: 700, marginTop: 10 }}>Scanning {symbol}...</div></div>}

        {/* ═══ RESULTS ═══ */}
        {!loading && activeSetups.length > 0 && !(view === "opportunities" && topLoading) && <>
          <div style={{ display: "flex", gap: 5, marginBottom: 10, flexWrap: "wrap", alignItems: "center" }}>
            <div style={{ display: "flex", gap: 2, background: t.border, borderRadius: 5, padding: 2 }}>
              {["ALL", "LONG", "SHORT"].map(b => <Pill key={b} active={filterBias === b} onClick={() => setFilterBias(b)} t={t} small>{b}</Pill>)}
            </div>
            <div style={{ display: "flex", gap: 2, background: t.border, borderRadius: 5, padding: 2 }}>
              {[["ALL","All"],["classical","Classical"],["candlestick","Candle"],["smb_scalp","SMB"],["quant","Quant"]].map(([v,l]) => (
                <Pill key={v} active={filterCat === v} onClick={() => setFilterCat(v)} t={t} small>{l}</Pill>
              ))}
            </div>
            <div style={{ marginLeft: "auto", display: "flex", gap: 2, background: t.border, borderRadius: 5, padding: 2 }}>
              {[["score","Score"],["rr","R:R"]].map(([k,l]) => <Pill key={k} active={sortBy === k} onClick={() => setSortBy(k as any)} t={t} small>{l}</Pill>)}
            </div>
          </div>

          <div style={{
            display: "flex", alignItems: "center", gap: 8, padding: "5px 10px",
            borderBottom: `1px solid ${t.borderLight}`, fontSize: 8, fontWeight: 700,
            color: t.textMuted, letterSpacing: 0.6, textTransform: "uppercase",
          }}>
            <div style={{ width: 58 }}>Ticker</div>
            <div style={{ flex: 1 }}>Setup / Trigger / AI</div>
            <div style={{ display: "flex", gap: 8 }}>
              <div style={{ width: 48, textAlign: "right" }}>Entry</div>
              <div style={{ width: 48, textAlign: "right" }}>Stop</div>
              <div style={{ width: 48, textAlign: "right" }}>Target</div>
            </div>
            <div style={{ width: 32, textAlign: "center" }}>R:R</div>
            <div style={{ width: 38, textAlign: "center" }}>Score</div>
            <div style={{ width: 12 }} />
          </div>
          {filtered.map((s, i) => <SetupRow key={`${s.symbol}-${s.pattern_name}-${i}`} s={s} open={chartIdx === i} toggle={() => setChartIdx(chartIdx === i ? null : i)} t={t} />)}
          {filtered.length === 0 && <div style={{ textAlign: "center", padding: 24, color: t.textDim, fontSize: 11 }}>No setups match filters.</div>}
        </>}

        {!loading && view === "scan" && scanSetups.length === 0 && !error && (
          <div style={{ textAlign: "center", padding: 40, color: t.textDim }}>
            <BeanLogo size={40} />
            <div style={{ fontSize: 14, fontWeight: 700, marginTop: 10, color: t.text }}>Scan a ticker</div>
            <div style={{ fontSize: 10, marginTop: 4, maxWidth: 380, margin: "4px auto 0" }}>
              {patternCount} patterns • 6-factor scoring • Ollama AI evaluation
            </div>
          </div>
        )}

        <div style={{ textAlign: "center", marginTop: 30, padding: "12px 0", borderTop: `1px solid ${t.border}` }}>
          <span style={{ fontSize: 9, color: t.textMuted }}>AlphaBean v3.2 — {patternCount} Detectors — Ollama AI — All Local, All Free</span>
        </div>
      </div>
    </div>
  );
}

export default App;