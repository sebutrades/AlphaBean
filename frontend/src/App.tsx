import { useState, useMemo, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence, LayoutGroup } from "framer-motion";
const API = "http://localhost:8000";

// ═══════════════════════════════════════════════
// THEME
// ═══════════════════════════════════════════════
const DARK = {
  bg:"#08090d", bgCard:"#12141c", bgHover:"#1a1d28", bgModal:"#0e1018",
  border:"#1f2233", borderLight:"#2a2e42",
  text:"#eef0f6", textDim:"#8b90a8", textMuted:"#5c6080",
  accent:"#6c5ce7", accentLight:"#a29bfe",
  long:"#00d2a0", longBg:"rgba(0,210,160,0.1)",
  short:"#ff6b6b", shortBg:"rgba(255,107,107,0.1)",
  gold:"#fdcb6e", goldBg:"rgba(253,203,110,0.1)",
  purple:"#a29bfe",
  chartBg:"#08090d", chartGrid:"#1a1d28", chartText:"#5c6080",
  glow:"rgba(108,92,231,0.15)",
};
const LIGHT = {
  bg:"#f7f8fc", bgCard:"#ffffff", bgHover:"#f0f1f8", bgModal:"#f0f1f5",
  border:"#e2e4ef", borderLight:"#cdd0e0",
  text:"#1a1c2e", textDim:"#6b6f8a", textMuted:"#9396ab",
  accent:"#6c5ce7", accentLight:"#a29bfe",
  long:"#00b894", longBg:"rgba(0,184,148,0.08)",
  short:"#e74c3c", shortBg:"rgba(231,76,60,0.08)",
  gold:"#f39c12", goldBg:"rgba(243,156,18,0.08)",
  purple:"#6c5ce7",
  chartBg:"#ffffff", chartGrid:"#f0f1f8", chartText:"#9396ab",
  glow:"rgba(108,92,231,0.08)",
};
type T = typeof DARK;

// ═══════════════════════════════════════════════
// FRAMER MOTION VARIANTS
// ═══════════════════════════════════════════════
const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" } },
  exit: { opacity: 0, y: -8, transition: { duration: 0.2 } },
};
const stagger = {
  visible: { transition: { staggerChildren: 0.06 } },
};
const scaleIn = {
  hidden: { opacity: 0, scale: 0.95 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.3, ease: "easeOut" } },
};
const slideIn = {
  hidden: { opacity: 0, x: -20 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.35, ease: "easeOut" } },
};
const modalBg = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.2 } },
  exit: { opacity: 0, transition: { duration: 0.15 } },
};
const modalContent = {
  hidden: { opacity: 0, scale: 0.92, y: 20 },
  visible: { opacity: 1, scale: 1, y: 0, transition: { duration: 0.3, ease: [0.16, 1, 0.3, 1] } },
  exit: { opacity: 0, scale: 0.95, y: 10, transition: { duration: 0.2 } },
};

// ═══════════════════════════════════════════════
// LOGO — PNG placeholder (user will provide)
// Place your generated logo at: frontend/public/juicer-logo.png
// ═══════════════════════════════════════════════
function JuicerLogo({ size = 36 }: { size?: number }) {
  return (
    <motion.img
      src="/juicer-logo.png"
      alt="Juicer"
      width={size}
      height={size}
      style={{ borderRadius: 8, objectFit: "contain" }}
      whileHover={{ scale: 1.1, rotate: 3 }}
      whileTap={{ scale: 0.95 }}
      // Fallback if image not found yet — shows a colored box
      onError={(e: any) => {
        e.target.style.display = "none";
        e.target.nextSibling.style.display = "flex";
      }}
    />
  );
}
// Fallback SVG for when PNG isn't placed yet
function LogoFallback({ size = 36 }: { size?: number }) {
  return (
    <div style={{
      width: size, height: size, borderRadius: 8, display: "none",
      background: "linear-gradient(135deg, #6c5ce7, #a29bfe)",
      alignItems: "center", justifyContent: "center", fontSize: size * 0.5,
    }}>🧃</div>
  );
}
function Logo({ size = 36 }: { size?: number }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center" }}>
      <JuicerLogo size={size} />
      <LogoFallback size={size} />
    </span>
  );
}

// ═══════════════════════════════════════════════
// LOADING — Framer Motion smooth loader
// ═══════════════════════════════════════════════
function JuicerLoader({ t, msg = "Loading..." }: { t: T; msg?: string }) {
  return (
    <motion.div
      initial="hidden" animate="visible" variants={fadeUp}
      style={{ textAlign: "center", padding: "60px 20px" }}
    >
      {/* Animated juice drops */}
      <div style={{ display: "flex", justifyContent: "center", gap: 12, marginBottom: 24 }}>
        {[0, 1, 2, 3, 4].map(i => (
          <motion.div
            key={i}
            animate={{
              y: [0, -20, 0],
              scale: [1, 1.3, 1],
              opacity: [0.4, 1, 0.4],
            }}
            transition={{
              duration: 1.2,
              repeat: Infinity,
              delay: i * 0.15,
              ease: "easeInOut",
            }}
            style={{
              width: 14, height: 14, borderRadius: "50%",
              background: [t.accent, t.long, t.gold, t.accentLight, t.long][i],
              boxShadow: `0 0 12px ${[t.accent, t.long, t.gold, t.accentLight, t.long][i]}60`,
            }}
          />
        ))}
      </div>

      {/* Progress bar */}
      <div style={{
        width: 200, height: 4, borderRadius: 2, background: t.border,
        margin: "0 auto 20px", overflow: "hidden",
      }}>
        <motion.div
          animate={{ x: ["-100%", "200%"] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
          style={{
            width: "40%", height: "100%", borderRadius: 2,
            background: `linear-gradient(90deg, transparent, ${t.accent}, transparent)`,
          }}
        />
      </div>

      <motion.div
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 2, repeat: Infinity }}
        style={{ fontSize: 16, fontWeight: 600, color: t.text }}
      >
        {msg}
      </motion.div>
      <div style={{ fontSize: 12, color: t.textMuted, marginTop: 6 }}>
        Squeezing the market for opportunities...
      </div>
    </motion.div>
  );
}

// ═══════════════════════════════════════════════
// SMALL COMPONENTS
// ═══════════════════════════════════════════════
function Pill({ active, children, onClick, t, s }: { active: boolean; children: React.ReactNode; onClick: () => void; t: T; s?: boolean }) {
  return (
    <motion.button
      whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.97 }}
      onClick={onClick}
      style={{
        fontSize: s ? 11 : 12, fontWeight: active ? 700 : 500,
        padding: s ? "4px 10px" : "6px 14px", borderRadius: 8,
        cursor: "pointer", border: "none",
        background: active ? t.accent : "transparent",
        color: active ? "#fff" : t.textDim,
        fontFamily: "'Outfit',sans-serif", transition: "background .2s, color .2s",
      }}
    >{children}</motion.button>
  );
}

const CAT: Record<string, { c: string; l: string }> = {
  classical: { c: "#6c5ce7", l: "Classical" }, candlestick: { c: "#fdcb6e", l: "Candle" },
  smb_scalp: { c: "#a29bfe", l: "SMB" }, quant: { c: "#00d2a0", l: "Quant" },
};
function CatBadge({ cat }: { cat: string }) {
  const c = CAT[cat] || { c: "#888", l: cat };
  return <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 5, background: c.c + "18", color: c.c, textTransform: "uppercase", letterSpacing: 0.3 }}>{c.l}</span>;
}
function TfBadge({ tf, t }: { tf: string; t: T }) {
  const m = tf.includes("&");
  return <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 5, background: m ? t.accent + "20" : t.textMuted + "15", color: m ? t.accent : t.textMuted }}>{m ? "★ " : ""}{tf}</span>;
}
function VerdictBadge({ v, t }: { v: any; t: T }) {
  if (!v || v.verdict === "PENDING") return null;
  const m: Record<string, { bg: string; c: string; i: string }> = { CONFIRMED: { bg: t.long + "20", c: t.long, i: "✓" }, CAUTION: { bg: t.gold + "20", c: t.gold, i: "⚠" }, DENIED: { bg: t.short + "20", c: t.short, i: "✗" } };
  const c = m[v.verdict] || m.CAUTION;
  return <span title={v.reasoning} style={{ fontSize: 10, fontWeight: 800, padding: "2px 8px", borderRadius: 5, background: c.bg, color: c.c, border: `1px solid ${c.c}25`, cursor: "help" }}>{c.i} {v.verdict}</span>;
}
function ScoreCell({ score, t }: { score: number; t: T }) {
  const c = score >= 65 ? t.long : score >= 45 ? t.gold : t.short;
  return <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 15, fontWeight: 800, color: c, padding: "3px 8px", borderRadius: 6, background: c + "15", display: "inline-block", minWidth: 40, textAlign: "center" }}>{score.toFixed(0)}</span>;
}
function CorrLabel({ corr, t }: { corr: any; t: T }) {
  if (!corr || !corr.label || corr.label === "No data") return null;
  const color = corr.color || t.textDim;
  return (
    <span title={`${corr.symbol}: ${corr.stock_return_pct >= 0 ? "+" : ""}${corr.stock_return_pct}% vs SPY: ${corr.spy_return_pct >= 0 ? "+" : ""}${corr.spy_return_pct}%`}
      style={{ fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 5, background: color + "18", color, cursor: "help", border: `1px solid ${color}20` }}>
      {corr.label} ({corr.spread_pct >= 0 ? "+" : ""}{corr.spread_pct}%)
    </span>
  );
}

const REG: Record<string, { l: string; i: string; ck: keyof T }> = {
  trending_bull: { l: "BULL", i: "▲", ck: "long" }, trending_bear: { l: "BEAR", i: "▼", ck: "short" },
  high_volatility: { l: "HI-VOL", i: "◆", ck: "gold" }, mean_reverting: { l: "RANGE", i: "◎", ck: "accent" },
};
function RegimePill({ r, t }: { r: any; t: T }) {
  if (!r?.regime) return null; const c = REG[r.regime]; if (!c) return null;
  const co = t[c.ck] as string;
  return <span style={{ fontSize: 10, fontWeight: 700, padding: "3px 10px", borderRadius: 6, background: co + "12", color: co, border: `1px solid ${co}25` }}>{c.i} {c.l}</span>;
}

// ═══════════════════════════════════════════════
// BACKTEST MODAL
// ═══════════════════════════════════════════════
const PAT_CAT: Record<string, string> = { "Head & Shoulders": "classical", "Inverse H&S": "classical", "Double Top": "classical", "Double Bottom": "classical", "Triple Top": "classical", "Triple Bottom": "classical", "Ascending Triangle": "classical", "Descending Triangle": "classical", "Symmetrical Triangle": "classical", "Bull Flag": "classical", "Bear Flag": "classical", "Pennant": "classical", "Cup & Handle": "classical", "Rectangle": "classical", "Rising Wedge": "classical", "Falling Wedge": "classical", "Bullish Engulfing": "candlestick", "Bearish Engulfing": "candlestick", "Morning Star": "candlestick", "Evening Star": "candlestick", "Hammer": "candlestick", "Shooting Star": "candlestick", "Doji": "candlestick", "Dragonfly Doji": "candlestick", "Three White Soldiers": "candlestick", "Three Black Crows": "candlestick", "RubberBand Scalp": "smb_scalp", "HitchHiker Scalp": "smb_scalp", "ORB 15min": "smb_scalp", "ORB 30min": "smb_scalp", "Second Chance Scalp": "smb_scalp", "BackSide Scalp": "smb_scalp", "Fashionably Late": "smb_scalp", "Spencer Scalp": "smb_scalp", "Gap Give & Go": "smb_scalp", "Tidal Wave": "smb_scalp", "Breaking News": "smb_scalp", "Momentum Breakout": "quant", "Vol Compression Breakout": "quant", "Mean Reversion": "quant", "Trend Pullback": "quant", "Gap Fade": "quant", "Relative Strength Break": "quant", "Range Expansion": "quant", "Volume Breakout": "quant", "VWAP Reversion": "quant", "Donchian Breakout": "quant" };

function BacktestModal({ open, onClose, t }: { open: boolean; onClose: () => void; t: T }) {
  const [data, setData] = useState<any>(null); const [sort, setSort] = useState("edge_score"); const [ld, setLd] = useState(true);
  useEffect(() => { if (!open) return; setLd(true); fetch(`${API}/api/backtest/patterns?sort=${sort}`).then(r => r.json()).then(d => { setData(d); setLd(false) }).catch(() => setLd(false)) }, [open, sort]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div variants={modalBg} initial="hidden" animate="visible" exit="exit"
          style={{ position: "fixed", inset: 0, zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.7)", backdropFilter: "blur(8px)" }}
          onClick={onClose}>
          <motion.div variants={modalContent} initial="hidden" animate="visible" exit="exit"
            onClick={e => e.stopPropagation()}
            style={{ background: t.bgModal, border: `1px solid ${t.border}`, borderRadius: 20, width: "95vw", maxWidth: 1400, maxHeight: "90vh", overflow: "hidden", display: "flex", flexDirection: "column", boxShadow: `0 30px 80px rgba(0,0,0,0.5), 0 0 40px ${t.glow}` }}>
            <div style={{ padding: "18px 24px", borderBottom: `1px solid ${t.border}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div><span style={{ fontSize: 22, fontWeight: 800, color: t.text }}>Backtest Results</span>
                {data?.summary?.total_signals && <span style={{ fontSize: 13, color: t.textDim, marginLeft: 14 }}>{data.summary.total_symbols} symbols • {data.summary.total_signals?.toLocaleString()} signals • {data.summary.overall_win_rate}% WR</span>}</div>
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                {["edge_score", "win_rate", "profit_factor", "expectancy", "total_signals"].map(s => {
                  const lb: Record<string, string> = { edge_score: "Edge", win_rate: "WR%", profit_factor: "PF", expectancy: "Exp", total_signals: "Signals" };
                  return <Pill key={s} active={sort === s} onClick={() => setSort(s)} t={t} s>{lb[s]}</Pill>;
                })}
                <motion.button whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }} onClick={onClose}
                  style={{ background: "none", border: "none", color: t.textDim, fontSize: 22, cursor: "pointer", marginLeft: 12 }}>✕</motion.button>
              </div>
            </div>
            <div style={{ overflow: "auto", flex: 1 }}>
              {ld ? <JuicerLoader t={t} msg="Loading backtest data..." /> : (
                <motion.table initial="hidden" animate="visible" variants={stagger}
                  style={{ width: "100%", borderCollapse: "collapse", fontSize: 14, fontFamily: "'JetBrains Mono',monospace" }}>
                  <thead><tr style={{ position: "sticky", top: 0, background: t.bgModal, borderBottom: `2px solid ${t.borderLight}`, fontSize: 11, fontWeight: 700, color: t.textMuted, textTransform: "uppercase", letterSpacing: 0.5 }}>
                    {["Pattern", "Cat", "Signals", "Win%", "PF", "Exp/R", "Avg W", "Avg L", "Edge", "Grade"].map(h => <th key={h} style={{ padding: "12px 10px", textAlign: h === "Pattern" ? "left" : "right" }}>{h}</th>)}
                  </tr></thead>
                  <tbody>{(data?.patterns || []).map((p: any) => {
                    const gr = p.edge_score >= 70 && p.total_signals >= 20 ? "A" : p.edge_score >= 55 && p.total_signals >= 10 ? "B" : p.edge_score >= 40 && p.total_signals >= 5 ? "C" : p.edge_score >= 25 ? "D" : "F";
                    const gc: Record<string, string> = { A: t.long, B: t.accent, C: t.gold, D: t.short, F: t.short }; const gc2 = gc[gr] || t.textDim;
                    return (
                      <motion.tr key={p.name} variants={fadeUp} style={{ borderBottom: `1px solid ${t.border}` }}>
                        <td style={{ padding: "10px", fontWeight: 700, color: t.text, fontFamily: "'Outfit',sans-serif", fontSize: 14 }}>{p.name}</td>
                        <td style={{ textAlign: "right" }}><CatBadge cat={PAT_CAT[p.name] || ""} /></td>
                        <td style={{ textAlign: "right", color: t.textDim }}>{p.total_signals?.toLocaleString()}</td>
                        <td style={{ textAlign: "right", color: p.win_rate >= 55 ? t.long : p.win_rate >= 45 ? t.gold : t.short, fontWeight: 700 }}>{p.win_rate?.toFixed(1)}%</td>
                        <td style={{ textAlign: "right", color: p.profit_factor >= 2 ? t.long : p.profit_factor >= 1 ? t.text : t.short }}>{p.profit_factor?.toFixed(1)}</td>
                        <td style={{ textAlign: "right", color: p.expectancy >= 0 ? t.long : t.short }}>{p.expectancy >= 0 ? "+" : ""}{p.expectancy?.toFixed(3)}</td>
                        <td style={{ textAlign: "right", color: t.long }}>{p.avg_win_r?.toFixed(1)}R</td>
                        <td style={{ textAlign: "right", color: t.short }}>{p.avg_loss_r?.toFixed(1)}R</td>
                        <td style={{ textAlign: "right" }}><span style={{ padding: "3px 8px", borderRadius: 6, background: gc2 + "15", color: gc2, fontWeight: 800 }}>{p.edge_score?.toFixed(0)}</span></td>
                        <td style={{ textAlign: "right" }}><span style={{ fontWeight: 800, color: gc2, fontSize: 15 }}>{gr}</span></td>
                      </motion.tr>);
                  })}</tbody>
                </motion.table>)}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
// ═══════════════════════════════════════════════
// SYMBOL ANALYTICS MODAL
// ═══════════════════════════════════════════════
function SymbolAnalytics({ symbol, pattern, t }: { symbol: string; pattern: string; t: any }) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [show, setShow] = useState(false);
 
  const fetchAnalytics = () => {
    if (data) { setShow(!show); return; } // Toggle if already loaded
    setLoading(true); setShow(true);
    fetch(`${API}/api/analytics/${symbol}?pattern=${encodeURIComponent(pattern)}`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  };
 
  const gradeColor = (edge: number) => edge >= 65 ? t.long : edge >= 45 ? t.gold : t.short;
 
  return (
    <div style={{ marginTop: 8 }}>
      <motion.button
        whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}
        onClick={fetchAnalytics}
        style={{
          fontSize: 12, fontWeight: 700, padding: "6px 14px", borderRadius: 8,
          border: `1px solid ${t.border}`, background: t.bgCard,
          color: t.textDim, cursor: "pointer", display: "flex", alignItems: "center", gap: 6,
        }}>
        📊 {show ? "Hide" : "Show"} Pattern Stats on {symbol}
        {loading && <motion.span animate={{ opacity: [0.3, 1, 0.3] }} transition={{ duration: 1, repeat: Infinity }}>...</motion.span>}
      </motion.button>
 
      <AnimatePresence>
        {show && data && !loading && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
            style={{ overflow: "hidden", marginTop: 8, padding: "14px 16px", background: t.bgCard, borderRadius: 12, border: `1px solid ${t.border}` }}>
 
            {/* Highlighted pattern */}
            {data.highlighted ? (
              <div style={{ marginBottom: 14, padding: "10px 14px", borderRadius: 10, background: t.accent + "08", border: `1px solid ${t.accent}20` }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: t.text, marginBottom: 6 }}>
                  {pattern} on {symbol}
                </div>
                <div style={{ display: "flex", gap: 16, flexWrap: "wrap", fontSize: 13, fontFamily: "'JetBrains Mono',monospace" }}>
                  <span><span style={{ color: t.textMuted, fontSize: 10 }}>Signals:</span> <span style={{ fontWeight: 700 }}>{data.highlighted.signals}</span></span>
                  <span><span style={{ color: t.textMuted, fontSize: 10 }}>WR:</span> <span style={{ fontWeight: 700, color: data.highlighted.win_rate >= 55 ? t.long : data.highlighted.win_rate >= 45 ? t.gold : t.short }}>{data.highlighted.win_rate}%</span></span>
                  <span><span style={{ color: t.textMuted, fontSize: 10 }}>PF:</span> <span style={{ fontWeight: 700 }}>{data.highlighted.profit_factor}</span></span>
                  <span><span style={{ color: t.textMuted, fontSize: 10 }}>Exp:</span> <span style={{ fontWeight: 700, color: data.highlighted.expectancy >= 0 ? t.long : t.short }}>{data.highlighted.expectancy >= 0 ? "+" : ""}{data.highlighted.expectancy}</span></span>
                  <span><span style={{ color: t.textMuted, fontSize: 10 }}>Avg Win:</span> <span style={{ fontWeight: 700, color: t.long }}>{data.highlighted.avg_win_r}R</span></span>
                  <span><span style={{ color: t.textMuted, fontSize: 10 }}>Avg Loss:</span> <span style={{ fontWeight: 700, color: t.short }}>{data.highlighted.avg_loss_r}R</span></span>
                  <span><span style={{ color: t.textMuted, fontSize: 10 }}>Edge:</span> <span style={{ fontWeight: 800, padding: "1px 6px", borderRadius: 4, background: gradeColor(data.highlighted.edge_score) + "15", color: gradeColor(data.highlighted.edge_score) }}>{data.highlighted.edge_score}</span></span>
                </div>
              </div>
            ) : pattern && (
              <div style={{ marginBottom: 14, fontSize: 12, color: t.textMuted, fontStyle: "italic" }}>
                No backtest data for {pattern} on {symbol} yet.
              </div>
            )}
 
            {/* Top 5 patterns on this stock */}
            {data.top_patterns?.length > 0 && (
              <div>
                <div style={{ fontSize: 11, fontWeight: 700, color: t.textMuted, marginBottom: 6, letterSpacing: 0.4 }}>
                  TOP PATTERNS ON {symbol} ({data.total_signals} total signals)
                </div>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, fontFamily: "'JetBrains Mono',monospace" }}>
                  <thead>
                    <tr style={{ fontSize: 10, color: t.textMuted, borderBottom: `1px solid ${t.border}` }}>
                      <th style={{ padding: "6px 0", textAlign: "left" }}>Pattern</th>
                      <th style={{ textAlign: "right" }}>Signals</th>
                      <th style={{ textAlign: "right" }}>WR%</th>
                      <th style={{ textAlign: "right" }}>PF</th>
                      <th style={{ textAlign: "right" }}>Exp</th>
                      <th style={{ textAlign: "right" }}>Edge</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.top_patterns.map((p: any) => (
                      <tr key={p.name} style={{ borderBottom: `1px solid ${t.border}` }}>
                        <td style={{ padding: "6px 0", fontWeight: p.name === pattern ? 700 : 500, color: p.name === pattern ? t.accent : t.text, fontFamily: "'Outfit',sans-serif" }}>{p.name}</td>
                        <td style={{ textAlign: "right", color: t.textDim }}>{p.signals}</td>
                        <td style={{ textAlign: "right", color: p.win_rate >= 55 ? t.long : p.win_rate >= 45 ? t.gold : t.short, fontWeight: 700 }}>{p.win_rate}%</td>
                        <td style={{ textAlign: "right" }}>{p.profit_factor}</td>
                        <td style={{ textAlign: "right", color: p.expectancy >= 0 ? t.long : t.short }}>{p.expectancy >= 0 ? "+" : ""}{p.expectancy}</td>
                        <td style={{ textAlign: "right" }}><span style={{ padding: "1px 5px", borderRadius: 4, background: gradeColor(p.edge_score) + "12", color: gradeColor(p.edge_score), fontWeight: 800 }}>{p.edge_score}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ fontSize: 10, color: t.textMuted, marginTop: 6 }}>
                  Source: {data.source === "cached" ? "Pre-computed backtest" : data.source === "computed" ? "Computed on-demand (90 days)" : "Backtest cache"} • {data.total_patterns} patterns analyzed
                </div>
              </div>
            )}
 
            {data.total_signals === 0 && (
              <div style={{ fontSize: 12, color: t.textMuted }}>
                No backtest data available for {symbol}. Run a backtest that includes this symbol.
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
 
 
// === INTEGRATION INTO TradeChart ===
// Find the TradeChart component's return block. After the chart div, 
// BEFORE the closing </motion.div>, add:
//
//   <SymbolAnalytics symbol={setup.symbol} pattern={setup.pattern_name} t={t} />
//
// So the end of TradeChart looks like:
//
//   {ld && <div ...>Loading chart...</div>}
//   {err && <div ...>Chart: {err}</div>}
//   <div ref={ref} style={{...}} />
//   <SymbolAnalytics symbol={setup.symbol} pattern={setup.pattern_name} t={t} />
// </motion.div>

// ═══════════════════════════════════════════════
// CHART
// ═══════════════════════════════════════════════
function TradeChart({ setup, onClose, t }: { setup: any; onClose: () => void; t: T }) {
  const ref = useRef<HTMLDivElement>(null); const [ld, setLd] = useState(true); const [err, setErr] = useState("");
  useEffect(() => {
    if (!ref.current) return; let ch: any = null;
    (async () => {
      try {
        const lc = await import("lightweight-charts"); const tf = setup.timeframe_detected?.includes("15m") ? "15min" : "5min";
        const r = await fetch(`${API}/api/chart/${setup.symbol}?timeframe=${tf}&days_back=5`); const d = await r.json();
        if (d.error) throw new Error(d.error); if (!d.bars?.length) throw new Error("No data");
        if (ref.current) ref.current.innerHTML = "";
        ch = lc.createChart(ref.current!, { width: ref.current!.clientWidth, height: 400, layout: { background: { color: t.chartBg } as any, textColor: t.chartText, fontFamily: "JetBrains Mono,monospace" }, grid: { vertLines: { color: t.chartGrid }, horzLines: { color: t.chartGrid } }, crosshair: { mode: lc.CrosshairMode.Normal }, rightPriceScale: { borderColor: t.border }, timeScale: { borderColor: t.border, timeVisible: true } });
        let s: any; if ((lc as any).CandlestickSeries) s = ch.addSeries((lc as any).CandlestickSeries, { upColor: t.long, downColor: t.short, borderUpColor: t.long, borderDownColor: t.short, wickUpColor: t.long, wickDownColor: t.short }); else s = ch.addCandlestickSeries({ upColor: t.long, downColor: t.short, borderUpColor: t.long, borderDownColor: t.short, wickUpColor: t.long, wickDownColor: t.short });
        s.setData(d.bars);

        // Signal arrow — snap to nearest bar for LWC marker rendering
        const detRaw = setup.detected_at || setup.entered_at;
        if (detRaw && d.bars?.length) {
          try {
            const sigTs = Math.floor(new Date(detRaw).getTime() / 1000);
            const nearest = d.bars.reduce((a: any, b: any) =>
              Math.abs(b.time - sigTs) < Math.abs(a.time - sigTs) ? b : a
            );
            const isLong = setup.bias === "long";
            const mk = { time: nearest.time, position: isLong ? "belowBar" : "aboveBar", color: isLong ? t.long : t.short, shape: isLong ? "arrowUp" : "arrowDown", text: "Signal", size: 2 };
            if (typeof (lc as any).createSeriesMarkers === "function") (lc as any).createSeriesMarkers(s, [mk]);
            else s.setMarkers([mk]);
          } catch {}
        }

        // Helper: add price line safely (works on v3 + v4)
        const addLine = (price: number, color: string, title: string, style: number, width: number) => {
          try { s.createPriceLine({ price, color, lineWidth: width, lineStyle: style, axisLabelVisible: true, title }); } catch {}
        };

        // 3 clean lines only
        addLine(setup.entry_price, t.accent, `ENTRY $${setup.entry_price?.toFixed(2)}`, lc.LineStyle.Dashed, 2);
        addLine(setup.stop_loss, t.short, "STOP", lc.LineStyle.Dashed, 2);
        addLine(setup.target_price, t.long, "TARGET", lc.LineStyle.Dashed, 2);
        ch.timeScale().fitContent(); setLd(false);
        const ro = new ResizeObserver(() => { if (ref.current && ch) ch.applyOptions({ width: ref.current.clientWidth }) }); ro.observe(ref.current!);
      } catch (e: any) { setErr(e.message); setLd(false) }
    })(); return () => { if (ch) ch.remove() };
  }, [setup, t]);

  return (
    <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
      style={{ background: t.bg, border: `1px solid ${t.border}`, borderRadius: 14, padding: 16, margin: "8px 0", overflow: "hidden" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <span style={{ fontSize: 16, fontWeight: 700, color: t.text }}>{setup.symbol} — {setup.pattern_name} ({setup.timeframe_detected})</span>
        <div style={{ display: "flex", gap: 14, fontSize: 14, fontWeight: 700, alignItems: "center" }}>
          <span style={{ color: t.accent }}>Entry ${setup.entry_price?.toFixed(2)}</span>
          <span style={{ color: t.short }}>Stop ${setup.stop_loss?.toFixed(2)}</span>
          <span style={{ color: t.long }}>Target ${setup.target_price?.toFixed(2)}</span>
          <motion.button whileHover={{ scale: 1.2 }} whileTap={{ scale: 0.9 }} onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: t.textDim, fontSize: 18 }}>✕</motion.button>
        </div>
      </div>
      {setup.ai_verdict?.reasoning && setup.ai_verdict.verdict !== "PENDING" && (
        <div style={{ fontSize: 13, color: t.textDim, padding: "10px 14px", background: t.bgCard, borderRadius: 10, marginBottom: 10, border: `1px solid ${t.border}` }}>
          <span style={{ fontWeight: 700, color: t.text }}>AI: </span>{setup.ai_verdict.reasoning}
        </div>)}
      {ld && <div style={{ textAlign: "center", padding: 30, color: t.textDim }}>Loading chart...</div>}
      {err && <div style={{ textAlign: "center", padding: 14, color: t.short, fontSize: 13 }}>Chart: {err}</div>}
      <div ref={ref} style={{ width: "100%", minHeight: ld ? 0 : 400, marginTop: 8 }} />
      <SymbolAnalytics symbol={setup.symbol} pattern={setup.pattern_name} t={t} />
    </motion.div>
  );
}

// ═══════════════════════════════════════════════
// SETUP ROW
// ═══════════════════════════════════════════════
function SetupRow({ s, open, toggle, t, onTrack }: { s: any; open: boolean; toggle: () => void; t: T; onTrack: (s: any) => void }) {
  const isL = s.bias === "long";
  const trigger = isL ? `⚡ BUY if ${s.symbol} reaches $${s.entry_price?.toFixed(2)}` : `⚡ SHORT if ${s.symbol} breaks $${s.entry_price?.toFixed(2)}`;

  return (
    <motion.div variants={fadeUp}>
      <motion.div
        whileHover={{ backgroundColor: t.bgHover }}
        style={{ display: "flex", alignItems: "center", gap: 8, padding: "12px 16px", borderBottom: `1px solid ${t.border}`, cursor: "pointer", borderRadius: open ? "12px 12px 0 0" : 0, transition: "background .15s" }}>
        <div onClick={toggle} style={{ display: "flex", alignItems: "center", gap: 10, flex: 1, minWidth: 0 }}>
          <div style={{ width: 70 }}>
            <div style={{ fontSize: 17, fontWeight: 800, color: t.text }}>{s.symbol}</div>
            <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 5, background: isL ? t.longBg : t.shortBg, color: isL ? t.long : t.short }}>{s.bias?.toUpperCase()}</span>
          </div>
          <div style={{ flex: 1, minWidth: 150 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 5, flexWrap: "wrap" }}>
              <span style={{ fontSize: 15, fontWeight: 700, color: t.text }}>{s.pattern_name}</span>
              <CatBadge cat={s.category} /> <TfBadge tf={s.timeframe_detected} t={t} />
              <VerdictBadge v={s.ai_verdict} t={t} />
              <CorrLabel corr={s.spy_correlation} t={t} />
            </div>
            <div style={{ fontSize: 13, color: t.accent, fontWeight: 700, marginTop: 4 }}>{trigger}</div>
            <div style={{ fontSize: 11, color: t.textMuted, marginTop: 2 }}>Target: ${s.target_price?.toFixed(2)} ({s.risk_reward_ratio?.toFixed(1)}R) • Stop: ${s.stop_loss?.toFixed(2)}</div>
          </div>
          <div style={{ display: "flex", gap: 12 }}>
            {[{ l: "ENTRY", v: s.entry_price, c: t.text }, { l: "STOP", v: s.stop_loss, c: t.short }, { l: "TARGET", v: s.target_price, c: t.long }].map(({ l, v, c }) => (
              <div key={l} style={{ textAlign: "right", minWidth: 58 }}>
                <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700, letterSpacing: 0.4 }}>{l}</div>
                <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 15, fontWeight: 700, color: c }}>${v?.toFixed(2)}</div>
              </div>))}
          </div>
          {/* R:R */}
          <div style={{ width: 42, textAlign: "center" }}>
            <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700 }}>R:R</div>
            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 15, fontWeight: 800, color: s.risk_reward_ratio >= 2 ? t.long : s.risk_reward_ratio >= 1.5 ? t.text : t.gold }}>{s.risk_reward_ratio?.toFixed(1)}</div>
          </div>
          <ScoreCell score={s.composite_score || 0} t={t} />
          <motion.div animate={{ rotate: open ? 180 : 0 }} style={{ width: 16, textAlign: "center", fontSize: 13, color: t.textDim }}>▾</motion.div>
        </div>
        <motion.button whileHover={{ scale: 1.08 }} whileTap={{ scale: 0.95 }}
          onClick={e => { e.stopPropagation(); onTrack(s) }}
          style={{ background: t.accent + "15", border: `1px solid ${t.accent}30`, borderRadius: 8, padding: "6px 12px", cursor: "pointer", fontSize: 12, fontWeight: 700, color: t.accent, whiteSpace: "nowrap" }}>
          + Track
        </motion.button>
      </motion.div>
      <AnimatePresence>{open && <TradeChart setup={s} onClose={toggle} t={t} />}</AnimatePresence>
    </motion.div>
  );
}

// ═══════════════════════════════════════════════
// TRACKED TRADE
// ═══════════════════════════════════════════════
function TrackedTrade({ trade, price, t, onRemove }: { trade: any; price: any; t: T; onRemove: () => void }) {
  const isL = trade.bias === "long"; const entry = trade.entry_price; const stop = trade.stop_loss; const target = trade.target_price;
  const cur = price?.price || 0; const risk = Math.abs(entry - stop);
  const pnlR = risk > 0 ? (isL ? (cur - entry) / risk : (entry - cur) / risk) : 0;
  const hitT = isL ? cur >= target : cur <= target; const hitS = isL ? cur <= stop : cur >= stop;
  const status = hitT ? "🎯 TARGET" : hitS ? "🛑 STOPPED" : "⏳ ACTIVE";
  const sc = hitT ? t.long : hitS ? t.short : t.accent;
  return (
    <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 20 }}
      style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 16px", borderBottom: `1px solid ${t.border}`, fontSize: 14 }}>
      <span style={{ fontWeight: 800, color: t.text, width: 60 }}>{trade.symbol}</span>
      <span style={{ fontSize: 10, padding: "2px 7px", borderRadius: 5, background: isL ? t.longBg : t.shortBg, color: isL ? t.long : t.short, fontWeight: 700 }}>{trade.bias?.toUpperCase()}</span>
      <span style={{ color: t.textDim, flex: 1 }}>{trade.pattern_name}</span>
      <span style={{ fontFamily: "'JetBrains Mono',monospace" }}>
        <span style={{ color: t.textDim }}>In: ${entry?.toFixed(2)}</span>
        <span style={{ color: t.text, marginLeft: 12 }}>Now: ${cur ? cur.toFixed(2) : "..."}</span>
        <span style={{ color: pnlR >= 0 ? t.long : t.short, marginLeft: 12, fontWeight: 800, fontSize: 15 }}>{pnlR >= 0 ? "+" : ""}{pnlR.toFixed(2)}R</span>
      </span>
      <span style={{ fontSize: 11, fontWeight: 700, padding: "3px 10px", borderRadius: 6, background: sc + "15", color: sc }}>{status}</span>
      <motion.button whileHover={{ scale: 1.2 }} whileTap={{ scale: 0.9 }} onClick={onRemove} style={{ background: "none", border: "none", color: t.textDim, cursor: "pointer", fontSize: 16 }}>✕</motion.button>
    </motion.div>
  );
}

// ═══════════════════════════════════════════════
// IN-PLAY CARD
// ═══════════════════════════════════════════════
function InPlayCard({ stock, onClick, t }: { stock: any; onClick: () => void; t: T }) {
  return (
    <motion.div
      whileHover={{ scale: 1.04, y: -3, borderColor: t.accent }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      style={{ padding: "10px 14px", background: t.bgCard, borderRadius: 12, border: `1px solid ${t.border}`, cursor: "pointer", minWidth: 130, flex: "0 0 auto" }}>
      <span style={{ fontSize: 16, fontWeight: 800, color: t.text }}>{stock.symbol}</span>
      <div style={{ fontSize: 11, color: t.textDim, marginTop: 3 }}>{stock.reason?.slice(0, 50)}</div>
    </motion.div>
  );
}

// ═══════════════════════════════════════════════
// TRADE TRACKER VIEW
// ═══════════════════════════════════════════════

const TSTAT: Record<string, { label: string; icon: string }> = {
  PENDING: { label: "Pending", icon: "⏳" },
  ACTIVE: { label: "Active", icon: "⚡" },
  AT_T1: { label: "T1 Hit", icon: "🎯" },
  AT_T2: { label: "T2 Hit", icon: "🎯🎯" },
  TRAILING: { label: "Trailing", icon: "🚀" },
  STOPPED: { label: "Stopped", icon: "🛑" },
  CLOSED: { label: "Closed", icon: "✓" },
  EXPIRED: { label: "Expired", icon: "⌛" },
};

function tStatColor(status: string, t: T): string {
  switch (status) {
    case "PENDING": return t.gold;
    case "ACTIVE": return t.accent;
    case "AT_T1": case "AT_T2": case "TRAILING": return t.long;
    case "STOPPED": return t.short;
    default: return t.textDim;
  }
}

function RBar({ trade, t }: { trade: any; t: T }) {
  const { entry_price: entry = 0, stop_loss: stop = 0, target_1: t1 = 0, target_2: t2raw = 0, unrealized_r: curR = 0, bias } = trade;
  const t2 = t2raw && t2raw !== t1 ? t2raw : 0;
  const risk = Math.abs(entry - stop);
  if (risk <= 0 || !entry) return null;
  const isL = bias === "long";
  const t1R = t1 ? (isL ? (t1 - entry) / risk : (entry - t1) / risk) : 1;
  const t2R = t2 ? (isL ? (t2 - entry) / risk : (entry - t2) / risk) : t1R * 1.8;
  const maxR = Math.max(t2R + 0.3, (curR || 0) + 0.3, 2.3);
  const range = maxR + 1.2;
  const toX = (r: number) => Math.max(0, Math.min(98, ((r + 1.2) / range) * 100));
  const entryX = toX(0); const stopX = toX(-1); const t1X = toX(t1R); const t2X = toX(t2R); const curX = toX(curR || 0);
  const fill = (curR || 0) >= 0 ? t.long : t.short;
  return (
    <div style={{ position: "relative", height: 20, marginTop: 6 }}>
      <div style={{ position: "absolute", top: 8, left: 0, right: 0, height: 3, background: t.border, borderRadius: 2 }} />
      <div style={{ position: "absolute", top: 8, left: `${Math.min(entryX, curX)}%`, width: `${Math.abs(curX - entryX)}%`, height: 3, background: fill, borderRadius: 2, opacity: 0.85 }} />
      {[{ x: stopX, c: t.short, tip: "Stop" }, { x: entryX, c: t.accent, tip: "Entry" }, { x: t1X, c: t.long, tip: "T1" }, { x: t2X, c: t.long + "88", tip: "T2" }].map(({ x, c, tip }) => (
        <div key={tip} title={tip} style={{ position: "absolute", top: 4, left: `${x}%`, width: 2, height: 12, background: c, transform: "translateX(-50%)" }} />
      ))}
      <div title={`${(curR || 0) >= 0 ? "+" : ""}${(curR || 0).toFixed(2)}R`} style={{ position: "absolute", top: 5, left: `${Math.max(1, Math.min(97, curX))}%`, width: 10, height: 10, borderRadius: "50%", background: fill, transform: "translateX(-50%)", boxShadow: `0 0 6px ${fill}80`, zIndex: 1 }} />
    </div>
  );
}

function TrackerChart({ tradeId, t, onClose }: { tradeId: string; t: T; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const [ld, setLd] = useState(true);
  const [err, setErr] = useState("");
  useEffect(() => {
    if (!ref.current) return; let ch: any = null;
    (async () => {
      try {
        const lc = await import("lightweight-charts");
        const d = await (await fetch(`${API}/api/tracker/trades/${tradeId}/chart?bars=100`)).json();
        if (d.error) throw new Error(d.error);
        if (!d.bars?.length) throw new Error("No data");
        if (ref.current) ref.current.innerHTML = "";
        ch = lc.createChart(ref.current!, { width: ref.current!.clientWidth, height: 340, layout: { background: { color: t.chartBg } as any, textColor: t.chartText, fontFamily: "JetBrains Mono,monospace" }, grid: { vertLines: { color: t.chartGrid }, horzLines: { color: t.chartGrid } }, crosshair: { mode: lc.CrosshairMode.Normal }, rightPriceScale: { borderColor: t.border }, timeScale: { borderColor: t.border, timeVisible: true } });
        let s: any;
        if ((lc as any).CandlestickSeries) s = ch.addSeries((lc as any).CandlestickSeries, { upColor: t.long, downColor: t.short, borderUpColor: t.long, borderDownColor: t.short, wickUpColor: t.long, wickDownColor: t.short });
        else s = ch.addCandlestickSeries({ upColor: t.long, downColor: t.short, borderUpColor: t.long, borderDownColor: t.short, wickUpColor: t.long, wickDownColor: t.short });
        const bars = d.bars.map((b: any) => ({ time: Math.floor(new Date(b.t).getTime() / 1000), open: b.o, high: b.h, low: b.l, close: b.c })).filter((b: any) => !isNaN(b.time));
        s.setData(bars);

        // Signal marker — snap to nearest bar so LWC always renders it
        const trade = d.trade || {};
        const sigRaw = trade.detected_at || trade.entered_at;
        if (sigRaw && bars.length) {
          const sigTs = Math.floor(new Date(sigRaw).getTime() / 1000);
          const nearest = bars.reduce((a: any, b: any) => Math.abs(b.time - sigTs) < Math.abs(a.time - sigTs) ? b : a);
          const isLong = trade.bias === "long";
          const marker = { time: nearest.time, position: isLong ? "belowBar" : "aboveBar", color: isLong ? t.long : t.short, shape: isLong ? "arrowUp" : "arrowDown", text: "Signal", size: 2 };
          try { if (typeof (lc as any).createSeriesMarkers === "function") (lc as any).createSeriesMarkers(s, [marker]); else s.setMarkers([marker]); } catch {}
        }

        const al = (price: number, color: string, title: string) => { if (price) try { s.createPriceLine({ price, color, lineWidth: 1.5, lineStyle: 2, axisLabelVisible: true, title }); } catch {} };
        const lv = d.levels || {};
        al(lv.entry, t.accent, `ENTRY $${lv.entry?.toFixed(2)}`);
        const isBreakeven = lv.t1_hit && lv.stop && lv.entry && Math.abs(lv.stop - lv.entry) < 0.01;
        al(lv.stop, isBreakeven ? t.long : t.short, isBreakeven ? `B/E $${lv.stop?.toFixed(2)}` : `STOP $${lv.stop?.toFixed(2)}`);
        al(lv.target_1, t.long, `T1 $${lv.target_1?.toFixed(2)}`);
        if (lv.target_2 && lv.target_2 !== lv.target_1) al(lv.target_2, t.long, `T2 $${lv.target_2?.toFixed(2)}`);
        if (lv.trailing_stop > 0) al(lv.trailing_stop, t.gold, `TRAIL $${lv.trailing_stop?.toFixed(2)}`);
        ch.timeScale().fitContent(); setLd(false);
        const ro = new ResizeObserver(() => { if (ref.current && ch) ch.applyOptions({ width: ref.current.clientWidth }) }); ro.observe(ref.current!);
      } catch (e: any) { setErr(e.message); setLd(false); }
    })();
    return () => { if (ch) ch.remove(); };
  }, [tradeId, t]);
  return (
    <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
      style={{ background: t.bg, border: `1px solid ${t.border}`, borderRadius: 14, padding: 14, marginTop: 6, overflow: "hidden" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <span style={{ fontWeight: 700, color: t.accent, fontSize: 13 }}>Chart — Entry / Stop / T1 / T2</span>
        <motion.button whileHover={{ scale: 1.2 }} whileTap={{ scale: 0.9 }} onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: t.textDim, fontSize: 16 }}>✕</motion.button>
      </div>
      {ld && <div style={{ textAlign: "center", padding: 20, color: t.textDim }}>Loading chart...</div>}
      {err && <div style={{ color: t.short, fontSize: 12, padding: 10 }}>{err}</div>}
      <div ref={ref} style={{ width: "100%", minHeight: ld ? 0 : 340 }} />
    </motion.div>
  );
}

function AddTradeModal({ open, onClose, onAdd, t }: { open: boolean; onClose: () => void; onAdd: () => void; t: T }) {
  const empty = { symbol: "", pattern_name: "Manual", bias: "long", timeframe: "1d", entry_price: "", stop_loss: "", target_1: "", target_2: "", notes: "" };
  const [form, setForm] = useState(empty);
  const [saving, setSaving] = useState(false);
  const set = (k: string, v: string) => setForm(p => ({ ...p, [k]: v }));
  const canSubmit = !saving && form.symbol && form.entry_price && form.stop_loss && form.target_1;
  const handleSubmit = async () => {
    if (!canSubmit) return; setSaving(true);
    try {
      await fetch(`${API}/api/tracker/trades`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...form, entry_price: parseFloat(form.entry_price), stop_loss: parseFloat(form.stop_loss), target_1: parseFloat(form.target_1), target_2: form.target_2 ? parseFloat(form.target_2) : 0 }) });
      onAdd(); onClose(); setForm(empty);
    } finally { setSaving(false); }
  };
  const inp = (extra?: object) => ({ fontSize: 13, padding: "7px 10px", borderRadius: 8, border: `1.5px solid ${t.border}`, background: t.bgCard, color: t.text, fontFamily: "'Outfit',sans-serif", width: "100%" as const, ...extra });
  const lbl = { fontSize: 10, fontWeight: 700 as const, color: t.textMuted, display: "block" as const, marginBottom: 3, letterSpacing: 0.4 };
  return (
    <AnimatePresence>
      {open && (
        <motion.div variants={modalBg} initial="hidden" animate="visible" exit="exit"
          style={{ position: "fixed", inset: 0, zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.7)", backdropFilter: "blur(8px)" }}
          onClick={onClose}>
          <motion.div variants={modalContent} initial="hidden" animate="visible" exit="exit"
            onClick={e => e.stopPropagation()}
            style={{ background: t.bgModal, border: `1px solid ${t.border}`, borderRadius: 20, width: "90vw", maxWidth: 480, padding: 24, boxShadow: `0 30px 80px rgba(0,0,0,0.5), 0 0 40px ${t.glow}` }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 18 }}>
              <span style={{ fontSize: 18, fontWeight: 800, color: t.text }}>Add Trade</span>
              <motion.button whileHover={{ scale: 1.2 }} onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: t.textDim, fontSize: 20 }}>✕</motion.button>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div><label style={lbl}>SYMBOL</label><input style={inp()} value={form.symbol} onChange={e => set("symbol", e.target.value.toUpperCase())} placeholder="AAPL" /></div>
              <div><label style={lbl}>PATTERN</label><input style={inp()} value={form.pattern_name} onChange={e => set("pattern_name", e.target.value)} /></div>
              <div><label style={lbl}>BIAS</label>
                <div style={{ display: "flex", gap: 4 }}>
                  {["long", "short"].map(b => (
                    <button key={b} onClick={() => set("bias", b)} style={{ flex: 1, padding: "7px 0", borderRadius: 8, border: `1.5px solid ${form.bias === b ? (b === "long" ? t.long : t.short) : t.border}`, background: form.bias === b ? (b === "long" ? t.longBg : t.shortBg) : "transparent", color: form.bias === b ? (b === "long" ? t.long : t.short) : t.textDim, fontWeight: 700, cursor: "pointer", fontSize: 12 }}>{b.toUpperCase()}</button>
                  ))}
                </div>
              </div>
              <div><label style={lbl}>TIMEFRAME</label>
                <select style={inp()} value={form.timeframe} onChange={e => set("timeframe", e.target.value)}>
                  {["5min", "15min", "1h", "1d"].map(tf => <option key={tf} value={tf}>{tf}</option>)}
                </select>
              </div>
              <div><label style={lbl}>ENTRY</label><input type="number" style={inp()} value={form.entry_price} onChange={e => set("entry_price", e.target.value)} placeholder="150.00" /></div>
              <div><label style={lbl}>STOP</label><input type="number" style={inp()} value={form.stop_loss} onChange={e => set("stop_loss", e.target.value)} placeholder="145.00" /></div>
              <div><label style={lbl}>TARGET 1</label><input type="number" style={inp()} value={form.target_1} onChange={e => set("target_1", e.target.value)} placeholder="160.00" /></div>
              <div><label style={lbl}>TARGET 2 (opt)</label><input type="number" style={inp()} value={form.target_2} onChange={e => set("target_2", e.target.value)} placeholder="170.00" /></div>
            </div>
            <div style={{ marginTop: 10 }}><label style={lbl}>NOTES</label><input style={inp()} value={form.notes} onChange={e => set("notes", e.target.value)} placeholder="Optional notes..." /></div>
            <motion.button whileHover={{ scale: canSubmit ? 1.02 : 1 }} whileTap={{ scale: canSubmit ? 0.98 : 1 }} onClick={handleSubmit}
              style={{ marginTop: 18, width: "100%", padding: "11px", borderRadius: 12, border: "none", background: canSubmit ? `linear-gradient(135deg, ${t.accent}, ${t.accentLight})` : t.border, color: "#fff", fontWeight: 800, fontSize: 15, cursor: canSubmit ? "pointer" : "not-allowed", opacity: canSubmit ? 1 : 0.5 }}>
              {saving ? "Adding..." : "Add Trade"}
            </motion.button>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function TrackerRow({ trade, onClose, onRemove, t }: { trade: any; onClose: () => void; onRemove: () => void; t: T }) {
  const [showChart, setShowChart] = useState(false);
  const sc = tStatColor(trade.status, t);
  const stat = TSTAT[trade.status] || { label: trade.status, icon: "" };
  const isL = trade.bias === "long";
  const r = trade.unrealized_r || 0;
  const rColor = r > 0.05 ? t.long : r < -0.05 ? t.short : t.textDim;
  return (
    <motion.div variants={fadeUp}>
      <div style={{ padding: "12px 16px", borderBottom: `1px solid ${t.border}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <div style={{ minWidth: 80 }}>
            <div style={{ fontSize: 16, fontWeight: 800, color: t.text }}>{trade.symbol}</div>
            <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 5, background: isL ? t.longBg : t.shortBg, color: isL ? t.long : t.short }}>{isL ? "LONG" : "SHORT"}</span>
          </div>
          <div style={{ flex: 1, minWidth: 160 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: t.text }}>{trade.pattern_name}
              {trade.timeframe && <span style={{ fontSize: 10, color: t.textMuted, marginLeft: 6 }}>{trade.timeframe}</span>}
            </div>
            <div style={{ display: "flex", gap: 6, marginTop: 3, flexWrap: "wrap", alignItems: "center" }}>
              <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 5, background: sc + "20", color: sc }}>{stat.icon} {stat.label}</span>
              {trade.t1_hit && <span style={{ fontSize: 10, color: t.long, fontWeight: 700 }}>✓T1</span>}
              {trade.t2_hit && <span style={{ fontSize: 10, color: t.long, fontWeight: 700 }}>✓T2</span>}
              {trade.source === "manual" && <span style={{ fontSize: 10, color: t.textMuted }}>manual</span>}
              {trade.confidence > 0 && <span style={{ fontSize: 10, color: t.textMuted }}>{(trade.confidence * 100).toFixed(0)}% conf</span>}
            </div>
          </div>
          <div style={{ display: "flex", gap: 16, fontFamily: "'JetBrains Mono',monospace", fontSize: 12 }}>
            {[{ l: "ENTRY", v: trade.entry_price, c: t.accent }, { l: trade.t1_hit && Math.abs(trade.stop_loss - trade.entry_price) < 0.01 ? "B/E STOP" : "STOP", v: trade.stop_loss, c: t.t1_hit && Math.abs(trade.stop_loss - trade.entry_price) < 0.01 ? t.long : t.short }, { l: "T1", v: trade.target_1, c: t.long }, { l: "NOW", v: trade.current_price, c: t.text }].map(({ l, v, c }) => (
              <div key={l} style={{ textAlign: "right" }}>
                <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700 }}>{l}</div>
                <div style={{ fontWeight: 700, color: c }}>{v ? `$${v.toFixed(2)}` : "—"}</div>
              </div>
            ))}
          </div>
          <div style={{ textAlign: "center", minWidth: 72 }}>
            <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700, marginBottom: 1 }}>UNREAL R</div>
            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 20, fontWeight: 800, color: rColor }}>{r >= 0 ? "+" : ""}{r.toFixed(2)}R</div>
          </div>
          <div style={{ textAlign: "center", minWidth: 52 }}>
            <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700, marginBottom: 1 }}>PEAK</div>
            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 12, fontWeight: 700, color: t.long }}>+{(trade.peak_r || 0).toFixed(2)}R</div>
          </div>
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => setShowChart(!showChart)}
              style={{ fontSize: 12, padding: "5px 10px", borderRadius: 8, border: `1px solid ${showChart ? t.accent + "50" : t.border}`, background: showChart ? t.accent + "15" : t.bgCard, color: showChart ? t.accent : t.textDim, cursor: "pointer", fontWeight: 700 }}>📈</motion.button>
            <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={onClose}
              style={{ fontSize: 11, fontWeight: 700, padding: "5px 10px", borderRadius: 8, border: `1px solid ${t.short}30`, background: t.short + "10", color: t.short, cursor: "pointer" }}>Close</motion.button>
            <motion.button whileHover={{ scale: 1.15 }} whileTap={{ scale: 0.9 }} onClick={onRemove}
              style={{ background: "none", border: "none", cursor: "pointer", color: t.textMuted, fontSize: 15 }}>✕</motion.button>
          </div>
        </div>
        <RBar trade={trade} t={t} />
      </div>
      <AnimatePresence>
        {showChart && <div style={{ padding: "0 16px 12px" }}><TrackerChart tradeId={trade.id} t={t} onClose={() => setShowChart(false)} /></div>}
      </AnimatePresence>
    </motion.div>
  );
}

function TradeTrackerView({ t }: { t: T }) {
  const [trades, setTrades] = useState<any[]>([]);
  const [closedTrades, setClosedTrades] = useState<any[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [schedule, setSchedule] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [showClosed, setShowClosed] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [addOpen, setAddOpen] = useState(false);
  const [archiving, setArchiving] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const [ar, sr] = await Promise.all([fetch(`${API}/api/tracker/trades`), fetch(`${API}/api/tracker/summary`)]);
      const [ad, sd] = await Promise.all([ar.json(), sr.json()]);
      setTrades(ad.trades || []); setSummary(sd);
    } catch {}
  }, []);

  const fetchClosed = useCallback(async () => {
    try { const d = await (await fetch(`${API}/api/tracker/trades/closed`)).json(); setClosedTrades(d.trades || []); } catch {}
  }, []);

  useEffect(() => {
    (async () => {
      setLoading(true);
      await fetchData();
      setLoading(false);
    })();
    fetch(`${API}/api/tracker/schedule`).then(r => r.json()).then(setSchedule).catch(() => {});
  }, []);
  useEffect(() => { if (showClosed) fetchClosed(); }, [showClosed]);

  const handleScan = async () => { setScanning(true); try { await fetch(`${API}/api/tracker/scan`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ top_n: 50 }) }); await fetchData(); } finally { setScanning(false); } };
  const handleRefresh = async () => { setRefreshing(true); try { await fetch(`${API}/api/tracker/refresh`, { method: "POST" }); await fetchData(); } finally { setRefreshing(false); } };
  const handleClose = async (id: string) => { await fetch(`${API}/api/tracker/trades/${id}/close`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reason: "manual" }) }); fetchData(); };
  const handleRemove = async (id: string) => { await fetch(`${API}/api/tracker/trades/${id}`, { method: "DELETE" }); fetchData(); };
  const handleArchive = async () => { setArchiving(true); try { await fetch(`${API}/api/tracker/archive`, { method: "POST" }); await fetchData(); if (showClosed) fetchClosed(); } finally { setArchiving(false); } };

  const activeOnly = trades.filter(tr => !["STOPPED", "CLOSED", "EXPIRED"].includes(tr.status));
  const recentClosed = trades.filter(tr => ["STOPPED", "CLOSED", "EXPIRED"].includes(tr.status));

  const statCards = summary ? [
    { l: "Active", v: String(summary.active_count), c: t.accent },
    { l: "Long / Short", v: `${summary.active_long}L  ${summary.active_short}S`, c: t.text },
    { l: "Unrealized", v: `${(summary.total_unrealized_r || 0) >= 0 ? "+" : ""}${(summary.total_unrealized_r || 0).toFixed(2)}R`, c: (summary.total_unrealized_r || 0) >= 0 ? t.long : t.short },
    { l: "Realized", v: `${(summary.total_realized_r || 0) >= 0 ? "+" : ""}${(summary.total_realized_r || 0).toFixed(2)}R`, c: (summary.total_realized_r || 0) >= 0 ? t.long : t.short },
    { l: "Win Rate", v: `${(summary.win_rate || 0).toFixed(0)}%`, c: (summary.win_rate || 0) >= 55 ? t.long : (summary.win_rate || 0) >= 45 ? t.gold : t.textDim },
    { l: "Closed", v: String(summary.closed_count || 0), c: t.textDim },
  ] : [];

  return (
    <motion.div initial="hidden" animate="visible" variants={fadeUp}>
      {/* Summary stat cards */}
      {summary && (
        <motion.div variants={fadeUp} style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 8, marginBottom: 14 }}>
          {statCards.map(({ l, v, c }) => (
            <div key={l} style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 12, padding: "12px 14px", textAlign: "center" }}>
              <div style={{ fontSize: 10, color: t.textMuted, fontWeight: 700, marginBottom: 4, letterSpacing: 0.4 }}>{l.toUpperCase()}</div>
              <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 17, fontWeight: 800, color: c }}>{v}</div>
            </div>
          ))}
        </motion.div>
      )}

      {/* Action bar */}
      <motion.div variants={fadeUp} style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap", alignItems: "center" }}>
        <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }} onClick={handleScan} disabled={scanning}
          style={{ fontSize: 13, fontWeight: 700, padding: "8px 18px", borderRadius: 10, border: "none", background: scanning ? t.border : `linear-gradient(135deg, ${t.accent}, ${t.accentLight})`, color: "#fff", cursor: scanning ? "wait" : "pointer" }}>
          {scanning ? "Scanning..." : "🔍 Scan Setups"}
        </motion.button>
        <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }} onClick={handleRefresh} disabled={refreshing}
          style={{ fontSize: 13, fontWeight: 700, padding: "8px 18px", borderRadius: 10, border: `1px solid ${t.border}`, background: t.bgCard, color: t.textDim, cursor: refreshing ? "wait" : "pointer" }}>
          {refreshing ? "Refreshing..." : "🔄 Refresh Prices"}
        </motion.button>
        <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }} onClick={() => setAddOpen(true)}
          style={{ fontSize: 13, fontWeight: 700, padding: "8px 18px", borderRadius: 10, border: `1px solid ${t.accent}40`, background: t.accent + "10", color: t.accent, cursor: "pointer" }}>
          + Add Trade
        </motion.button>
        <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }} onClick={() => setShowClosed(!showClosed)}
          style={{ fontSize: 13, fontWeight: 700, padding: "8px 18px", borderRadius: 10, border: `1px solid ${t.border}`, background: showClosed ? t.bgHover : t.bgCard, color: t.textDim, cursor: "pointer" }}>
          📋 History {showClosed ? "▲" : "▼"}
        </motion.button>
        {recentClosed.length > 0 && (
          <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }} onClick={handleArchive} disabled={archiving}
            style={{ fontSize: 13, fontWeight: 700, padding: "8px 18px", borderRadius: 10, border: `1px solid ${t.border}`, background: t.bgCard, color: t.textMuted, cursor: archiving ? "wait" : "pointer" }}>
            {archiving ? "Archiving..." : `📦 Archive ${recentClosed.length}`}
          </motion.button>
        )}
        <div style={{ marginLeft: "auto", display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 2 }}>
          {summary?.last_updated && <span style={{ fontSize: 11, color: t.textMuted }}>Updated: {new Date(summary.last_updated).toLocaleTimeString()}</span>}
          {schedule?.jobs && (() => {
            const scan = schedule.jobs.find((j: any) => j.id === "daily_scan");
            const ref = schedule.jobs.find((j: any) => j.id === "refresh");
            const fmt = (iso: string | null) => iso ? new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "—";
            return (
              <span style={{ fontSize: 10, color: t.textMuted }}>
                ⏰ Next scan {fmt(scan?.next_run)} · refresh {fmt(ref?.next_run)}
                {schedule.last_scan?.ran_at && <span style={{ marginLeft: 6 }}>· last scan {new Date(schedule.last_scan.ran_at).toLocaleDateString()} (+{schedule.last_scan.added})</span>}
              </span>
            );
          })()}
        </div>
      </motion.div>

      {/* Active trades */}
      <motion.div variants={fadeUp} style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 14, overflow: "hidden", marginBottom: 12 }}>
        <div style={{ padding: "10px 16px", borderBottom: `1px solid ${t.border}`, display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 14, fontWeight: 800, color: t.text }}>Active Trades</span>
          <span style={{ fontSize: 12, color: t.textMuted }}>({activeOnly.length})</span>
          {activeOnly.length > 0 && <span style={{ fontSize: 11, color: t.textDim, marginLeft: 4 }}>— click 📈 for chart, Close to lock in R, ✕ to remove</span>}
        </div>
        {loading ? <JuicerLoader t={t} msg="Loading tracker..." /> : activeOnly.length === 0 ? (
          <div style={{ padding: "36px 24px", textAlign: "center", color: t.textDim }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>📭</div>
            <div style={{ fontSize: 15, fontWeight: 700, color: t.text }}>No active trades</div>
            <div style={{ fontSize: 13, color: t.textMuted, marginTop: 6 }}>Click <strong style={{ color: t.accent }}>Scan Setups</strong> to auto-detect high-confidence setups, or <strong style={{ color: t.accent }}>+ Add Trade</strong> to track one manually.</div>
          </div>
        ) : (
          <motion.div initial="hidden" animate="visible" variants={stagger}>
            <AnimatePresence>
              {activeOnly.map(tr => <TrackerRow key={tr.id} trade={tr} t={t} onClose={() => handleClose(tr.id)} onRemove={() => handleRemove(tr.id)} />)}
            </AnimatePresence>
          </motion.div>
        )}
      </motion.div>

      {/* Recently closed (not yet archived) */}
      {recentClosed.length > 0 && (
        <motion.div variants={fadeUp} style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 14, overflow: "hidden", marginBottom: 12 }}>
          <div style={{ padding: "10px 16px", borderBottom: `1px solid ${t.border}`, display: "flex", gap: 8, alignItems: "center" }}>
            <span style={{ fontSize: 13, fontWeight: 700, color: t.textDim }}>Recently Closed ({recentClosed.length})</span>
            <span style={{ fontSize: 11, color: t.textMuted }}>— archive to move to history</span>
          </div>
          {recentClosed.map(tr => {
            const sc = tStatColor(tr.status, t); const rz = tr.realized_r || 0;
            const risk = Math.abs((tr.entry_price || 0) - (tr.stop_loss || 0));
            const dps = risk > 0 ? rz * risk : 0;
            const pct = tr.entry_price > 0 && risk > 0 ? (dps / tr.entry_price * 100) : 0;
            const col = rz > 0 ? t.long : rz < 0 ? t.short : t.textDim;
            return (
              <div key={tr.id} style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 16px", borderBottom: `1px solid ${t.border}`, fontSize: 13 }}>
                <span style={{ fontWeight: 800, color: t.text, minWidth: 60 }}>{tr.symbol}</span>
                <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 5, background: sc + "15", color: sc }}>{TSTAT[tr.status]?.icon} {tr.status}</span>
                <span style={{ color: t.textDim, flex: 1 }}>{tr.pattern_name}</span>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontFamily: "'JetBrains Mono',monospace", fontWeight: 800, color: col, fontSize: 15 }}>{rz >= 0 ? "+" : ""}{rz.toFixed(2)}R</div>
                  {dps !== 0 && <div style={{ fontSize: 10, color: col }}>{dps >= 0 ? "+" : ""}${Math.abs(dps).toFixed(2)}/sh · {pct >= 0 ? "+" : ""}{pct.toFixed(1)}%</div>}
                </div>
                {tr.closed_at && <span style={{ fontSize: 11, color: t.textMuted }}>{new Date(tr.closed_at).toLocaleDateString()}</span>}
                <motion.button whileHover={{ scale: 1.1 }} onClick={() => handleRemove(tr.id)} style={{ background: "none", border: "none", cursor: "pointer", color: t.textMuted, fontSize: 14 }}>✕</motion.button>
              </div>
            );
          })}
        </motion.div>
      )}

      {/* Archived history accordion */}
      <AnimatePresence>
        {showClosed && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
            style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 14, overflow: "hidden", marginBottom: 12 }}>
            <div style={{ padding: "10px 16px", borderBottom: `1px solid ${t.border}` }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: t.textDim }}>Archived History ({closedTrades.length})</span>
            </div>
            {closedTrades.length === 0
              ? <div style={{ padding: 24, textAlign: "center", color: t.textMuted, fontSize: 13 }}>No archived trades yet. Archive closed trades above to build history.</div>
              : closedTrades.slice(0, 100).map(tr => {
                  const rz = tr.realized_r || 0;
                  const col = rz > 0 ? t.long : rz < 0 ? t.short : t.textDim;
                  const risk = Math.abs((tr.entry_price || 0) - (tr.stop_loss || 0));
                  const dps = risk > 0 ? rz * risk : 0;
                  const pct = tr.entry_price > 0 && risk > 0 ? (dps / tr.entry_price * 100) : 0;
                  return (
                    <div key={tr.id} style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 16px", borderBottom: `1px solid ${t.border}`, fontSize: 12 }}>
                      <span style={{ fontWeight: 800, color: t.text, minWidth: 55 }}>{tr.symbol}</span>
                      <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 4, background: t.border, color: t.textDim }}>{tr.status}</span>
                      <span style={{ color: t.textDim, flex: 1 }}>{tr.pattern_name}</span>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontFamily: "'JetBrains Mono',monospace", fontWeight: 700, color: col }}>{rz >= 0 ? "+" : ""}{rz.toFixed(2)}R</div>
                        {dps !== 0 && <div style={{ fontSize: 10, color: col }}>{dps >= 0 ? "+" : ""}${Math.abs(dps).toFixed(2)}/sh · {pct >= 0 ? "+" : ""}{pct.toFixed(1)}%</div>}
                      </div>
                      {tr.closed_at && <span style={{ color: t.textMuted, fontSize: 11 }}>{new Date(tr.closed_at).toLocaleDateString()}</span>}
                    </div>
                  );
                })
            }
          </motion.div>
        )}
      </AnimatePresence>

      <AddTradeModal open={addOpen} onClose={() => setAddOpen(false)} onAdd={fetchData} t={t} />
    </motion.div>
  );
}

// ═══════════════════════════════════════════════
// ANALYZE VIEW — on-demand symbol deep-dive
// ═══════════════════════════════════════════════
function AnalyzeChart({ bars, setups, t }: { bars: any[]; setups: any[]; t: T }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current || !bars.length) return;
    let ch: any = null;
    (async () => {
      try {
        const lc = await import("lightweight-charts");
        if (ref.current) ref.current.innerHTML = "";
        ch = lc.createChart(ref.current!, {
          width: ref.current!.clientWidth, height: 380,
          layout: { background: { color: t.chartBg } as any, textColor: t.chartText, fontFamily: "JetBrains Mono,monospace" },
          grid: { vertLines: { color: t.chartGrid }, horzLines: { color: t.chartGrid } },
          crosshair: { mode: lc.CrosshairMode.Normal },
          rightPriceScale: { borderColor: t.border }, timeScale: { borderColor: t.border, timeVisible: true },
        });
        let s: any;
        if ((lc as any).CandlestickSeries) s = ch.addSeries((lc as any).CandlestickSeries, { upColor: t.long, downColor: t.short, borderUpColor: t.long, borderDownColor: t.short, wickUpColor: t.long, wickDownColor: t.short });
        else s = ch.addCandlestickSeries({ upColor: t.long, downColor: t.short, borderUpColor: t.long, borderDownColor: t.short, wickUpColor: t.long, wickDownColor: t.short });
        s.setData(bars);
        // Draw setup levels for each detected setup
        const addLine = (price: number, color: string, title: string) => {
          if (!price) return;
          try { s.createPriceLine({ price, color, lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title }); } catch {}
        };
        setups.forEach((su: any) => {
          addLine(su.entry_price, t.accent, `${su.pattern_name} ENTRY`);
          addLine(su.stop_loss, t.short, "STOP");
          addLine(su.target_1 || su.target_price, t.long, "T1");
          if (su.target_2 && su.target_2 !== su.target_1) addLine(su.target_2, t.long, "T2");
        });
        // Arrow markers for each setup
        if (setups.length && bars.length) {
          const lastBar = bars[bars.length - 1];
          const markers = setups.map((su: any) => {
            const isLong = su.bias === "long";
            return { time: lastBar.time, position: isLong ? "belowBar" : "aboveBar", color: isLong ? t.long : t.short, shape: isLong ? "arrowUp" : "arrowDown", text: su.pattern_name, size: 1 };
          });
          try {
            if (typeof (lc as any).createSeriesMarkers === "function") (lc as any).createSeriesMarkers(s, markers);
            else s.setMarkers(markers);
          } catch {}
        }
        ch.timeScale().fitContent();
        const ro = new ResizeObserver(() => { if (ref.current && ch) ch.applyOptions({ width: ref.current.clientWidth }) });
        ro.observe(ref.current!);
      } catch {}
    })();
    return () => { if (ch) ch.remove(); };
  }, [bars, setups, t]);
  return <div ref={ref} style={{ width: "100%", minHeight: 380 }} />;
}

function AnalyzeView({ t, onTrack }: { t: T; onTrack: (s: any) => void }) {
  const [sym, setSym] = useState("");
  const [tf, setTf] = useState("1d");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [err, setErr] = useState("");
  const [openIdx, setOpenIdx] = useState<number | null>(null);

  const run = async (s?: string, f?: string) => {
    const symbol = (s || sym).trim().toUpperCase();
    const timeframe = f || tf;
    if (!symbol) return;
    setLoading(true); setErr(""); setResult(null); setOpenIdx(null);
    try {
      const r = await fetch(`${API}/api/analyze/${symbol}?timeframe=${timeframe}&days=365`);
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      setResult(d);
    } catch (e: any) { setErr(e.message); }
    finally { setLoading(false); }
  };

  const chg = result?.change_pct ?? 0;
  const chgColor = chg >= 0 ? t.long : t.short;

  return (
    <motion.div initial="hidden" animate="visible" variants={fadeUp}>
      {/* Search bar */}
      <div style={{ display: "flex", gap: 8, alignItems: "flex-end", marginBottom: 16, flexWrap: "wrap" }}>
        <div>
          <label style={{ fontSize: 10, color: t.textMuted, display: "block", marginBottom: 3, fontWeight: 700 }}>SYMBOL</label>
          <input
            type="text" value={sym}
            onChange={e => setSym(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === "Enter" && run()}
            placeholder="AAPL, NVDA, TSLA..."
            style={{ fontSize: 17, fontWeight: 800, padding: "8px 14px", borderRadius: 10, width: 160, border: `1.5px solid ${t.border}`, background: t.bgCard, color: t.text, fontFamily: "'Outfit',sans-serif" }}
          />
        </div>
        <div style={{ display: "flex", gap: 2, background: t.border, borderRadius: 8, padding: 3 }}>
          {(["1d", "5min", "15min"] as const).map(f => (
            <Pill key={f} active={tf === f} onClick={() => setTf(f)} t={t} s>{f}</Pill>
          ))}
        </div>
        <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}
          onClick={() => run()} disabled={loading || !sym}
          style={{ fontSize: 14, fontWeight: 700, padding: "9px 24px", borderRadius: 10, border: "none", background: loading ? t.border : `linear-gradient(135deg, ${t.accent}, ${t.accentLight})`, color: "#fff", cursor: loading ? "wait" : "pointer" }}>
          {loading ? "Analyzing..." : "Analyze 🔬"}
        </motion.button>
        <span style={{ fontSize: 12, color: t.textMuted, alignSelf: "center" }}>Any ticker on the market — fetched on demand</span>
      </div>

      {err && <div style={{ padding: "10px 14px", borderRadius: 10, background: t.shortBg, color: t.short, fontSize: 13, marginBottom: 12 }}>{err}</div>}
      {loading && <JuicerLoader t={t} msg={`Fetching and analyzing ${sym}...`} />}

      {result && (
        <motion.div initial="hidden" animate="visible" variants={stagger}>
          {/* Price header */}
          <motion.div variants={fadeUp} style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 14, padding: "16px 20px", marginBottom: 12 }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
              <span style={{ fontSize: 28, fontWeight: 900, color: t.text }}>{result.symbol}</span>
              <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 24, fontWeight: 800, color: t.text }}>${result.last_price}</span>
              <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 16, fontWeight: 700, color: chgColor }}>{chg >= 0 ? "+" : ""}{chg}%</span>
              <span style={{ fontSize: 12, color: t.textMuted }}>{result.timeframe} • {result.bars_count} bars</span>
            </div>
            {/* Stat grid */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 8, marginTop: 14 }}>
              {[
                { l: "ATR", v: `$${result.atr} (${result.atr_pct}%)`, c: t.text },
                { l: "RSI 14", v: result.rsi14 != null ? String(result.rsi14) : "—", c: result.rsi14 > 70 ? t.short : result.rsi14 < 30 ? t.long : t.text },
                { l: "52W HIGH", v: `$${result.high_52w}`, c: t.text },
                { l: "52W LOW", v: `$${result.low_52w}`, c: t.text },
                { l: "FROM HIGH", v: `${result.pct_from_high}%`, c: result.pct_from_high < -20 ? t.short : t.text },
                { l: "FROM LOW", v: `+${result.pct_from_low}%`, c: result.pct_from_low > 50 ? t.long : t.text },
                { l: "20D SMA", v: result.sma20 ? `$${result.sma20}` : "—", c: result.sma20 && result.last_price > result.sma20 ? t.long : t.short },
                { l: "50D SMA", v: result.sma50 ? `$${result.sma50}` : "—", c: result.sma50 && result.last_price > result.sma50 ? t.long : t.short },
                { l: "200D SMA", v: result.sma200 ? `$${result.sma200}` : "—", c: result.sma200 && result.last_price > result.sma200 ? t.long : t.short },
              ].map(({ l, v, c }) => (
                <div key={l} style={{ background: t.bg, borderRadius: 8, padding: "8px 12px" }}>
                  <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700, marginBottom: 2 }}>{l}</div>
                  <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 13, fontWeight: 700, color: c }}>{v}</div>
                </div>
              ))}
            </div>
            {/* Trend narrative */}
            <div style={{ marginTop: 12, fontSize: 13, color: t.textDim, borderTop: `1px solid ${t.border}`, paddingTop: 10 }}>
              <span style={{ fontWeight: 700, color: t.text }}>Trend: </span>{result.trend_line}
            </div>
          </motion.div>

          {/* Chart */}
          <motion.div variants={fadeUp} style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 14, padding: 14, marginBottom: 12 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: t.accent, marginBottom: 10 }}>
              {result.symbol} — {result.timeframe.toUpperCase()} Chart
              {result.setups_count > 0 && <span style={{ marginLeft: 10, fontSize: 11, color: t.textMuted }}>• {result.setups_count} setup{result.setups_count !== 1 ? "s" : ""} detected (levels shown)</span>}
            </div>
            <AnalyzeChart bars={result.chart_bars} setups={result.setups} t={t} />
          </motion.div>

          {/* Detected setups */}
          {result.setups_count > 0 ? (
            <motion.div variants={fadeUp} style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 14, overflow: "hidden", marginBottom: 12 }}>
              <div style={{ padding: "10px 16px", borderBottom: `1px solid ${t.border}`, fontSize: 13, fontWeight: 700, color: t.text }}>
                Detected Setups ({result.setups_count})
              </div>
              <AnimatePresence>
                {result.setups.map((s: any, i: number) => (
                  <SetupRow key={`${s.pattern_name}-${i}`} s={s} open={openIdx === i} toggle={() => setOpenIdx(openIdx === i ? null : i)} t={t} onTrack={onTrack} />
                ))}
              </AnimatePresence>
            </motion.div>
          ) : (
            <motion.div variants={fadeUp} style={{ textAlign: "center", padding: 30, color: t.textDim, fontSize: 14, background: t.bgCard, borderRadius: 14, border: `1px solid ${t.border}` }}>
              No setups detected at current {result.timeframe} structure for {result.symbol}
            </motion.div>
          )}
        </motion.div>
      )}

      {!loading && !result && !err && (
        <motion.div initial="hidden" animate="visible" variants={fadeUp} style={{ textAlign: "center", padding: 60 }}>
          <Logo size={60} />
          <div style={{ fontSize: 20, fontWeight: 700, marginTop: 16, color: t.text }}>Analyze any stock</div>
          <div style={{ fontSize: 14, color: t.textDim, marginTop: 6, maxWidth: 420, margin: "8px auto 0" }}>
            Enter any ticker — daily bars are fetched live, setups are detected, and full price structure is analyzed. Results cached for the day.
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}

// ═══════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════
export default function App() {
  const [dark, setDark] = useState(true); const t = dark ? DARK : LIGHT;
  const [view, setView] = useState<"opp" | "scan" | "tracker" | "analyze">("opp");
  const [symbol, setSymbol] = useState("AAPL"); const [scanSetups, setScanSetups] = useState<any[]>([]);
  const [topSetups, setTopSetups] = useState<any[]>([]); const [inPlay, setInPlay] = useState<any[]>([]);
  const [mktSummary, setMktSummary] = useState("");
  const [loading, setLoading] = useState(false); const [topLoading, setTopLoading] = useState(true);
  const [error, setError] = useState(""); const [chartIdx, setChartIdx] = useState<number | null>(null);
  const [fBias, setFBias] = useState("ALL"); const [fCat, setFCat] = useState("ALL"); const [sortBy, setSortBy] = useState<"score" | "rr">("score");
  const [mode, setMode] = useState<"today" | "active">("today");
  const [regime, setRegime] = useState<any>(null); const [hotStrats, setHotStrats] = useState<any[]>([]);
  const [pc, setPc] = useState(47); const [mktOpen, setMktOpen] = useState(true);
  const [btOpen, setBtOpen] = useState(false);
  const [tracked, setTracked] = useState<any[]>([]); const [prices, setPrices] = useState<any>({});

  useEffect(() => {
    fetch(`${API}/api/health`).then(r => r.json()).then(d => { if (d.patterns) setPc(d.patterns); if (d.market_open !== undefined) setMktOpen(d.market_open) }).catch(() => { });
    fetch(`${API}/api/regime`).then(r => r.json()).then(d => { if (d.regime) setRegime(d) }).catch(() => { });
    fetch(`${API}/api/hot-strategies?top_n=5`).then(r => r.json()).then(d => { if (d.strategies) setHotStrats(d.strategies) }).catch(() => { });
    setTopLoading(true);
    fetch(`${API}/api/top-opportunities`).then(r => r.json()).then(d => {
      if (d.setups) setTopSetups(d.setups); if (d.in_play?.stocks) setInPlay(d.in_play.stocks);
      if (d.market_summary) setMktSummary(d.market_summary); if (d.market_open !== undefined) setMktOpen(d.market_open);
    }).catch(() => { }).finally(() => setTopLoading(false));
  }, []);

  useEffect(() => {
    if (!tracked.length) return;
    const f = () => { fetch(`${API}/api/track-prices?symbols=${tracked.map((x: any) => x.symbol).join(",")}`).then(r => r.json()).then(d => { if (d.prices) setPrices(d.prices) }).catch(() => { }) };
    f(); const iv = setInterval(f, 300000); return () => clearInterval(iv);
  }, [tracked]);

  const handleScan = useCallback(async () => {
    setLoading(true); setError(""); setScanSetups([]); setChartIdx(null);
    try { const r = await fetch(`${API}/api/scan?symbol=${symbol}&mode=${mode}&ai=true`); const d = await r.json(); if (d.error) throw new Error(d.error); setScanSetups(d.setups) } catch (e: any) { setError(e.message) } finally { setLoading(false) }
  }, [symbol, mode]);

  const scanSym = (sym: string) => {
    setSymbol(sym); setView("scan"); setTimeout(() => {
      setLoading(true); setError(""); setScanSetups([]); setChartIdx(null);
      fetch(`${API}/api/scan?symbol=${sym}&mode=active&ai=true`).then(r => r.json()).then(d => { if (!d.error) setScanSetups(d.setups) }).catch(e => setError(e.message)).finally(() => setLoading(false));
    }, 50)
  };

  const addTrack = (s: any) => {
    if (!tracked.find((x: any) => x.symbol === s.symbol && x.pattern_name === s.pattern_name)) setTracked(p => [...p, s]);
    fetch(`${API}/api/tracker/trades`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ symbol: s.symbol, pattern_name: s.pattern_name, bias: s.bias || "long", timeframe: (s.timeframe_detected || "5min").split("&")[0].trim() || "5min", entry_price: s.entry_price, stop_loss: s.stop_loss, target_1: s.target_price, target_2: s.target_2 || 0, confidence: (s.composite_score || 50) / 100, description: s.description || "" }) }).catch(() => {});
  };
  const active = view === "opp" ? topSetups : view === "scan" ? scanSetups : [];
  const filtered = useMemo(() => { let r = active; r = r.filter((s: any) => (s.composite_score || 0) >= 45); if (fBias !== "ALL") r = r.filter((s: any) => s.bias === fBias.toLowerCase()); if (fCat !== "ALL") r = r.filter((s: any) => s.category === fCat); return [...r].sort((a: any, b: any) => sortBy === "rr" ? b.risk_reward_ratio - a.risk_reward_ratio : (b.composite_score || 0) - (a.composite_score || 0)) }, [active, fBias, fCat, sortBy]);

  return (
    <div style={{ background: t.bg, minHeight: "100vh", fontFamily: "'Outfit',sans-serif", color: t.text, transition: "background .3s" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Outfit:wght@400;500;600;700;800;900&display=swap');*{box-sizing:border-box;margin:0;padding:0}::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:${t.border};border-radius:3px}input:focus,button:focus{outline:none}`}</style>

      {/* Header */}
      <motion.div initial={{ y: -20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.4 }}
        style={{ padding: "10px 24px", borderBottom: `1px solid ${t.border}`, display: "flex", justifyContent: "space-between", alignItems: "center", background: t.bgCard }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Logo size={34} />
          <span style={{ fontSize: 24, fontWeight: 900, letterSpacing: -0.5, background: `linear-gradient(135deg, ${t.accent}, ${t.accentLight})`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Juicer</span>
          <span style={{ fontSize: 10, color: t.textMuted }}>v1.0</span>
          <div style={{ display: "flex", gap: 2, marginLeft: 14, background: t.border, borderRadius: 8, padding: 3 }}>
            <Pill active={view === "opp"} onClick={() => setView("opp")} t={t}>Opportunities</Pill>
            <Pill active={view === "scan"} onClick={() => setView("scan")} t={t}>Scan</Pill>
            <Pill active={view === "tracker"} onClick={() => setView("tracker")} t={t}>Tracker</Pill>
            <Pill active={view === "analyze"} onClick={() => setView("analyze")} t={t}>Analyze</Pill>
          </div>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <RegimePill r={regime} t={t} />
          {!mktOpen && <span style={{ fontSize: 10, padding: "3px 8px", borderRadius: 5, background: t.short + "15", color: t.short, fontWeight: 700 }}>MARKET CLOSED</span>}
          {active.length > 0 && <span style={{ fontSize: 12, color: t.textDim, fontWeight: 600 }}>{active.filter((s: any) => s.bias === "long").length}L / {active.filter((s: any) => s.bias === "short").length}S</span>}
          <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => setBtOpen(true)}
            style={{ fontSize: 12, fontWeight: 700, padding: "5px 12px", borderRadius: 8, border: `1px solid ${t.border}`, background: t.bgCard, color: t.textDim, cursor: "pointer" }}>📊 Backtest</motion.button>
          <motion.button whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}
            onClick={() => setDark(!dark)} style={{ background: t.border, border: "none", borderRadius: 8, padding: "5px 10px", cursor: "pointer", fontSize: 14, color: t.text }}>{dark ? "☀️" : "🌙"}</motion.button>
        </div>
      </motion.div>

      <div style={{ padding: "14px 24px" }}>
        {/* Tracked */}
        <AnimatePresence>
          {tracked.length > 0 && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
              style={{ background: t.bgCard, border: `1px solid ${t.accent}25`, borderRadius: 14, marginBottom: 14, overflow: "hidden" }}>
              <div style={{ padding: "8px 16px", borderBottom: `1px solid ${t.border}`, fontSize: 13, fontWeight: 700, color: t.accent }}>📡 LIVE TRACKING ({tracked.length}) <span style={{ color: t.textMuted, fontWeight: 500, fontSize: 11, marginLeft: 10 }}>Updates every 5 min</span></div>
              <AnimatePresence>
                {tracked.map((tr: any, i: number) => <TrackedTrade key={`${tr.symbol}-${tr.pattern_name}`} trade={tr} price={prices[tr.symbol]} t={t} onRemove={() => setTracked(p => p.filter((_: any, idx: number) => idx !== i))} />)}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>

        {/* OPPORTUNITIES */}
        <AnimatePresence mode="wait">
          {view === "opp" && (
            <motion.div key="opp" initial="hidden" animate="visible" exit="exit" variants={fadeUp}>
              {mktSummary && <motion.div variants={fadeUp} style={{ padding: "10px 16px", background: t.bgCard, borderRadius: 12, border: `1px solid ${t.border}`, marginBottom: 12, fontSize: 14, color: t.textDim }}><span style={{ fontWeight: 700, color: t.text }}>Market: </span>{mktSummary}</motion.div>}
              {inPlay.length > 0 && <motion.div variants={fadeUp} style={{ marginBottom: 14 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: t.textMuted, marginBottom: 6 }}>TRENDING</div>
                <div style={{ display: "flex", gap: 8, overflowX: "auto", paddingBottom: 6 }}>{inPlay.map((s: any) => <InPlayCard key={s.symbol} stock={s} onClick={() => scanSym(s.symbol)} t={t} />)}</div>
              </motion.div>}
              {hotStrats.length > 0 && <motion.div variants={fadeUp} style={{ display: "flex", gap: 6, marginBottom: 12, flexWrap: "wrap", alignItems: "center" }}>
                <span style={{ fontSize: 12, fontWeight: 700, color: t.textMuted }}>🔥 HOT:</span>
                {hotStrats.map((s: any) => <span key={s.name} style={{ fontSize: 11, padding: "3px 10px", borderRadius: 6, background: t.bgCard, border: `1px solid ${t.border}`, color: t.textDim }}><span style={{ fontWeight: 700, color: t.text }}>{s.name}</span> <span style={{ color: t.long }}>{(s.win_rate * 100).toFixed(0)}%</span></span>)}
              </motion.div>}
              {topLoading && <JuicerLoader t={t} msg="Scanning trending tickers for opportunities..." />}
            </motion.div>
          )}
        </AnimatePresence>

        {/* TRACKER */}
        <AnimatePresence mode="wait">
          {view === "tracker" && (
            <motion.div key="tracker" initial="hidden" animate="visible" exit="exit" variants={fadeUp}>
              <TradeTrackerView t={t} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* ANALYZE */}
        <AnimatePresence mode="wait">
          {view === "analyze" && (
            <motion.div key="analyze" initial="hidden" animate="visible" exit="exit" variants={fadeUp}>
              <AnalyzeView t={t} onTrack={addTrack} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* SCAN */}
        <AnimatePresence mode="wait">
          {view === "scan" && (
            <motion.div key="scan" initial="hidden" animate="visible" variants={fadeUp}
              style={{ display: "flex", gap: 10, alignItems: "end", marginBottom: 14, flexWrap: "wrap" }}>
              <div><label style={{ fontSize: 10, color: t.textMuted, display: "block", marginBottom: 3, fontWeight: 700 }}>SYMBOL</label>
                <input type="text" value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} onKeyDown={e => e.key === "Enter" && handleScan()} placeholder="AAPL"
                  style={{ fontSize: 16, padding: "8px 12px", borderRadius: 10, width: 110, border: `1.5px solid ${t.border}`, fontWeight: 800, background: t.bgCard, color: t.text, fontFamily: "'Outfit',sans-serif" }} />
              </div>
              <div style={{ display: "flex", gap: 2, background: t.border, borderRadius: 8, padding: 3 }}>
                <Pill active={mode === "today"} onClick={() => setMode("today")} t={t} s>Today</Pill>
                <Pill active={mode === "active"} onClick={() => setMode("active")} t={t} s>Active</Pill>
              </div>
              <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}
                onClick={handleScan} disabled={loading || !symbol}
                style={{ fontSize: 14, fontWeight: 700, padding: "9px 24px", borderRadius: 10, border: "none", background: loading ? t.border : `linear-gradient(135deg, ${t.accent}, ${t.accentLight})`, color: "#fff", cursor: loading ? "wait" : "pointer" }}>
                {loading ? "Scanning..." : "Squeeze 🧃"}
              </motion.button>
              <span style={{ fontSize: 12, color: t.textMuted }}>{pc} patterns • AI</span>
            </motion.div>
          )}
        </AnimatePresence>

        {error && <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ padding: "10px 14px", borderRadius: 10, background: t.shortBg, color: t.short, fontSize: 13, marginBottom: 12, border: `1px solid ${t.short}25` }}>{error}</motion.div>}
        {loading && view === "scan" && <JuicerLoader t={t} msg={`Squeezing ${symbol}...`} />}

        {/* RESULTS */}
        {!loading && active.length > 0 && !(view === "opp" && topLoading) && view !== "tracker" && (
          <motion.div initial="hidden" animate="visible" variants={stagger}>
            <div style={{ display: "flex", gap: 6, marginBottom: 10, flexWrap: "wrap", alignItems: "center" }}>
              <div style={{ display: "flex", gap: 2, background: t.border, borderRadius: 8, padding: 3 }}>{["ALL", "LONG", "SHORT"].map(b => <Pill key={b} active={fBias === b} onClick={() => setFBias(b)} t={t} s>{b}</Pill>)}</div>
              <div style={{ display: "flex", gap: 2, background: t.border, borderRadius: 8, padding: 3 }}>{[["ALL", "All"], ["classical", "Classical"], ["candlestick", "Candle"], ["smb_scalp", "SMB"], ["quant", "Quant"]].map(([v, l]) => <Pill key={v} active={fCat === v} onClick={() => setFCat(v)} t={t} s>{l}</Pill>)}</div>
              <div style={{ marginLeft: "auto", display: "flex", gap: 2, background: t.border, borderRadius: 8, padding: 3 }}>{[["score", "Score"], ["rr", "R:R"]].map(([k, l]) => <Pill key={k} active={sortBy === k} onClick={() => setSortBy(k as any)} t={t} s>{l}</Pill>)}</div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 16px", borderBottom: `2px solid ${t.borderLight}`, fontSize: 10, fontWeight: 700, color: t.textMuted, letterSpacing: 0.6, textTransform: "uppercase" }}>
              <div style={{ width: 70 }}>Ticker</div><div style={{ flex: 1 }}>Setup / Trigger / Analysis</div>
              <div style={{ display: "flex", gap: 12 }}><div style={{ width: 58, textAlign: "right" }}>Entry</div><div style={{ width: 58, textAlign: "right" }}>Stop</div><div style={{ width: 58, textAlign: "right" }}>Target</div></div>
              <div style={{ width: 42, textAlign: "center" }}>R:R</div><div style={{ width: 42, textAlign: "center" }}>Score</div><div style={{ width: 16 }} /><div style={{ width: 75 }} />
            </div>
            <AnimatePresence>
              {filtered.map((s: any, i: number) => <SetupRow key={`${s.symbol}-${s.pattern_name}-${i}`} s={s} open={chartIdx === i} toggle={() => setChartIdx(chartIdx === i ? null : i)} t={t} onTrack={addTrack} />)}
            </AnimatePresence>
            {filtered.length === 0 && <div style={{ textAlign: "center", padding: 30, color: t.textDim, fontSize: 14 }}>No setups match filters.</div>}
          </motion.div>
        )}

        {!loading && view === "scan" && scanSetups.length === 0 && !error && (
          <motion.div initial="hidden" animate="visible" variants={fadeUp} style={{ textAlign: "center", padding: 50 }}>
            <Logo size={60} />
            <div style={{ fontSize: 20, fontWeight: 700, marginTop: 16, color: t.text }}>Squeeze a ticker</div>
            <div style={{ fontSize: 14, color: t.textDim, marginTop: 6 }}>{pc} patterns • 6-factor scoring • AI evaluation</div>
          </motion.div>
        )}

        <div style={{ textAlign: "center", marginTop: 28, padding: "12px 0", borderTop: `1px solid ${t.border}` }}>
          <span style={{ fontSize: 11, color: t.textMuted }}>Juicer v1.0 — {pc} Detectors — Squeeze the Market 🧃</span>
        </div>
      </div>

      <BacktestModal open={btOpen} onClose={() => setBtOpen(false)} t={t} />
    </div>
  );
}