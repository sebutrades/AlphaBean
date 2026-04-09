import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";

const WS_URL = "ws://localhost:8000/ws/agent-trading";
const API = "http://localhost:8000";

type Theme = {
  bg: string; bgCard: string; bgHover: string; border: string; borderLight: string;
  text: string; textDim: string; textMuted: string;
  accent: string; accentLight: string;
  long: string; longBg: string; short: string; shortBg: string;
  gold: string; goldBg: string; purple: string;
  chartBg: string; chartGrid: string; chartText: string;
};

// ── Types for long-term results ────────────────────────────────────────────

type SimResults = {
  config: { starting_capital: number; risk_per_trade_pct: number; sim_days: number; universe_size: number; use_agents: boolean };
  stats: { total_trades: number; wins: number; losses: number; total_r: number; win_rate: number; avg_r: number; profit_factor: number; best_trade_r: number; worst_trade_r: number; open_positions: number; current_heat: number };
  equity_curve: { date: string; equity: number; cumulative_r: number; day_r: number; heat: number; daily_pnl: number }[];
  closed_trades: { symbol: string; pattern: string; strategy_type: string; bias: string; entry: number; stop: number; shares: number; dollar_risk: number; entry_date: string; exit_date: string; outcome: string; realized_r: number; bars_held: number; cum_pnl: number; cum_r: number; pnl: number }[];
};

// ── Multi-day Equity Chart (pure SVG) ──────────────────────────────────────

function EquityCurveChart({ curve, startCap, t }: { curve: SimResults["equity_curve"]; startCap: number; t: Theme }) {
  if (curve.length < 2) return null;

  const W = 800, H = 220, PAD = 55, PADR = 20, PADT = 20, PADB = 30;
  const values = curve.map(p => p.equity);
  const minV = Math.min(...values) * 0.995;
  const maxV = Math.max(...values) * 1.005;
  const range = maxV - minV || 1;

  const toX = (i: number) => PAD + (i / (curve.length - 1)) * (W - PAD - PADR);
  const toY = (v: number) => PADT + ((maxV - v) / range) * (H - PADT - PADB);
  const startY = toY(startCap);

  const pathParts = curve.map((p, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(p.equity).toFixed(1)}`);
  const linePath = pathParts.join(" ");
  const fillPath = linePath + ` L${toX(curve.length - 1).toFixed(1)},${H - PADB} L${toX(0).toFixed(1)},${H - PADB} Z`;

  const lastEq = values[values.length - 1];
  const color = lastEq >= startCap ? t.long : t.short;

  // Y-axis labels (5 ticks)
  const yTicks = Array.from({ length: 5 }, (_, i) => minV + (range * i) / 4);
  // X-axis labels (every ~15 days)
  const step = Math.max(1, Math.floor(curve.length / 6));
  const xLabels = curve.filter((_, i) => i % step === 0 || i === curve.length - 1).map((p, _, arr) => ({
    x: toX(curve.indexOf(p)),
    text: p.date.slice(5), // MM-DD
  }));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 220 }}>
      {/* Grid */}
      {yTicks.map((v, i) => (
        <line key={i} x1={PAD} y1={toY(v)} x2={W - PADR} y2={toY(v)} stroke={t.chartGrid} strokeWidth={0.5} />
      ))}
      {/* Starting capital line */}
      <line x1={PAD} y1={startY} x2={W - PADR} y2={startY} stroke={t.textMuted} strokeWidth={1} strokeDasharray="4,3" />
      <text x={PAD - 5} y={startY} textAnchor="end" fill={t.textMuted} fontSize={9} dominantBaseline="middle">Start</text>
      {/* Fill */}
      <path d={fillPath} fill={color} opacity={0.06} />
      {/* Line */}
      <path d={linePath} fill="none" stroke={color} strokeWidth={2} />
      {/* End dot */}
      <circle cx={toX(curve.length - 1)} cy={toY(lastEq)} r={4} fill={color} />
      {/* End label */}
      <text x={toX(curve.length - 1) + 8} y={toY(lastEq)} fill={color} fontSize={10} fontWeight={700} dominantBaseline="middle">
        ${lastEq.toLocaleString(undefined, { maximumFractionDigits: 0 })}
      </text>
      {/* Y-axis */}
      {yTicks.map((v, i) => (
        <text key={i} x={PAD - 5} y={toY(v)} textAnchor="end" fill={t.chartText} fontSize={9} dominantBaseline="middle">
          ${(v / 1000).toFixed(1)}k
        </text>
      ))}
      {/* X-axis */}
      {xLabels.map((l, i) => (
        <text key={i} x={l.x} y={H - 8} textAnchor="middle" fill={t.chartText} fontSize={9}>{l.text}</text>
      ))}
    </svg>
  );
}

// ── Daily P&L Bar Chart ────────────────────────────────────────────────────

function DailyPnlChart({ curve, t }: { curve: SimResults["equity_curve"]; t: Theme }) {
  if (curve.length < 2) return null;

  const W = 800, H = 140, PAD = 55, PADR = 20, PADT = 15, PADB = 25;
  const pnls = curve.map(p => p.daily_pnl);
  const maxAbs = Math.max(1, ...pnls.map(Math.abs));
  const barW = Math.max(1, (W - PAD - PADR) / curve.length - 1);

  const toX = (i: number) => PAD + (i / curve.length) * (W - PAD - PADR);
  const zeroY = PADT + (H - PADT - PADB) / 2;
  const scale = (H - PADT - PADB) / 2 / maxAbs;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 140 }}>
      {/* Zero line */}
      <line x1={PAD} y1={zeroY} x2={W - PADR} y2={zeroY} stroke={t.textMuted} strokeWidth={0.5} />
      {/* Bars */}
      {pnls.map((pnl, i) => {
        const h = Math.abs(pnl) * scale;
        const y = pnl >= 0 ? zeroY - h : zeroY;
        return (
          <rect key={i} x={toX(i)} y={y} width={barW} height={Math.max(0.5, h)}
            fill={pnl >= 0 ? t.long : t.short} opacity={0.7} rx={0.5}
          />
        );
      })}
      {/* Y labels */}
      <text x={PAD - 5} y={PADT + 5} textAnchor="end" fill={t.chartText} fontSize={9}>+${(maxAbs / 1000).toFixed(1)}k</text>
      <text x={PAD - 5} y={zeroY} textAnchor="end" fill={t.chartText} fontSize={9} dominantBaseline="middle">$0</text>
      <text x={PAD - 5} y={H - PADB - 2} textAnchor="end" fill={t.chartText} fontSize={9}>-${(maxAbs / 1000).toFixed(1)}k</text>
      {/* X labels */}
      {curve.filter((_, i) => i % Math.max(1, Math.floor(curve.length / 6)) === 0).map((p, i) => (
        <text key={i} x={toX(curve.indexOf(p)) + barW / 2} y={H - 5} textAnchor="middle" fill={t.chartText} fontSize={8}>{p.date.slice(5)}</text>
      ))}
    </svg>
  );
}

// ── Cumulative R Chart ─────────────────────────────────────────────────────

function CumulativeRChart({ curve, t }: { curve: SimResults["equity_curve"]; t: Theme }) {
  if (curve.length < 2) return null;

  const W = 800, H = 160, PAD = 55, PADR = 20, PADT = 15, PADB = 25;
  const values = curve.map(p => p.cumulative_r);
  const minV = Math.min(0, ...values);
  const maxV = Math.max(0.1, ...values);
  const range = maxV - minV || 1;

  const toX = (i: number) => PAD + (i / (curve.length - 1)) * (W - PAD - PADR);
  const toY = (v: number) => PADT + ((maxV - v) / range) * (H - PADT - PADB);
  const zeroY = toY(0);

  const pathParts = curve.map((p, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(p.cumulative_r).toFixed(1)}`);
  const linePath = pathParts.join(" ");
  const fillPath = linePath + ` L${toX(curve.length - 1).toFixed(1)},${zeroY.toFixed(1)} L${toX(0).toFixed(1)},${zeroY.toFixed(1)} Z`;

  const lastR = values[values.length - 1];
  const color = lastR >= 0 ? t.long : t.short;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 160 }}>
      {/* Zero line */}
      <line x1={PAD} y1={zeroY} x2={W - PADR} y2={zeroY} stroke={t.textMuted} strokeWidth={1} strokeDasharray="4,3" />
      <path d={fillPath} fill={color} opacity={0.06} />
      <path d={linePath} fill="none" stroke={color} strokeWidth={2} />
      <circle cx={toX(curve.length - 1)} cy={toY(lastR)} r={4} fill={color} />
      {/* Y labels */}
      {[minV, minV + range / 2, maxV].map((v, i) => (
        <text key={i} x={PAD - 5} y={toY(v)} textAnchor="end" fill={t.chartText} fontSize={9} dominantBaseline="middle">
          {v.toFixed(1)}R
        </text>
      ))}
      {/* X labels */}
      {curve.filter((_, i) => i % Math.max(1, Math.floor(curve.length / 6)) === 0).map((p, i) => (
        <text key={i} x={toX(curve.indexOf(p))} y={H - 5} textAnchor="middle" fill={t.chartText} fontSize={8}>{p.date.slice(5)}</text>
      ))}
    </svg>
  );
}

// ── Pattern Performance Table ──────────────────────────────────────────────

function PatternBreakdown({ trades, t }: { trades: SimResults["closed_trades"]; t: Theme }) {
  const byPattern: Record<string, { wins: number; losses: number; totalR: number; count: number; totalPnl: number }> = {};
  for (const tr of trades) {
    const key = tr.pattern;
    if (!byPattern[key]) byPattern[key] = { wins: 0, losses: 0, totalR: 0, count: 0, totalPnl: 0 };
    byPattern[key].count++;
    byPattern[key].totalR += tr.realized_r;
    byPattern[key].totalPnl += tr.pnl;
    if (tr.outcome === "win") byPattern[key].wins++;
    else byPattern[key].losses++;
  }

  const sorted = Object.entries(byPattern).sort((a, b) => b[1].totalR - a[1].totalR);

  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
      <thead>
        <tr style={{ borderBottom: `1px solid ${t.border}` }}>
          {["Pattern", "Trades", "W/L", "Win%", "Total R", "P&L", "Avg R"].map(h => (
            <th key={h} style={{ padding: "6px 8px", textAlign: "left", fontSize: 9, color: t.textMuted, fontWeight: 600 }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {sorted.map(([pattern, s]) => {
          const winRate = s.count > 0 ? (s.wins / s.count) * 100 : 0;
          const avgR = s.count > 0 ? s.totalR / s.count : 0;
          return (
            <tr key={pattern} style={{ borderBottom: `1px solid ${t.border}` }}>
              <td style={{ padding: "5px 8px", fontWeight: 600, color: t.text, fontSize: 10 }}>{pattern}</td>
              <td style={{ padding: "5px 8px", color: t.textDim }}>{s.count}</td>
              <td style={{ padding: "5px 8px", color: t.textDim }}>{s.wins}/{s.losses}</td>
              <td style={{ padding: "5px 8px", color: winRate >= 40 ? t.long : winRate >= 25 ? t.gold : t.short, fontWeight: 600 }}>{winRate.toFixed(0)}%</td>
              <td style={{ padding: "5px 8px", color: s.totalR >= 0 ? t.long : t.short, fontWeight: 700 }}>{s.totalR >= 0 ? "+" : ""}{s.totalR.toFixed(2)}R</td>
              <td style={{ padding: "5px 8px", color: s.totalPnl >= 0 ? t.long : t.short }}>${s.totalPnl.toFixed(0)}</td>
              <td style={{ padding: "5px 8px", color: avgR >= 0 ? t.long : t.short }}>{avgR >= 0 ? "+" : ""}{avgR.toFixed(2)}R</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ── Long-Term Performance Dashboard ────────────────────────────────────────

function LongTermView({ t }: { t: Theme }) {
  const [results, setResults] = useState<SimResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    setLoading(true);
    fetch(`${API}/api/agent-trading/results`)
      .then(r => r.json())
      .then(d => {
        if (d.error) setError(d.error);
        else setResults(d);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ color: t.textMuted, textAlign: "center", padding: 40 }}>Loading simulation results...</div>;
  if (error) return <div style={{ color: t.short, textAlign: "center", padding: 40 }}>{error}</div>;
  if (!results) return null;

  const { config, stats, equity_curve, closed_trades } = results;
  const startCap = config.starting_capital;
  const endEq = equity_curve.length ? equity_curve[equity_curve.length - 1].equity : startCap;
  const totalReturn = ((endEq - startCap) / startCap) * 100;
  const maxEq = Math.max(...equity_curve.map(p => p.equity));
  const minEq = Math.min(...equity_curve.map(p => p.equity));
  const maxDD = ((maxEq - minEq) / maxEq) * 100;

  // Compute max drawdown properly
  let peak = startCap;
  let worstDD = 0;
  for (const pt of equity_curve) {
    if (pt.equity > peak) peak = pt.equity;
    const dd = (peak - pt.equity) / peak;
    if (dd > worstDD) worstDD = dd;
  }

  // Win streak / loss streak
  let maxWinStreak = 0, maxLossStreak = 0, curWin = 0, curLoss = 0;
  for (const tr of closed_trades) {
    if (tr.outcome === "win") { curWin++; curLoss = 0; maxWinStreak = Math.max(maxWinStreak, curWin); }
    else { curLoss++; curWin = 0; maxLossStreak = Math.max(maxLossStreak, curLoss); }
  }

  const statCards = [
    { label: "FINAL EQUITY", value: `$${endEq.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, color: endEq >= startCap ? t.long : t.short },
    { label: "RETURN", value: `${totalReturn >= 0 ? "+" : ""}${totalReturn.toFixed(1)}%`, color: totalReturn >= 0 ? t.long : t.short },
    { label: "TOTAL R", value: `${stats.total_r >= 0 ? "+" : ""}${stats.total_r.toFixed(1)}R`, color: stats.total_r >= 0 ? t.long : t.short },
    { label: "TRADES", value: `${stats.total_trades}`, color: t.text },
    { label: "WIN RATE", value: `${stats.win_rate.toFixed(1)}%`, color: stats.win_rate >= 40 ? t.long : stats.win_rate >= 25 ? t.gold : t.short },
    { label: "W / L", value: `${stats.wins} / ${stats.losses}`, color: stats.wins > stats.losses ? t.long : t.short },
    { label: "AVG R", value: `${stats.avg_r >= 0 ? "+" : ""}${stats.avg_r.toFixed(3)}R`, color: stats.avg_r >= 0 ? t.long : t.short },
    { label: "MAX DD", value: `${(worstDD * 100).toFixed(1)}%`, color: t.short },
    { label: "BEST", value: `+${stats.best_trade_r.toFixed(2)}R`, color: t.long },
    { label: "WORST", value: `${stats.worst_trade_r.toFixed(2)}R`, color: t.short },
    { label: "WIN STREAK", value: `${maxWinStreak}`, color: t.long },
    { label: "LOSS STREAK", value: `${maxLossStreak}`, color: t.short },
  ];

  return (
    <div>
      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "12px 16px", background: t.bgCard, borderRadius: 12,
        border: `1px solid ${t.border}`, marginBottom: 12,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 14, fontWeight: 800, color: t.accent }}>SIMULATION RESULTS</span>
          <span style={{ fontSize: 11, color: t.textMuted }}>
            {equity_curve.length} days | {equity_curve[0]?.date} to {equity_curve[equity_curve.length - 1]?.date}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 10, color: t.textMuted }}>
            ${startCap.toLocaleString()} start | {(config as any).mode === "deterministic_plus" ? "Det+" : config.use_agents ? "Agent-driven" : "Deterministic"} | {config.risk_per_trade_pct}% risk
          </span>
        </div>
      </div>

      {/* Stats Grid */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 8, marginBottom: 12,
      }}>
        {statCards.map((s, i) => (
          <div key={i} style={{
            background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 8,
            padding: "8px 10px", textAlign: "center",
          }}>
            <div style={{ fontSize: 8, color: t.textMuted, fontWeight: 600, marginBottom: 3, letterSpacing: 0.5 }}>{s.label}</div>
            <div style={{ fontSize: 15, fontWeight: 800, color: s.color, fontFamily: "monospace" }}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Equity Curve */}
      <div style={{
        background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 12,
        padding: 16, marginBottom: 12,
      }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: t.textDim, marginBottom: 8 }}>EQUITY CURVE</div>
        <EquityCurveChart curve={equity_curve} startCap={startCap} t={t} />
      </div>

      {/* Two-column: Daily P&L + Cumulative R */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
        <div style={{
          background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 12, padding: 16,
        }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: t.textDim, marginBottom: 8 }}>DAILY P&L</div>
          <DailyPnlChart curve={equity_curve} t={t} />
        </div>
        <div style={{
          background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 12, padding: 16,
        }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: t.textDim, marginBottom: 8 }}>CUMULATIVE R</div>
          <CumulativeRChart curve={equity_curve} t={t} />
        </div>
      </div>

      {/* Pattern Breakdown */}
      <div style={{
        background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 12,
        padding: 16, marginBottom: 12,
      }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: t.textDim, marginBottom: 8 }}>
          PATTERN PERFORMANCE ({Object.keys(closed_trades.reduce((acc, tr) => ({ ...acc, [tr.pattern]: 1 }), {})).length} patterns)
        </div>
        <div style={{ maxHeight: 400, overflow: "auto" }}>
          <PatternBreakdown trades={closed_trades} t={t} />
        </div>
      </div>

      {/* Trade Log */}
      <div style={{
        background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 12,
        padding: 16,
      }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: t.textDim, marginBottom: 8 }}>
          TRADE LOG ({closed_trades.length} trades)
        </div>
        <div style={{ maxHeight: 400, overflow: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${t.border}`, position: "sticky", top: 0, background: t.bgCard }}>
                {["#", "Symbol", "Pattern", "Bias", "Entry", "Outcome", "R", "P&L", "Cum P&L", "Bars", "Date"].map(h => (
                  <th key={h} style={{ padding: "5px 6px", textAlign: "left", fontSize: 8, color: t.textMuted, fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {closed_trades.map((tr, i) => (
                <tr key={i} style={{ borderBottom: `1px solid ${t.border}` }}>
                  <td style={{ padding: "4px 6px", color: t.textMuted }}>{i + 1}</td>
                  <td style={{ padding: "4px 6px", fontWeight: 700, color: t.text }}>{tr.symbol}</td>
                  <td style={{ padding: "4px 6px", color: t.textDim, fontSize: 9 }}>{tr.pattern.slice(0, 25)}</td>
                  <td style={{ padding: "4px 6px", color: tr.bias === "long" ? t.long : t.short, fontWeight: 600, fontSize: 9 }}>{tr.bias.toUpperCase()}</td>
                  <td style={{ padding: "4px 6px" }}>${tr.entry.toFixed(2)}</td>
                  <td style={{ padding: "4px 6px", fontWeight: 700, color: tr.outcome === "win" ? t.long : t.short }}>{tr.outcome.toUpperCase()}</td>
                  <td style={{ padding: "4px 6px", fontWeight: 700, color: tr.realized_r > 0 ? t.long : t.short }}>
                    {tr.realized_r > 0 ? "+" : ""}{tr.realized_r.toFixed(2)}R
                  </td>
                  <td style={{ padding: "4px 6px", color: tr.pnl > 0 ? t.long : t.short }}>${tr.pnl.toFixed(0)}</td>
                  <td style={{ padding: "4px 6px", color: tr.cum_pnl >= 0 ? t.long : t.short, fontWeight: 600 }}>${tr.cum_pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                  <td style={{ padding: "4px 6px", color: t.textMuted }}>{tr.bars_held}</td>
                  <td style={{ padding: "4px 6px", color: t.textMuted, fontSize: 9 }}>{tr.exit_date}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

type AgentEvent = {
  type: string;
  timestamp: string;
  data: Record<string, any>;
};

type PnlPoint = {
  bar_index: number;
  total_bars: number;
  time: string;
  equity: number;
  cumulative_r: number;
  unrealized_r: number;
  total_r: number;
  cumulative_pnl: number;
  unrealized_pnl: number;
  total_pnl: number;
  positions: number;
  closed_trades: number;
  heat_pct: number;
};

type Position = {
  id: string;
  symbol: string;
  pattern: string;
  bias: string;
  entry: number;
  stop: number;
  t1: number;
  t2: number;
  current_price: number;
  unrealized_r: number;
  high_water_r: number;
  bars_held: number;
  t1_hit: boolean;
  t2_hit: boolean;
  remaining_weight: number;
  verdict: string;
  reasoning: string;
  dollar_risk: number;
  shares: number;
  score: number;
};

type ClosedTrade = {
  id: string;
  symbol: string;
  pattern: string;
  bias: string;
  outcome: string;
  realized_r: number;
  pnl: number;
  entry: number;
  exit: number;
  bars_held: number;
  entry_time: string;
  exit_time: string;
};

// ── PNL Chart (pure SVG) ────────────────────────────────────────────────────

function PnlChart({ points, t }: { points: PnlPoint[]; t: Theme }) {
  if (points.length < 2) {
    return (
      <div style={{ height: 200, display: "flex", alignItems: "center", justifyContent: "center", color: t.textMuted, fontSize: 13 }}>
        Waiting for data...
      </div>
    );
  }

  const W = 700, H = 180, PAD = 40;
  const values = points.map(p => p.total_pnl ?? 0);
  const minV = Math.min(0, ...values);
  const maxV = Math.max(1, ...values);
  const range = maxV - minV || 1;

  const toX = (i: number) => PAD + (i / (points.length - 1)) * (W - PAD * 2);
  const toY = (v: number) => H - PAD - ((v - minV) / range) * (H - PAD * 2);
  const zeroY = toY(0);

  // Build path — sample to avoid sluggish SVG with thousands of points
  const maxPts = 500;
  const step = points.length > maxPts ? Math.floor(points.length / maxPts) : 1;
  const sampled = points.filter((_, i) => i % step === 0 || i === points.length - 1);
  const pathParts = sampled.map((p, i) => {
    const origIdx = points.indexOf(p);
    const val = p.total_pnl ?? 0;
    return `${i === 0 ? "M" : "L"}${toX(origIdx).toFixed(1)},${toY(val).toFixed(1)}`;
  });
  const linePath = pathParts.join(" ");

  // Fill area
  const lastOrigIdx = points.length - 1;
  const fillPath = linePath + ` L${toX(lastOrigIdx).toFixed(1)},${zeroY.toFixed(1)} L${toX(0).toFixed(1)},${zeroY.toFixed(1)} Z`;

  const lastVal = values[values.length - 1];
  const color = lastVal >= 0 ? t.long : t.short;

  // Time labels — show date transitions for multi-day
  const labels: { x: number; text: string }[] = [];
  const labelStep = Math.max(1, Math.floor(points.length / 8));
  for (let i = 0; i < points.length; i += labelStep) {
    const raw = points[i].time;
    const dateStr = raw.slice(5, 10);
    const timeStr = raw.slice(11, 16);
    labels.push({ x: toX(i), text: `${dateStr} ${timeStr}` });
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 200 }}>
      {/* Grid lines */}
      {[0.25, 0.5, 0.75].map(f => {
        const y = PAD + f * (H - PAD * 2);
        return <line key={f} x1={PAD} y1={y} x2={W - PAD} y2={y} stroke={t.chartGrid} strokeWidth={0.5} />;
      })}
      {/* Zero line */}
      <line x1={PAD} y1={zeroY} x2={W - PAD} y2={zeroY} stroke={t.textMuted} strokeWidth={1} strokeDasharray="4,3" />
      {/* Fill */}
      <path d={fillPath} fill={color} opacity={0.08} />
      {/* Line */}
      <path d={linePath} fill="none" stroke={color} strokeWidth={1.5} />
      {/* Current dot */}
      <circle cx={toX(lastOrigIdx)} cy={toY(lastVal)} r={4} fill={color} />
      {/* Y-axis labels */}
      {[minV, minV + range / 2, maxV].map((v, i) => (
        <text key={i} x={PAD - 5} y={toY(v)} textAnchor="end" fill={t.chartText} fontSize={10} dominantBaseline="middle">
          {v >= 0 ? "+" : ""}${Math.abs(v) >= 1000 ? `${(v/1000).toFixed(1)}k` : v.toFixed(0)}
        </text>
      ))}
      {/* X-axis labels */}
      {labels.map((l, i) => (
        <text key={i} x={l.x} y={H - 5} textAnchor="middle" fill={t.chartText} fontSize={8}>{l.text}</text>
      ))}
    </svg>
  );
}

// ── Agent Log Entry ─────────────────────────────────────────────────────────

function LogEntry({ event, t }: { event: AgentEvent; t: Theme }) {
  const time = event.timestamp ? `${event.timestamp.slice(5, 10)} ${event.timestamp.slice(11, 16)}` : "";
  const d = event.data;

  const colors: Record<string, string> = {
    setup_detected: t.gold,
    agent_thinking: t.purple,
    agent_verdict: d.verdict === "CONFIRMED" ? t.long : d.verdict === "DENIED" ? t.short : t.gold,
    trade_open: t.long,
    trade_close: (d.realized_r ?? 0) > 0 ? t.long : t.short,
    position_update: t.textMuted,
    multi_day_start: t.accent,
    multi_day_end: t.accent,
    error: t.short,
  };

  const icons: Record<string, string> = {
    setup_detected: ">>",
    agent_thinking: "...",
    agent_verdict: d.verdict === "CONFIRMED" ? "++" : d.verdict === "DENIED" ? "XX" : "??",
    trade_open: "->",
    trade_close: "<-",
    position_update: "--",
    bar: "|",
    pnl: "$",
    day_start: "##",
    day_end: "##",
    multi_day_start: "**",
    multi_day_end: "**",
    error: "!!",
  };

  let message = "";
  switch (event.type) {
    case "setup_detected":
      message = `SETUP: ${d.pattern} on ${d.symbol} | ${d.bias?.toUpperCase()} | Score: ${d.score} | R:R ${d.rr}`;
      break;
    case "agent_thinking":
      message = d.message || `Evaluating ${d.pattern} on ${d.symbol}...`;
      break;
    case "agent_verdict":
      message = `[${d.verdict}] ${d.symbol} ${d.pattern} | ${d.action || ""} ${d.reasoning ? "- " + d.reasoning.slice(0, 200) : ""}`;
      break;
    case "trade_open":
      message = `OPEN ${d.id}: ${d.bias?.toUpperCase()} ${d.shares} x ${d.symbol} @ $${d.entry?.toFixed(2)} | Stop: $${d.stop?.toFixed(2)} | Risk: $${d.dollar_risk}`;
      break;
    case "trade_close":
      message = `CLOSE ${d.id}: ${d.symbol} ${d.outcome?.toUpperCase()} ${d.realized_r > 0 ? "+" : ""}${d.realized_r?.toFixed(2)}R ($${d.pnl?.toFixed(0)}) | Held ${d.bars_held} bars`;
      break;
    case "day_start":
      message = `=== DAY ${d.day_number || ""}: ${d.date} | ${d.symbols} symbols | ${d.total_bars} bars | ${d.open_positions || 0} open ===`;
      break;
    case "day_end":
      message = `=== END ${d.date || ""} | Trades: ${d.total_trades} | R: ${d.cumulative_r?.toFixed(2)} | W:${d.wins} L:${d.losses} | Equity: $${d.final_equity?.toLocaleString()} ===`;
      break;
    case "multi_day_start":
      message = `=== CONTINUOUS RUN: ${d.total_days} days | $${d.starting_capital?.toLocaleString()} starting capital ===`;
      break;
    case "multi_day_end":
      message = `=== RUN COMPLETE: ${d.total_days} days | ${d.total_trades} trades | ${d.cumulative_r?.toFixed(2)}R | Final: $${d.final_equity?.toLocaleString()} ===`;
      break;
    default:
      return null;
  }

  const c = colors[event.type] || t.textDim;

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      style={{
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 11,
        padding: "3px 8px",
        color: c,
        borderLeft: `2px solid ${c}30`,
        marginBottom: 1,
        lineHeight: 1.5,
      }}
    >
      <span style={{ color: t.textMuted, marginRight: 6 }}>{time}</span>
      <span style={{ color: c, marginRight: 6, fontWeight: 700 }}>{icons[event.type] || ">>"}</span>
      {message}
    </motion.div>
  );
}

// ── Position Row ────────────────────────────────────────────────────────────

function PositionRow({ pos, t }: { pos: Position; t: Theme }) {
  const rColor = pos.unrealized_r >= 0 ? t.long : t.short;
  const biasColor = pos.bias === "long" ? t.long : t.short;

  return (
    <tr style={{ borderBottom: `1px solid ${t.border}` }}>
      <td style={{ padding: "6px 8px", fontWeight: 700, color: biasColor }}>{pos.symbol}</td>
      <td style={{ padding: "6px 8px", color: t.textDim, fontSize: 10 }}>{pos.pattern.slice(0, 20)}</td>
      <td style={{ padding: "6px 8px", color: biasColor, fontSize: 10, fontWeight: 600 }}>{pos.bias.toUpperCase()}</td>
      <td style={{ padding: "6px 8px", fontFamily: "monospace", fontSize: 11 }}>${pos.entry?.toFixed(2)}</td>
      <td style={{ padding: "6px 8px", fontFamily: "monospace", fontSize: 11, color: t.short }}>${pos.stop?.toFixed(2)}</td>
      <td style={{ padding: "6px 8px", fontFamily: "monospace", fontSize: 11 }}>${pos.current_price?.toFixed(2)}</td>
      <td style={{ padding: "6px 8px", fontFamily: "monospace", fontSize: 11, fontWeight: 700, color: rColor }}>
        {pos.unrealized_r >= 0 ? "+" : ""}{pos.unrealized_r?.toFixed(2)}R
      </td>
      <td style={{ padding: "6px 8px", fontSize: 10, color: t.textMuted }}>
        {pos.t1_hit ? "T1" : ""}{pos.t2_hit ? " T2" : ""} {pos.bars_held}b
      </td>
    </tr>
  );
}

// ── Main Component ──────────────────────────────────────────────────────────

export default function AgentTradingView({ t }: { t: Theme }) {
  const [mode, setMode] = useState<"single" | "continuous">("continuous");
  const [dates, setDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [speed, setSpeed] = useState(10);
  const [engine, setEngine] = useState<"standard" | "agents" | "det_plus">("standard");
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [paused, setPaused] = useState(false);
  const [finished, setFinished] = useState(false);

  // Multi-day tracking
  const [currentDay, setCurrentDay] = useState(0);
  const [totalDays, setTotalDays] = useState(0);
  const [currentDate, setCurrentDate] = useState("");
  const isContinuousRef = useRef(false);

  // Data
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [pnlHistory, setPnlHistory] = useState<PnlPoint[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [closedTrades, setClosedTrades] = useState<ClosedTrade[]>([]);
  const [progress, setProgress] = useState({ bar: 0, total: 1, time: "" });
  const [dayStats, setDayStats] = useState<{ date: string; equity: number; day_pnl: number; trades_today: number }[]>([]);

  // Saved results for display after completion
  const [savedResults, setSavedResults] = useState<SimResults | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  // Fetch available dates
  useEffect(() => {
    fetch(`${API}/api/agent-trading/dates`)
      .then(r => r.json())
      .then(d => {
        if (d.dates?.length) {
          setDates(d.dates);
          setSelectedDate(d.dates[d.dates.length - 1]);
        }
      })
      .catch(() => {});
    // Load saved results if they exist (prefer det_plus, fallback to standard)
    Promise.all([
      fetch(`${API}/api/agent-trading/results-plus`).then(r => r.json()).catch(() => ({ error: true })),
      fetch(`${API}/api/agent-trading/results`).then(r => r.json()).catch(() => ({ error: true })),
    ]).then(([plus, standard]) => {
      if (!plus.error) setSavedResults(plus);
      else if (!standard.error) setSavedResults(standard);
    });
  }, []);

  // WebSocket connection
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => { setConnected(false); setRunning(false); };

    ws.onmessage = (msg) => {
      try {
        const event: AgentEvent = JSON.parse(msg.data);
        handleEvent(event);
      } catch {}
    };
  }, []);

  const handleEvent = useCallback((event: AgentEvent) => {
    const d = event.data;

    switch (event.type) {
      case "bar":
        setProgress({ bar: d.bar_index, total: d.total_bars, time: event.timestamp });
        break;

      case "pnl":
        setPnlHistory(prev => [...prev, d as PnlPoint]);
        break;

      case "multi_day_start":
        setTotalDays(d.total_days);
        setCurrentDay(0);
        isContinuousRef.current = true;
        setEvents(prev => [...prev.slice(-1000), event]);
        break;

      case "multi_day_end":
        setRunning(false);
        setFinished(true);
        setEvents(prev => [...prev.slice(-1000), event]);
        // Reload saved results (try det_plus first, then standard)
        Promise.all([
          fetch(`${API}/api/agent-trading/results-plus`).then(r => r.json()).catch(() => ({ error: true })),
          fetch(`${API}/api/agent-trading/results`).then(r => r.json()).catch(() => ({ error: true })),
        ]).then(([plus, standard]) => {
          if (!plus.error) setSavedResults(plus);
          else if (!standard.error) setSavedResults(standard);
        });
        break;

      case "day_start":
        setCurrentDay(d.day_number || 0);
        setCurrentDate(d.date || "");
        setEvents(prev => [...prev.slice(-1000), event]);
        break;

      case "day_end":
        setEvents(prev => [...prev.slice(-1000), event]);
        setDayStats(prev => [...prev, {
          date: d.date || "",
          equity: d.equity || 0,
          day_pnl: d.day_pnl || 0,
          trades_today: d.trades_today || 0,
        }]);
        // In single-day mode (no multi_day wrapper), stop when day ends
        if (!isContinuousRef.current) {
          setRunning(false);
          setFinished(true);
        }
        break;

      case "setup_detected":
      case "agent_thinking":
      case "agent_verdict":
      case "error":
        setEvents(prev => [...prev.slice(-1000), event]);
        break;

      case "trade_open":
        setEvents(prev => [...prev.slice(-1000), event]);
        setPositions(prev => [...prev, d as unknown as Position]);
        break;

      case "trade_close":
        setEvents(prev => [...prev.slice(-1000), event]);
        setPositions(prev => prev.filter(p => p.id !== d.id));
        setClosedTrades(prev => [...prev, d as unknown as ClosedTrade]);
        break;

      case "position_update":
        setPositions(prev =>
          prev.map(p => p.id === d.id ? { ...p, ...d } : p)
        );
        break;
    }
  }, []);

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [events]);

  const resetState = () => {
    setEvents([]);
    setPnlHistory([]);
    setPositions([]);
    setClosedTrades([]);
    setDayStats([]);
    setCurrentDay(0);
    setTotalDays(0);
    setCurrentDate("");
    setFinished(false);
    isContinuousRef.current = false;
  };

  const startSim = (continuous: boolean) => {
    resetState();

    const trySend = () => {
      const ws = wsRef.current;
      if (ws?.readyState === WebSocket.OPEN) {
        const capital = engine === "det_plus" ? 1_000_000 : 100_000;
        const minScore = engine === "det_plus" ? 50 : 50;
        if (continuous) {
          ws.send(JSON.stringify({
            action: "start_continuous",
            speed: speed,
            engine: engine === "agents" ? "standard" : engine === "det_plus" ? "det_plus" : "standard",
            use_agents: engine === "agents",
            capital: capital,
            min_score: minScore,
          }));
        } else {
          ws.send(JSON.stringify({
            action: "start",
            date: selectedDate,
            speed: speed,
            engine: engine === "agents" ? "standard" : engine === "det_plus" ? "det_plus" : "standard",
            use_agents: engine === "agents",
            capital: capital,
            min_score: minScore,
          }));
        }
        setRunning(true);
        setPaused(false);
      } else {
        setTimeout(trySend, 200);
      }
    };

    if (!connected || wsRef.current?.readyState !== WebSocket.OPEN) {
      connect();
      setTimeout(trySend, 500);
    } else {
      trySend();
    }
  };

  const send = (action: string, extra?: Record<string, any>) => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action, ...extra }));
    }
  };

  const pauseSim = () => { send("pause"); setPaused(true); };
  const resumeSim = () => { send("resume"); setPaused(false); };
  const stopSim = () => { send("stop"); setRunning(false); };
  const changeSpeed = (s: number) => { setSpeed(s); send("set_speed", { speed: s }); };

  // Live stats
  const totalR = pnlHistory.length ? pnlHistory[pnlHistory.length - 1].total_r : 0;
  const totalPnl = pnlHistory.length ? (pnlHistory[pnlHistory.length - 1] as any).total_pnl ?? 0 : 0;
  const equity = pnlHistory.length ? pnlHistory[pnlHistory.length - 1].equity : 100000;
  const heat = pnlHistory.length ? pnlHistory[pnlHistory.length - 1].heat_pct : 0;
  const wins = closedTrades.filter(ct => ct.realized_r > 0).length;
  const losses = closedTrades.filter(ct => ct.realized_r <= 0).length;
  const progressPct = progress.total > 0 ? (progress.bar / progress.total) * 100 : 0;

  // Max drawdown from daily equity snapshots
  const maxDrawdownPct = (() => {
    if (dayStats.length < 2) return 0;
    let peak = dayStats[0].equity;
    let worst = 0;
    for (const ds of dayStats) {
      if (ds.equity > peak) peak = ds.equity;
      const dd = peak > 0 ? ((peak - ds.equity) / peak) * 100 : 0;
      if (dd > worst) worst = dd;
    }
    return worst;
  })();

  // Show saved results when not running and results exist
  const showResults = !running && !finished && savedResults && pnlHistory.length === 0;

  return (
    <div style={{ padding: 0 }}>
      {/* Header Controls */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "12px 16px", background: t.bgCard, borderRadius: 12,
        border: `1px solid ${t.border}`, marginBottom: 12,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 14, fontWeight: 800, color: t.accent }}>AGENT TRADING</span>
          {running && (
            <span style={{ fontSize: 10, color: engine === "agents" ? t.gold : engine === "det_plus" ? t.accent : t.textMuted, fontWeight: 600, padding: "2px 6px", background: engine === "agents" ? t.goldBg : engine === "det_plus" ? t.accent + "20" : t.bgHover, borderRadius: 4 }}>
              {engine === "agents" ? "AI" : engine === "det_plus" ? "DET+" : "DET"}
            </span>
          )}

          {!running && (
            <div style={{ display: "flex", gap: 4 }}>
              {(["continuous", "single"] as const).map(m => (
                <button key={m} onClick={() => setMode(m)}
                  style={{
                    background: mode === m ? t.accent + "20" : "transparent",
                    color: mode === m ? t.accent : t.textMuted,
                    border: `1px solid ${mode === m ? t.accent + "50" : t.border}`,
                    borderRadius: 4, padding: "3px 8px", fontSize: 10, fontWeight: 600,
                    cursor: "pointer",
                  }}>
                  {m === "continuous" ? `All ${dates.length} Days` : "Single Day"}
                </button>
              ))}
            </div>
          )}

          {!running && (
            <div style={{ display: "flex", gap: 3 }}>
              {([
                { key: "standard" as const, label: "Deterministic" },
                { key: "det_plus" as const, label: "Det+" },
                { key: "agents" as const, label: "AI Agents" },
              ]).map(({ key, label }) => (
                <button key={key} onClick={() => setEngine(key)}
                  style={{
                    background: engine === key ? (key === "agents" ? t.gold + "20" : key === "det_plus" ? t.accent + "20" : t.bgHover) : "transparent",
                    color: engine === key ? (key === "agents" ? t.gold : key === "det_plus" ? t.accent : t.text) : t.textMuted,
                    border: `1px solid ${engine === key ? (key === "agents" ? t.gold : key === "det_plus" ? t.accent : t.border) : t.border}`,
                    borderRadius: 4, padding: "3px 8px", fontSize: 10, fontWeight: 600,
                    cursor: "pointer",
                  }}>
                  {label}
                </button>
              ))}
            </div>
          )}

          {mode === "single" && !running && (
            <select
              value={selectedDate}
              onChange={e => setSelectedDate(e.target.value)}
              style={{
                background: t.bg, color: t.text, border: `1px solid ${t.border}`,
                borderRadius: 6, padding: "4px 8px", fontSize: 12, cursor: "pointer",
              }}
            >
              {dates.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          )}

          {!running && engine === "det_plus" && (
            <span style={{ fontSize: 9, color: t.textMuted, maxWidth: 200 }}>
              $1M | Adaptive sizing | Multi-TF | Learning
            </span>
          )}

          {!running ? (
            <button
              onClick={() => startSim(mode === "continuous")}
              disabled={mode === "single" && !selectedDate}
              style={{
                background: engine === "det_plus" ? t.accent : t.accent, color: "#fff", border: "none", borderRadius: 6,
                padding: "5px 16px", fontSize: 12, fontWeight: 700, cursor: "pointer",
              }}
            >
              {mode === "continuous" ? `RUN ALL ${dates.length} DAYS` : "START"}
            </button>
          ) : (
            <div style={{ display: "flex", gap: 4 }}>
              <button onClick={paused ? resumeSim : pauseSim}
                style={{
                  background: t.bgHover, color: t.text, border: `1px solid ${t.border}`,
                  borderRadius: 6, padding: "5px 12px", fontSize: 11, fontWeight: 600, cursor: "pointer",
                }}>
                {paused ? "RESUME" : "PAUSE"}
              </button>
              <button onClick={stopSim}
                style={{
                  background: t.shortBg, color: t.short, border: `1px solid ${t.short}30`,
                  borderRadius: 6, padding: "5px 12px", fontSize: 11, fontWeight: 600, cursor: "pointer",
                }}>
                STOP
              </button>
            </div>
          )}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {/* Day counter for multi-day */}
          {running && totalDays > 0 && (
            <div style={{ fontSize: 11, color: t.accent, fontWeight: 700 }}>
              Day {currentDay}/{totalDays} {currentDate && `(${currentDate})`}
            </div>
          )}

          {/* Speed control */}
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ fontSize: 10, color: t.textMuted }}>Speed:</span>
            {[1, 5, 10, 25, 50].map(s => (
              <button key={s} onClick={() => changeSpeed(s)}
                style={{
                  background: speed === s ? t.accent : t.bg,
                  color: speed === s ? "#fff" : t.textDim,
                  border: `1px solid ${speed === s ? t.accent : t.border}`,
                  borderRadius: 4, padding: "2px 6px", fontSize: 10, cursor: "pointer",
                  fontWeight: speed === s ? 700 : 400,
                }}>
                {s}x
              </button>
            ))}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{
              width: 6, height: 6, borderRadius: "50%",
              background: connected ? t.long : t.short,
            }} />
            <span style={{ fontSize: 10, color: t.textMuted }}>
              {connected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>
      </div>

      {/* Show saved results when idle */}
      {showResults && <LongTermView t={t} />}

      {/* Live simulation view */}
      {(running || finished || pnlHistory.length > 0) && <>
        {/* Progress Bar */}
        {running && (
          <div style={{ marginBottom: 12 }}>
            <div style={{
              height: 3, background: t.border, borderRadius: 2, overflow: "hidden",
            }}>
              <motion.div
                animate={{ width: `${progressPct}%` }}
                transition={{ duration: 0.3 }}
                style={{ height: "100%", background: t.accent, borderRadius: 2 }}
              />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
              <span style={{ fontSize: 10, color: t.textMuted }}>
                Bar {progress.bar}/{progress.total} | {progress.time.slice(5, 16)}
              </span>
              <span style={{ fontSize: 10, color: t.textMuted }}>{progressPct.toFixed(0)}%</span>
            </div>
          </div>
        )}

        {/* Stats Row */}
        <div style={{
          display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: 8, marginBottom: 12,
        }}>
          {[
            { label: "EQUITY", value: `$${equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, color: t.text },
            { label: "TOTAL P&L", value: `${totalPnl >= 0 ? "+" : ""}$${totalPnl.toFixed(0)}`, color: totalPnl >= 0 ? t.long : t.short },
            { label: "TOTAL R", value: `${totalR >= 0 ? "+" : ""}${totalR.toFixed(2)}R`, color: totalR >= 0 ? t.long : t.short },
            { label: "W / L", value: `${wins} / ${losses}`, color: wins > losses ? t.long : wins < losses ? t.short : t.textDim },
            { label: "MAX DD", value: `${maxDrawdownPct.toFixed(1)}%`, color: maxDrawdownPct > 5 ? t.short : maxDrawdownPct > 2 ? t.gold : t.textDim },
            { label: "OPEN", value: `${positions.length}`, color: t.accent },
            { label: "HEAT", value: `${heat.toFixed(1)}%`, color: heat > 5 ? t.gold : t.textDim },
            { label: "TRADES", value: `${closedTrades.length}`, color: t.textDim },
          ].map((s, i) => (
            <div key={i} style={{
              background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 8,
              padding: "8px 12px", textAlign: "center",
            }}>
              <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 600, marginBottom: 4 }}>{s.label}</div>
              <div style={{ fontSize: 16, fontWeight: 800, color: s.color, fontFamily: "monospace" }}>{s.value}</div>
            </div>
          ))}
        </div>

        {/* Main Grid: Chart + Positions | Agent Log */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          {/* Left: PNL Chart + Positions */}
          <div>
            {/* PNL Chart */}
            <div style={{
              background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 12,
              padding: 12, marginBottom: 12,
            }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: t.textDim, marginBottom: 8 }}>P&L CURVE ($)</div>
              <PnlChart points={pnlHistory} t={t} />
            </div>

            {/* Open Positions */}
            <div style={{
              background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 12,
              padding: 12, marginBottom: 12,
            }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: t.textDim, marginBottom: 8 }}>
                OPEN POSITIONS ({positions.length})
              </div>
              {positions.length === 0 ? (
                <div style={{ color: t.textMuted, fontSize: 11, padding: 12, textAlign: "center" }}>No open positions</div>
              ) : (
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
                  <thead>
                    <tr style={{ borderBottom: `1px solid ${t.border}` }}>
                      {["Symbol", "Pattern", "Bias", "Entry", "Stop", "Price", "R", "Status"].map(h => (
                        <th key={h} style={{ padding: "4px 8px", textAlign: "left", fontSize: 9, color: t.textMuted, fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map(p => <PositionRow key={p.id} pos={p} t={t} />)}
                  </tbody>
                </table>
              )}
            </div>

            {/* Closed Trades */}
            {closedTrades.length > 0 && (
              <div style={{
                background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 12,
                padding: 12,
              }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: t.textDim, marginBottom: 8 }}>
                  CLOSED TRADES ({closedTrades.length})
                </div>
                <div style={{ maxHeight: 400, overflow: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${t.border}`, position: "sticky", top: 0, background: t.bgCard }}>
                        {["ID", "Symbol", "Outcome", "R", "P&L", "Bars"].map(h => (
                          <th key={h} style={{ padding: "4px 8px", textAlign: "left", fontSize: 9, color: t.textMuted, fontWeight: 600 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {closedTrades.slice().reverse().map(tr => (
                        <tr key={tr.id} style={{ borderBottom: `1px solid ${t.border}` }}>
                          <td style={{ padding: "4px 8px", color: t.textMuted }}>{tr.id}</td>
                          <td style={{ padding: "4px 8px", fontWeight: 700 }}>{tr.symbol}</td>
                          <td style={{
                            padding: "4px 8px", fontWeight: 700, fontSize: 10,
                            color: tr.outcome === "win" ? t.long : tr.outcome === "loss" ? t.short : t.gold,
                          }}>
                            {tr.outcome.toUpperCase()}
                          </td>
                          <td style={{
                            padding: "4px 8px", fontWeight: 700,
                            color: tr.realized_r > 0 ? t.long : t.short,
                          }}>
                            {tr.realized_r > 0 ? "+" : ""}{tr.realized_r.toFixed(2)}R
                          </td>
                          <td style={{
                            padding: "4px 8px",
                            color: tr.pnl > 0 ? t.long : t.short,
                          }}>
                            ${tr.pnl.toFixed(0)}
                          </td>
                          <td style={{ padding: "4px 8px", color: t.textMuted }}>{tr.bars_held}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>

          {/* Right: Agent Log */}
          <div style={{
            background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 12,
            padding: 12, display: "flex", flexDirection: "column",
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: t.textDim, marginBottom: 8 }}>
              AGENT LOG ({events.length} events)
            </div>
            <div
              ref={logRef}
              style={{
                flex: 1, minHeight: 400, maxHeight: "calc(100vh - 360px)",
                overflow: "auto", background: t.bg, borderRadius: 8,
                padding: 8, fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
              }}
            >
              {events.length === 0 ? (
                <div style={{ color: t.textMuted, fontSize: 11, padding: 20, textAlign: "center" }}>
                  {running ? "Waiting for events..." : "Press START to begin simulation"}
                </div>
              ) : (
                events.map((e, i) => <LogEntry key={i} event={e} t={t} />)
              )}
            </div>
          </div>
        </div>

        {/* Daily Performance Charts */}
        {dayStats.length >= 2 && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 12 }}>
            {/* Daily P&L Histogram */}
            <div style={{
              background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 12, padding: 12,
            }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: t.textDim, marginBottom: 8 }}>
                DAILY P&L HISTOGRAM
              </div>
              {(() => {
                const W = 400, H = 140, PAD = 50, PADR = 10, PADT = 10, PADB = 22;
                const pnls = dayStats.map(d => d.day_pnl);
                const maxAbs = Math.max(1, ...pnls.map(Math.abs));
                const barW = Math.max(1, (W - PAD - PADR) / dayStats.length - 1);
                const toX = (i: number) => PAD + (i / dayStats.length) * (W - PAD - PADR);
                const zeroY = PADT + (H - PADT - PADB) / 2;
                const scale = (H - PADT - PADB) / 2 / maxAbs;
                const step = Math.max(1, Math.floor(dayStats.length / 6));

                return (
                  <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 140 }}>
                    <line x1={PAD} y1={zeroY} x2={W - PADR} y2={zeroY} stroke={t.textMuted} strokeWidth={0.5} />
                    {pnls.map((pnl, i) => {
                      const h = Math.abs(pnl) * scale;
                      const y = pnl >= 0 ? zeroY - h : zeroY;
                      return (
                        <rect key={i} x={toX(i)} y={y} width={barW} height={Math.max(0.5, h)}
                          fill={pnl >= 0 ? t.long : t.short} opacity={0.75} rx={0.5} />
                      );
                    })}
                    <text x={PAD - 4} y={PADT + 5} textAnchor="end" fill={t.textMuted} fontSize={8}>
                      +${(maxAbs / 1000).toFixed(1)}k
                    </text>
                    <text x={PAD - 4} y={zeroY} textAnchor="end" fill={t.textMuted} fontSize={8} dominantBaseline="middle">$0</text>
                    <text x={PAD - 4} y={H - PADB - 2} textAnchor="end" fill={t.textMuted} fontSize={8}>
                      -${(maxAbs / 1000).toFixed(1)}k
                    </text>
                    {dayStats.filter((_, i) => i % step === 0 || i === dayStats.length - 1).map((ds, i) => (
                      <text key={i} x={toX(dayStats.indexOf(ds)) + barW / 2} y={H - 5}
                        textAnchor="middle" fill={t.textMuted} fontSize={7}>{ds.date.slice(5)}</text>
                    ))}
                  </svg>
                );
              })()}
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, fontSize: 9, color: t.textMuted }}>
                <span>
                  Green: {dayStats.filter(d => d.day_pnl >= 0).length} |
                  Red: {dayStats.filter(d => d.day_pnl < 0).length}
                </span>
                <span>
                  Best: ${Math.max(...dayStats.map(d => d.day_pnl)).toLocaleString(undefined, { maximumFractionDigits: 0 })} |
                  Worst: ${Math.min(...dayStats.map(d => d.day_pnl)).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </span>
              </div>
            </div>

            {/* Drawdown Chart */}
            <div style={{
              background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 12, padding: 12,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: t.textDim }}>DRAWDOWN</span>
                <span style={{ fontSize: 11, fontWeight: 800, color: t.short, fontFamily: "monospace" }}>
                  Max: {maxDrawdownPct.toFixed(2)}%
                </span>
              </div>
              {(() => {
                const W = 400, H = 140, PAD = 50, PADR = 10, PADT = 10, PADB = 22;
                // Compute drawdown series
                let peak = dayStats[0].equity;
                const ddSeries = dayStats.map(ds => {
                  if (ds.equity > peak) peak = ds.equity;
                  return peak > 0 ? ((peak - ds.equity) / peak) * 100 : 0;
                });
                const maxDD = Math.max(0.1, ...ddSeries);

                const toX = (i: number) => PAD + (i / (dayStats.length - 1)) * (W - PAD - PADR);
                const toY = (dd: number) => PADT + (dd / maxDD) * (H - PADT - PADB);

                const pathParts = ddSeries.map((dd, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(dd).toFixed(1)}`);
                const linePath = pathParts.join(" ");
                const fillPath = linePath + ` L${toX(ddSeries.length - 1).toFixed(1)},${PADT} L${toX(0).toFixed(1)},${PADT} Z`;

                const step = Math.max(1, Math.floor(dayStats.length / 6));

                return (
                  <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 140 }}>
                    {/* 0% line at top */}
                    <line x1={PAD} y1={PADT} x2={W - PADR} y2={PADT} stroke={t.textMuted} strokeWidth={0.5} strokeDasharray="3,3" />
                    {/* Grid lines */}
                    {[0.25, 0.5, 0.75].map((frac, i) => (
                      <line key={i} x1={PAD} y1={toY(maxDD * frac)} x2={W - PADR} y2={toY(maxDD * frac)}
                        stroke={t.border} strokeWidth={0.5} />
                    ))}
                    {/* Fill */}
                    <path d={fillPath} fill={t.short} opacity={0.1} />
                    {/* Line */}
                    <path d={linePath} fill="none" stroke={t.short} strokeWidth={1.5} />
                    {/* Max DD marker */}
                    {(() => {
                      const maxIdx = ddSeries.indexOf(Math.max(...ddSeries));
                      return (
                        <circle cx={toX(maxIdx)} cy={toY(ddSeries[maxIdx])} r={3} fill={t.short} />
                      );
                    })()}
                    {/* Y-axis labels */}
                    <text x={PAD - 4} y={PADT} textAnchor="end" fill={t.textMuted} fontSize={8} dominantBaseline="middle">0%</text>
                    <text x={PAD - 4} y={toY(maxDD / 2)} textAnchor="end" fill={t.textMuted} fontSize={8} dominantBaseline="middle">
                      -{(maxDD / 2).toFixed(1)}%
                    </text>
                    <text x={PAD - 4} y={toY(maxDD)} textAnchor="end" fill={t.textMuted} fontSize={8} dominantBaseline="middle">
                      -{maxDD.toFixed(1)}%
                    </text>
                    {/* X-axis labels */}
                    {dayStats.filter((_, i) => i % step === 0 || i === dayStats.length - 1).map((ds, i) => (
                      <text key={i} x={toX(dayStats.indexOf(ds))} y={H - 5}
                        textAnchor="middle" fill={t.textMuted} fontSize={7}>{ds.date.slice(5)}</text>
                    ))}
                  </svg>
                );
              })()}
            </div>
          </div>
        )}

        {/* Analytics Section */}
        {closedTrades.length >= 3 && (() => {
          // Strategy breakdown
          const byStrategy: Record<string, { wins: number; losses: number; pnl: number; totalR: number; count: number }> = {};
          const byHour: Record<string, { wins: number; losses: number; pnl: number; count: number }> = {};

          for (const tr of closedTrades) {
            // By strategy/pattern
            const pat = tr.pattern || "Unknown";
            if (!byStrategy[pat]) byStrategy[pat] = { wins: 0, losses: 0, pnl: 0, totalR: 0, count: 0 };
            byStrategy[pat].count++;
            byStrategy[pat].pnl += tr.pnl || 0;
            byStrategy[pat].totalR += tr.realized_r || 0;
            if (tr.realized_r > 0) byStrategy[pat].wins++;
            else byStrategy[pat].losses++;

            // By hour of entry
            const hour = tr.entry_time?.slice(11, 13) || "??";
            const label = `${hour}:00`;
            if (!byHour[label]) byHour[label] = { wins: 0, losses: 0, pnl: 0, count: 0 };
            byHour[label].count++;
            byHour[label].pnl += tr.pnl || 0;
            if (tr.realized_r > 0) byHour[label].wins++;
            else byHour[label].losses++;
          }

          const stratSorted = Object.entries(byStrategy).sort((a, b) => b[1].pnl - a[1].pnl);
          const hourSorted = Object.entries(byHour).sort((a, b) => a[0].localeCompare(b[0]));

          return (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 12 }}>
              {/* Strategy Performance */}
              <div style={{
                background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 12,
                padding: 12,
              }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: t.textDim, marginBottom: 8 }}>
                  STRATEGY PERFORMANCE
                </div>
                <div style={{ maxHeight: 300, overflow: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}>
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${t.border}` }}>
                        {["Pattern", "Trades", "W/L", "Win%", "P&L", "Avg R"].map(h => (
                          <th key={h} style={{ padding: "4px 6px", textAlign: "left", fontSize: 8, color: t.textMuted, fontWeight: 600 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {stratSorted.map(([pat, s]) => (
                        <tr key={pat} style={{ borderBottom: `1px solid ${t.border}` }}>
                          <td style={{ padding: "4px 6px", color: t.text, fontSize: 9 }}>{pat.slice(0, 25)}</td>
                          <td style={{ padding: "4px 6px", color: t.textDim }}>{s.count}</td>
                          <td style={{ padding: "4px 6px", color: t.textDim }}>{s.wins}/{s.losses}</td>
                          <td style={{ padding: "4px 6px", color: s.count > 0 && s.wins / s.count >= 0.4 ? t.long : t.short, fontWeight: 600 }}>
                            {s.count > 0 ? ((s.wins / s.count) * 100).toFixed(0) : 0}%
                          </td>
                          <td style={{ padding: "4px 6px", color: s.pnl >= 0 ? t.long : t.short, fontWeight: 700 }}>
                            {s.pnl >= 0 ? "+" : ""}${s.pnl.toFixed(0)}
                          </td>
                          <td style={{ padding: "4px 6px", color: s.count > 0 && s.totalR / s.count >= 0 ? t.long : t.short }}>
                            {s.count > 0 ? (s.totalR / s.count).toFixed(2) : "0.00"}R
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Time of Day Performance */}
              <div style={{
                background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 12,
                padding: 12,
              }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: t.textDim, marginBottom: 8 }}>
                  TIME OF DAY PERFORMANCE
                </div>
                <div style={{ maxHeight: 300, overflow: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}>
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${t.border}` }}>
                        {["Hour", "Trades", "W/L", "Win%", "P&L"].map(h => (
                          <th key={h} style={{ padding: "4px 6px", textAlign: "left", fontSize: 8, color: t.textMuted, fontWeight: 600 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {hourSorted.map(([hour, s]) => (
                        <tr key={hour} style={{ borderBottom: `1px solid ${t.border}` }}>
                          <td style={{ padding: "4px 6px", color: t.text, fontWeight: 600 }}>{hour}</td>
                          <td style={{ padding: "4px 6px", color: t.textDim }}>{s.count}</td>
                          <td style={{ padding: "4px 6px", color: t.textDim }}>{s.wins}/{s.losses}</td>
                          <td style={{ padding: "4px 6px", color: s.count > 0 && s.wins / s.count >= 0.4 ? t.long : t.short, fontWeight: 600 }}>
                            {s.count > 0 ? ((s.wins / s.count) * 100).toFixed(0) : 0}%
                          </td>
                          <td style={{ padding: "4px 6px", color: s.pnl >= 0 ? t.long : t.short, fontWeight: 700 }}>
                            {s.pnl >= 0 ? "+" : ""}${s.pnl.toFixed(0)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          );
        })()}
      </>}
    </div>
  );
}
