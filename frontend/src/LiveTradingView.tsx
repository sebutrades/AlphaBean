import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";

const WS_URL = "ws://localhost:8000/ws/live-trading";

type T = {
  bg: string; bgCard: string; bgHover: string; border: string; borderLight: string;
  text: string; textDim: string; textMuted: string;
  accent: string; accentLight: string;
  long: string; longBg: string; short: string; shortBg: string;
  gold: string; goldBg: string; purple: string;
  chartBg: string; chartGrid: string; chartText: string;
  glow: string;
};

// ── Types ────────────────────────────────────────────────────────────────────

type LiveEvent = {
  type: string;
  timestamp: string;
  data: Record<string, any>;
};

type Setup = {
  symbol: string;
  pattern: string;
  bias: string;
  entry_price: number;
  stop_loss: number;
  target_1: number;
  target_2: number;
  risk_reward: number;
  composite_score: number;
  pattern_confidence: number;
  feature_score: number;
  strategy_score: number;
  regime_score: number;
  backtest_score: number;
  volume_score: number;
  rr_score: number;
  timeframe: string;
  in_position: boolean;
};

type Position = {
  symbol: string;
  bias: string;
  entry_price: number;
  stop_loss: number;
  target_1: number;
  target_2: number;
  shares: number;
  pattern: string;
  score: number;
  timeframe: string;
  status: string;
  t1_hit: boolean;
  t2_hit: boolean;
  entered_at: string;
};

type ClosedTrade = {
  symbol: string;
  bias: string;
  entry_price: number;
  exit_price: number;
  pnl: number;
  exit_reason: string;
  pattern: string;
  score: number;
};

type PnlSnapshot = {
  timestamp: string;
  open_positions: number;
  closed_trades: number;
  total_orders: number;
  realized_pnl: number;
  positions: Position[];
  closed: ClosedTrade[];
};

// ── Live Price Chart (lightweight-charts) ────────────────────────────────────

const API = "http://localhost:8000";

function LivePriceChart({
  symbol, positions, closedTrades, t,
}: {
  symbol: string;
  positions: Position[];
  closedTrades: ClosedTrade[];
  t: T;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const seriesRef = useRef<any>(null);
  const [error, setError] = useState("");

  // Find active position or most recent closed trade for this symbol
  const pos = positions.find(p => p.symbol === symbol);
  const closedPos = closedTrades.filter(tr => tr.symbol === symbol).slice(-1)[0];
  const activeEntry = pos || closedPos;

  useEffect(() => {
    if (!containerRef.current || !symbol) return;
    let chart: any = null;

    (async () => {
      try {
        const lc = await import("lightweight-charts");
        const r = await fetch(`${API}/api/chart/${symbol}?timeframe=5min&days_back=3`);
        const d = await r.json();
        if (d.error) throw new Error(d.error);
        if (!d.bars?.length) throw new Error("No bar data");

        if (containerRef.current) containerRef.current.innerHTML = "";

        chart = lc.createChart(containerRef.current!, {
          width: containerRef.current!.clientWidth,
          height: 350,
          layout: { background: { color: t.chartBg } as any, textColor: t.chartText, fontFamily: "JetBrains Mono,monospace" },
          grid: { vertLines: { color: t.chartGrid }, horzLines: { color: t.chartGrid } },
          crosshair: { mode: lc.CrosshairMode.Normal },
          rightPriceScale: { borderColor: t.border },
          timeScale: { borderColor: t.border, timeVisible: true },
        });
        chartRef.current = chart;

        let series: any;
        if ((lc as any).CandlestickSeries) {
          series = chart.addSeries((lc as any).CandlestickSeries, {
            upColor: t.long, downColor: t.short,
            borderUpColor: t.long, borderDownColor: t.short,
            wickUpColor: t.long, wickDownColor: t.short,
          });
        } else {
          series = chart.addCandlestickSeries({
            upColor: t.long, downColor: t.short,
            borderUpColor: t.long, borderDownColor: t.short,
            wickUpColor: t.long, wickDownColor: t.short,
          });
        }
        series.setData(d.bars);
        seriesRef.current = series;

        // Helper: add price line
        const addLine = (price: number, color: string, title: string, style: number, width: number) => {
          try { series.createPriceLine({ price, color, lineWidth: width, lineStyle: style, axisLabelVisible: true, title }); } catch {}
        };

        // Draw entry/stop/target lines from active position
        if (pos) {
          addLine(pos.entry_price, t.accent, `ENTRY $${pos.entry_price.toFixed(2)}`, lc.LineStyle.Dashed, 2);
          addLine(pos.stop_loss, t.short, "STOP", lc.LineStyle.Dashed, 2);
          addLine(pos.target_1, t.long, "T1", lc.LineStyle.Dashed, 1);
          addLine(pos.target_2, t.long, "T2", lc.LineStyle.Dashed, 1);

          // Entry marker
          if (pos.entered_at && d.bars?.length) {
            try {
              const sigTs = Math.floor(new Date(pos.entered_at).getTime() / 1000);
              const nearest = d.bars.reduce((a: any, b: any) =>
                Math.abs(b.time - sigTs) < Math.abs(a.time - sigTs) ? b : a
              );
              const isLong = pos.bias === "long";
              const mk = {
                time: nearest.time,
                position: isLong ? "belowBar" : "aboveBar",
                color: isLong ? t.long : t.short,
                shape: isLong ? "arrowUp" : "arrowDown",
                text: pos.pattern, size: 2,
              };
              if (typeof (lc as any).createSeriesMarkers === "function") (lc as any).createSeriesMarkers(series, [mk]);
              else series.setMarkers([mk]);
            } catch {}
          }
        } else if (closedPos) {
          addLine(closedPos.entry_price, t.textMuted, `ENTRY $${closedPos.entry_price.toFixed(2)}`, lc.LineStyle.Dotted, 1);
          addLine(closedPos.exit_price, closedPos.pnl >= 0 ? t.long : t.short,
            `EXIT $${closedPos.exit_price.toFixed(2)} (${closedPos.pnl >= 0 ? "+" : ""}$${closedPos.pnl.toFixed(2)})`,
            lc.LineStyle.Dotted, 1);
        }

        chart.timeScale().fitContent();
        setError("");

        const ro = new ResizeObserver(() => {
          if (containerRef.current && chart) chart.applyOptions({ width: containerRef.current.clientWidth });
        });
        ro.observe(containerRef.current!);
      } catch (e: any) {
        setError(e.message || "Chart error");
      }
    })();

    return () => { if (chart) chart.remove(); chartRef.current = null; seriesRef.current = null; };
  }, [symbol, pos?.status, pos?.t1_hit, pos?.t2_hit, closedTrades.length, t]);

  if (error) return (
    <div style={{ color: t.textMuted, fontSize: 11, padding: 20, textAlign: "center" }}>
      Chart unavailable: {error}
    </div>
  );

  return <div ref={containerRef} style={{ width: "100%", minHeight: 350 }} />;
}

// ── P&L Equity Chart (pure SVG) ─────────────────────────────────────────────

function PnlChart({ history, t }: { history: PnlSnapshot[]; t: T }) {
  if (history.length < 2) return null;

  const W = 800, H = 200, PAD = 60, PADR = 20, PADT = 15, PADB = 25;
  const values = history.map(p => p.realized_pnl);
  const minV = Math.min(0, ...values);
  const maxV = Math.max(0.01, ...values);
  const range = maxV - minV || 1;

  const toX = (i: number) => PAD + (i / (history.length - 1)) * (W - PAD - PADR);
  const toY = (v: number) => PADT + ((maxV - v) / range) * (H - PADT - PADB);
  const zeroY = toY(0);

  const pathParts = history.map((p, i) =>
    `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(p.realized_pnl).toFixed(1)}`
  );
  const linePath = pathParts.join(" ");
  const fillPath = linePath + ` L${toX(history.length - 1).toFixed(1)},${zeroY.toFixed(1)} L${toX(0).toFixed(1)},${zeroY.toFixed(1)} Z`;

  const lastPnl = values[values.length - 1];
  const color = lastPnl >= 0 ? t.long : t.short;

  const yTicks = [minV, minV + range / 2, maxV];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 200 }}>
      {yTicks.map((v, i) => (
        <line key={i} x1={PAD} y1={toY(v)} x2={W - PADR} y2={toY(v)} stroke={t.chartGrid} strokeWidth={0.5} />
      ))}
      <line x1={PAD} y1={zeroY} x2={W - PADR} y2={zeroY} stroke={t.textMuted} strokeWidth={1} strokeDasharray="4,3" />
      <path d={fillPath} fill={color} opacity={0.08} />
      <path d={linePath} fill="none" stroke={color} strokeWidth={2} />
      <circle cx={toX(history.length - 1)} cy={toY(lastPnl)} r={4} fill={color} />
      <text x={toX(history.length - 1) + 8} y={toY(lastPnl)} fill={color} fontSize={11} fontWeight={700} dominantBaseline="middle">
        ${lastPnl.toFixed(2)}
      </text>
      {yTicks.map((v, i) => (
        <text key={i} x={PAD - 5} y={toY(v)} textAnchor="end" fill={t.chartText} fontSize={9} dominantBaseline="middle">
          ${v.toFixed(0)}
        </text>
      ))}
      <text x={PAD - 5} y={zeroY} textAnchor="end" fill={t.chartText} fontSize={9} dominantBaseline="middle">$0</text>
    </svg>
  );
}

// ── Trade P&L Bar Chart ─────────────────────────────────────────────────────

function TradePnlBars({ trades, t }: { trades: ClosedTrade[]; t: T }) {
  if (trades.length === 0) return null;

  const W = 800, H = 120, PAD = 50, PADR = 20, PADT = 10, PADB = 20;
  const pnls = trades.map(tr => tr.pnl);
  const maxAbs = Math.max(1, ...pnls.map(Math.abs));
  const barW = Math.max(2, Math.min(20, (W - PAD - PADR) / trades.length - 1));
  const zeroY = PADT + (H - PADT - PADB) / 2;
  const scale = (H - PADT - PADB) / 2 / maxAbs;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 120 }}>
      <line x1={PAD} y1={zeroY} x2={W - PADR} y2={zeroY} stroke={t.textMuted} strokeWidth={0.5} />
      {pnls.map((pnl, i) => {
        const h = Math.abs(pnl) * scale;
        const x = PAD + (i / trades.length) * (W - PAD - PADR);
        const y = pnl >= 0 ? zeroY - h : zeroY;
        return (
          <rect key={i} x={x} y={y} width={barW} height={Math.max(0.5, h)}
            fill={pnl >= 0 ? t.long : t.short} opacity={0.8} rx={1} />
        );
      })}
      <text x={PAD - 5} y={PADT + 5} textAnchor="end" fill={t.chartText} fontSize={9}>+${maxAbs.toFixed(0)}</text>
      <text x={PAD - 5} y={zeroY} textAnchor="end" fill={t.chartText} fontSize={9} dominantBaseline="middle">$0</text>
      <text x={PAD - 5} y={H - PADB} textAnchor="end" fill={t.chartText} fontSize={9}>-${maxAbs.toFixed(0)}</text>
    </svg>
  );
}

// ── Pattern Breakdown ───────────────────────────────────────────────────────

function PatternBreakdown({ trades, t }: { trades: ClosedTrade[]; t: T }) {
  const byPattern: Record<string, { wins: number; losses: number; totalPnl: number; count: number }> = {};
  for (const tr of trades) {
    const key = tr.pattern;
    if (!byPattern[key]) byPattern[key] = { wins: 0, losses: 0, totalPnl: 0, count: 0 };
    byPattern[key].count++;
    byPattern[key].totalPnl += tr.pnl;
    if (tr.pnl >= 0) byPattern[key].wins++;
    else byPattern[key].losses++;
  }

  const sorted = Object.entries(byPattern).sort((a, b) => b[1].totalPnl - a[1].totalPnl);
  if (sorted.length === 0) return null;

  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
      <thead>
        <tr style={{ borderBottom: `1px solid ${t.border}` }}>
          {["Pattern", "Trades", "W/L", "Win%", "P&L"].map(h => (
            <th key={h} style={{ padding: "6px 8px", textAlign: "left", fontSize: 9, color: t.textMuted, fontWeight: 600 }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {sorted.map(([pattern, s]) => {
          const wr = s.count > 0 ? (s.wins / s.count) * 100 : 0;
          return (
            <tr key={pattern} style={{ borderBottom: `1px solid ${t.border}` }}>
              <td style={{ padding: "5px 8px", fontWeight: 600, color: t.text, fontSize: 10 }}>{pattern}</td>
              <td style={{ padding: "5px 8px", color: t.textDim }}>{s.count}</td>
              <td style={{ padding: "5px 8px", color: t.textDim }}>{s.wins}/{s.losses}</td>
              <td style={{ padding: "5px 8px", color: wr >= 50 ? t.long : t.short, fontWeight: 600 }}>{wr.toFixed(0)}%</td>
              <td style={{ padding: "5px 8px", color: s.totalPnl >= 0 ? t.long : t.short, fontWeight: 700 }}>
                {s.totalPnl >= 0 ? "+" : ""}${s.totalPnl.toFixed(2)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ── Log Event Row ─────────────────────────────────────────────��─────────────

function LogRow({ event, t }: { event: LiveEvent; t: T }) {
  const ts = event.timestamp.split("T")[1]?.slice(0, 8) || "";
  const d = event.data;

  const typeColors: Record<string, string> = {
    scan_start: t.accent,
    scan_done: t.accent,
    log: t.textDim,
    trade_open: t.long,
    trade_close: t.textDim,
    stop_hit: t.short,
    target_hit: t.gold,
    error: t.short,
    session_start: t.purple,
    pnl_snapshot: t.textMuted,
    fetch_progress: t.textMuted,
  };

  const color = typeColors[event.type] || t.textDim;
  let message = "";

  switch (event.type) {
    case "log":
      message = d.message || "";
      break;
    case "scan_start":
      message = `Scan #${d.scan_number} starting (${d.symbols} symbols, ${d.open_positions} positions)`;
      break;
    case "scan_done":
      message = `Found ${d.total_setups} setups`;
      break;
    case "trade_open":
      message = `${d.side} ${d.shares}x ${d.symbol} @ $${d.entry_price?.toFixed(2)} | Stop $${d.stop_loss?.toFixed(2)} | ${d.pattern} (${d.score})`;
      break;
    case "stop_hit":
      message = `STOP ${d.symbol} @ $${d.exit_price?.toFixed(2)} | P&L $${d.pnl?.toFixed(2)} (${d.r_multiple?.toFixed(1)}R)`;
      break;
    case "target_hit":
      message = `${d.target} HIT ${d.symbol} @ $${d.price?.toFixed(2)} | Sold ${d.shares_sold} | ${d.r_multiple?.toFixed(1)}R`;
      break;
    case "trade_close":
      message = `CLOSED ${d.symbol} | ${d.exit_reason} | P&L $${d.pnl?.toFixed(2)}`;
      break;
    case "error":
      message = d.message || "Unknown error";
      break;
    case "session_start":
      message = `Session started: ${d.symbols?.length || 0} symbols, interval=${d.scan_interval}s`;
      break;
    default:
      message = JSON.stringify(d).slice(0, 100);
  }

  return (
    <div style={{ display: "flex", gap: 8, padding: "2px 0", fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}>
      <span style={{ color: t.textMuted, minWidth: 60 }}>{ts}</span>
      <span style={{ color, fontWeight: 600, minWidth: 80, textTransform: "uppercase" }}>{event.type.replace(/_/g, " ")}</span>
      <span style={{ color: t.text, flex: 1 }}>{message}</span>
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────────���───────────

export default function LiveTradingView({ t }: { t: T }) {
  // Connection + control
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [paused, setPaused] = useState(false);
  const [dryRun, setDryRun] = useState(true);
  const [maxSymbols, setMaxSymbols] = useState(30);
  const [scanInterval, setScanInterval] = useState(300);
  const [customSymbols, setCustomSymbols] = useState("");

  // Data
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [closedTrades, setClosedTrades] = useState<ClosedTrade[]>([]);
  const [pnlHistory, setPnlHistory] = useState<PnlSnapshot[]>([]);
  const [setups, setSetups] = useState<Setup[]>([]);
  const [scanNumber, setScanNumber] = useState(0);
  const [chartSymbol, setChartSymbol] = useState("");

  const wsRef = useRef<WebSocket | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  // WebSocket connection
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => { setConnected(false); setRunning(false); };

    ws.onmessage = (msg) => {
      try {
        const event: LiveEvent = JSON.parse(msg.data);
        handleEvent(event);
      } catch {}
    };
  }, []);

  const handleEvent = useCallback((event: LiveEvent) => {
    const d = event.data;

    // Always log interesting events
    if (!["fetch_progress", "pnl_snapshot"].includes(event.type)) {
      setEvents(prev => [...prev.slice(-500), event]);
    }

    switch (event.type) {
      case "session_start":
        setEvents(prev => [...prev.slice(-500), event]);
        break;

      case "scan_start":
        setScanNumber(d.scan_number || 0);
        break;

      case "scan_done":
        setSetups(d.setups || []);
        break;

      case "trade_open":
        setEvents(prev => [...prev.slice(-500), event]);
        setChartSymbol(prev => prev || d.symbol);
        break;

      case "trade_close":
        setClosedTrades(prev => [...prev, {
          symbol: d.symbol, bias: d.bias,
          entry_price: d.entry_price, exit_price: d.exit_price,
          pnl: d.pnl, exit_reason: d.exit_reason,
          pattern: d.pattern, score: d.score,
        }]);
        break;

      case "stop_hit":
        setEvents(prev => [...prev.slice(-500), event]);
        break;

      case "target_hit":
        setEvents(prev => [...prev.slice(-500), event]);
        break;

      case "pnl_snapshot":
        setPnlHistory(prev => [...prev, d as PnlSnapshot]);
        setPositions(d.positions || []);
        break;

      case "session_summary":
        setRunning(false);
        setEvents(prev => [...prev.slice(-500), event]);
        break;

      case "error":
        setEvents(prev => [...prev.slice(-500), event]);
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
    setPositions([]);
    setClosedTrades([]);
    setPnlHistory([]);
    setSetups([]);
    setScanNumber(0);
    setChartSymbol("");
  };

  const startScanner = () => {
    resetState();

    const trySend = () => {
      const ws = wsRef.current;
      if (ws?.readyState === WebSocket.OPEN) {
        const payload: Record<string, any> = {
          action: "start",
          dry_run: dryRun,
          live_mode: !dryRun,
          scan_interval: scanInterval,
        };
        if (customSymbols.trim()) {
          payload.symbols = customSymbols.split(",").map(s => s.trim().toUpperCase()).filter(Boolean);
        } else {
          payload.max_symbols = maxSymbols;
        }
        ws.send(JSON.stringify(payload));
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

  const send = (action: string) => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action }));
    }
  };

  const pauseScanner = () => { send("pause"); setPaused(true); };
  const resumeScanner = () => { send("resume"); setPaused(false); };
  const stopScanner = () => { send("stop"); setRunning(false); };

  // Live stats
  const lastSnapshot = pnlHistory.length ? pnlHistory[pnlHistory.length - 1] : null;
  const totalPnl = lastSnapshot?.realized_pnl || 0;
  const openCount = positions.length;
  const closedCount = closedTrades.length;
  const wins = closedTrades.filter(ct => ct.pnl >= 0).length;
  const losses = closedTrades.filter(ct => ct.pnl < 0).length;
  const winRate = closedCount > 0 ? (wins / closedCount) * 100 : 0;

  // Tab state for bottom section
  const [tab, setTab] = useState<"setups" | "positions" | "closed" | "log">("log");

  const cardStyle = {
    background: t.bgCard, borderRadius: 12, border: `1px solid ${t.border}`,
    padding: "12px 16px", marginBottom: 12,
  };

  return (
    <div style={{ padding: 0 }}>
      {/* ── Header Controls ─────────────────────────────────────────────── */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "12px 16px", background: t.bgCard, borderRadius: 12,
        border: `1px solid ${t.border}`, marginBottom: 12,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 14, fontWeight: 800, color: t.accent }}>LIVE TRADING</span>
          {running && (
            <span style={{
              fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 4,
              color: dryRun ? t.gold : t.long,
              background: dryRun ? t.goldBg : t.longBg,
              animation: "pulse 2s infinite",
            }}>
              {dryRun ? "PAPER" : "LIVE (KORE)"} {paused ? "(PAUSED)" : ""}
            </span>
          )}
          {running && (
            <span style={{ fontSize: 10, color: t.textMuted }}>Scan #{scanNumber}</span>
          )}

          {!running && (
            <>
              <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                {([true, false] as const).map(dr => (
                  <button key={String(dr)} onClick={() => {
                    if (!dr && dryRun) {
                      if (!window.confirm("Switch to LIVE mode? This will place REAL orders on forward test account 203979.")) return;
                    }
                    setDryRun(dr);
                  }}
                    style={{
                      background: dryRun === dr ? (dr ? t.goldBg : t.longBg) : "transparent",
                      color: dryRun === dr ? (dr ? t.gold : t.long) : t.textMuted,
                      border: `1px solid ${dryRun === dr ? (dr ? t.gold : t.long) : t.border}`,
                      borderRadius: 4, padding: "3px 8px", fontSize: 10, fontWeight: 600, cursor: "pointer",
                    }}>
                    {dr ? "Paper" : "Live"}
                  </button>
                ))}
              </div>

              <input
                value={customSymbols}
                onChange={e => setCustomSymbols(e.target.value)}
                placeholder="AAPL,NVDA,TSLA or leave empty"
                style={{
                  background: t.bg, color: t.text, border: `1px solid ${t.border}`,
                  borderRadius: 6, padding: "4px 8px", fontSize: 11, width: 180,
                }}
              />

              {!customSymbols.trim() && (
                <select value={maxSymbols} onChange={e => setMaxSymbols(Number(e.target.value))}
                  style={{ background: t.bg, color: t.text, border: `1px solid ${t.border}`, borderRadius: 6, padding: "4px 6px", fontSize: 11, cursor: "pointer" }}>
                  {[10, 20, 30, 50, 75, 100].map(n => <option key={n} value={n}>{n} symbols</option>)}
                </select>
              )}

              <select value={scanInterval} onChange={e => setScanInterval(Number(e.target.value))}
                style={{ background: t.bg, color: t.text, border: `1px solid ${t.border}`, borderRadius: 6, padding: "4px 6px", fontSize: 11, cursor: "pointer" }}>
                {[
                  { v: 60, l: "1m" }, { v: 120, l: "2m" }, { v: 300, l: "5m" },
                  { v: 600, l: "10m" }, { v: 900, l: "15m" },
                ].map(({ v, l }) => <option key={v} value={v}>{l} interval</option>)}
              </select>
            </>
          )}
        </div>

        <div style={{ display: "flex", gap: 6 }}>
          {!running ? (
            <button onClick={startScanner}
              style={{
                background: t.accent, color: "#fff", border: "none", borderRadius: 6,
                padding: "6px 16px", fontSize: 12, fontWeight: 700, cursor: "pointer",
              }}>
              Start Scanner
            </button>
          ) : (
            <>
              {paused ? (
                <button onClick={resumeScanner}
                  style={{ background: t.long, color: "#fff", border: "none", borderRadius: 6, padding: "5px 12px", fontSize: 11, fontWeight: 700, cursor: "pointer" }}>
                  Resume
                </button>
              ) : (
                <button onClick={pauseScanner}
                  style={{ background: t.gold, color: "#000", border: "none", borderRadius: 6, padding: "5px 12px", fontSize: 11, fontWeight: 700, cursor: "pointer" }}>
                  Pause
                </button>
              )}
              <button onClick={stopScanner}
                style={{ background: t.short, color: "#fff", border: "none", borderRadius: 6, padding: "5px 12px", fontSize: 11, fontWeight: 700, cursor: "pointer" }}>
                Stop
              </button>
            </>
          )}
        </div>
      </div>

      {/* ��─ Stats Row ───────────────────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 8, marginBottom: 12 }}>
        {[
          { label: "P&L", value: `${totalPnl >= 0 ? "+" : ""}$${totalPnl.toFixed(2)}`, color: totalPnl >= 0 ? t.long : t.short },
          { label: "OPEN", value: `${openCount}`, color: openCount > 0 ? t.accent : t.textDim },
          { label: "CLOSED", value: `${closedCount}`, color: t.text },
          { label: "WINS", value: `${wins}`, color: t.long },
          { label: "LOSSES", value: `${losses}`, color: t.short },
          { label: "WIN RATE", value: closedCount > 0 ? `${winRate.toFixed(0)}%` : "--", color: winRate >= 50 ? t.long : winRate > 0 ? t.short : t.textMuted },
          { label: "SCANS", value: `${scanNumber}`, color: t.textDim },
        ].map(({ label, value, color }) => (
          <div key={label} style={{
            background: t.bgCard, borderRadius: 10, border: `1px solid ${t.border}`,
            padding: "10px 12px", textAlign: "center",
          }}>
            <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 600, marginBottom: 4 }}>{label}</div>
            <div style={{ fontSize: 18, fontWeight: 800, color, fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
          </div>
        ))}
      </div>

      {/* ── Price Chart ────────────────────────────────────────────────── */}
      {running && (positions.length > 0 || closedTrades.length > 0 || setups.length > 0) && (
        <div style={{ ...cardStyle, marginBottom: 12 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: t.textMuted }}>PRICE CHART</div>
            <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
              {/* Symbol pills from positions, then setups */}
              {[...new Set([
                ...positions.map(p => p.symbol),
                ...closedTrades.map(tr => tr.symbol),
                ...setups.slice(0, 10).map(s => s.symbol),
              ])].map(sym => (
                <button key={sym} onClick={() => setChartSymbol(sym)}
                  style={{
                    background: chartSymbol === sym ? t.accentLight : "transparent",
                    color: chartSymbol === sym ? t.accent : t.textMuted,
                    border: `1px solid ${chartSymbol === sym ? t.accent : t.border}`,
                    borderRadius: 4, padding: "2px 6px", fontSize: 10, fontWeight: 600,
                    cursor: "pointer",
                  }}>
                  {sym}
                  {positions.find(p => p.symbol === sym) ? " *" : ""}
                </button>
              ))}
            </div>
          </div>
          {chartSymbol ? (
            <LivePriceChart symbol={chartSymbol} positions={positions} closedTrades={closedTrades} t={t} />
          ) : (
            <div style={{ color: t.textMuted, fontSize: 11, textAlign: "center", padding: 40 }}>
              Select a symbol above to view the chart
            </div>
          )}
        </div>
      )}

      {/* ── P&L Charts ─────────────────────────────────────────────────── */}
      {(pnlHistory.length >= 2 || closedTrades.length >= 1) && (
        <div style={{ display: "grid", gridTemplateColumns: closedTrades.length > 0 ? "1fr 1fr" : "1fr", gap: 12, marginBottom: 12 }}>
          {pnlHistory.length >= 2 && (
            <div style={cardStyle}>
              <div style={{ fontSize: 11, fontWeight: 700, color: t.textMuted, marginBottom: 8 }}>REALIZED P&L</div>
              <PnlChart history={pnlHistory} t={t} />
            </div>
          )}
          {closedTrades.length > 0 && (
            <div style={cardStyle}>
              <div style={{ fontSize: 11, fontWeight: 700, color: t.textMuted, marginBottom: 8 }}>TRADE P&L</div>
              <TradePnlBars trades={closedTrades} t={t} />
            </div>
          )}
        </div>
      )}

      {/* ── Tab Bar ─────────────────────────────────────────────────────── */}
      <div style={{
        display: "flex", gap: 2, background: t.border, borderRadius: 8, padding: 3, marginBottom: 12, width: "fit-content",
      }}>
        {([
          { key: "log" as const, label: `Log (${events.length})` },
          { key: "setups" as const, label: `Setups (${setups.length})` },
          { key: "positions" as const, label: `Positions (${openCount})` },
          { key: "closed" as const, label: `Closed (${closedCount})` },
        ]).map(({ key, label }) => (
          <button key={key} onClick={() => setTab(key)}
            style={{
              background: tab === key ? t.bgCard : "transparent",
              color: tab === key ? t.text : t.textMuted,
              border: "none", borderRadius: 6, padding: "5px 14px",
              fontSize: 11, fontWeight: 600, cursor: "pointer",
            }}>
            {label}
          </button>
        ))}
      </div>

      {/* ── Tab Content ─────────────────────────────────────────────────── */}
      <div style={cardStyle}>
        <AnimatePresence mode="wait">
          {/* LOG */}
          {tab === "log" && (
            <motion.div key="log" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <div ref={logRef} style={{
                height: 360, overflowY: "auto", overflowX: "hidden",
                padding: "4px 0",
              }}>
                {events.length === 0 ? (
                  <div style={{ color: t.textMuted, textAlign: "center", padding: 40, fontSize: 12 }}>
                    {running ? "Waiting for events..." : "Start the scanner to see live events"}
                  </div>
                ) : (
                  events.map((ev, i) => <LogRow key={i} event={ev} t={t} />)
                )}
              </div>
            </motion.div>
          )}

          {/* SETUPS */}
          {tab === "setups" && (
            <motion.div key="setups" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              {setups.length === 0 ? (
                <div style={{ color: t.textMuted, textAlign: "center", padding: 40, fontSize: 12 }}>
                  No setups detected yet
                </div>
              ) : (
                <div style={{ overflowX: "auto", maxHeight: 360, overflowY: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${t.border}`, position: "sticky", top: 0, background: t.bgCard }}>
                        {["#", "Symbol", "Bias", "Pattern", "Score", "Entry", "Stop", "T1", "T2", "R:R", "TF"].map(h => (
                          <th key={h} style={{ padding: "6px 8px", textAlign: "left", fontSize: 9, color: t.textMuted, fontWeight: 600 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {setups.map((s, i) => {
                        const biasColor = s.bias === "long" ? t.long : t.short;
                        return (
                          <tr key={i} style={{ borderBottom: `1px solid ${t.border}`, cursor: "pointer" }} onClick={() => setChartSymbol(s.symbol)}>
                            <td style={{ padding: "5px 8px", color: t.textMuted }}>{i + 1}</td>
                            <td style={{ padding: "5px 8px", fontWeight: 700, color: t.accent, textDecoration: "underline" }}>{s.symbol}</td>
                            <td style={{ padding: "5px 8px", color: biasColor, fontWeight: 600, fontSize: 10 }}>{s.bias.toUpperCase()}</td>
                            <td style={{ padding: "5px 8px", color: t.textDim, fontSize: 10 }}>{s.pattern.slice(0, 24)}</td>
                            <td style={{ padding: "5px 8px", fontWeight: 700, color: s.composite_score >= 70 ? t.long : s.composite_score >= 55 ? t.gold : t.textDim }}>
                              {s.composite_score.toFixed(1)}
                            </td>
                            <td style={{ padding: "5px 8px" }}>${s.entry_price.toFixed(2)}</td>
                            <td style={{ padding: "5px 8px", color: t.short }}>${s.stop_loss.toFixed(2)}</td>
                            <td style={{ padding: "5px 8px", color: t.long }}>${s.target_1.toFixed(2)}</td>
                            <td style={{ padding: "5px 8px", color: t.long }}>${s.target_2.toFixed(2)}</td>
                            <td style={{ padding: "5px 8px", color: t.textDim }}>{s.risk_reward.toFixed(1)}</td>
                            <td style={{ padding: "5px 8px", color: t.textMuted, fontSize: 9 }}>{s.timeframe}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </motion.div>
          )}

          {/* POSITIONS */}
          {tab === "positions" && (
            <motion.div key="positions" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              {positions.length === 0 ? (
                <div style={{ color: t.textMuted, textAlign: "center", padding: 40, fontSize: 12 }}>
                  No open positions
                </div>
              ) : (
                <div style={{ overflowX: "auto", maxHeight: 360, overflowY: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${t.border}`, position: "sticky", top: 0, background: t.bgCard }}>
                        {["Symbol", "Bias", "Pattern", "Score", "Entry", "Stop", "T1", "T2", "Shares", "Status"].map(h => (
                          <th key={h} style={{ padding: "6px 8px", textAlign: "left", fontSize: 9, color: t.textMuted, fontWeight: 600 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {positions.map((p, i) => {
                        const biasColor = p.bias === "long" ? t.long : t.short;
                        return (
                          <tr key={i} style={{ borderBottom: `1px solid ${t.border}`, cursor: "pointer" }} onClick={() => setChartSymbol(p.symbol)}>
                            <td style={{ padding: "5px 8px", fontWeight: 700, color: t.accent, textDecoration: "underline" }}>{p.symbol}</td>
                            <td style={{ padding: "5px 8px", color: biasColor, fontSize: 10, fontWeight: 600 }}>{p.bias.toUpperCase()}</td>
                            <td style={{ padding: "5px 8px", color: t.textDim, fontSize: 10 }}>{p.pattern.slice(0, 22)}</td>
                            <td style={{ padding: "5px 8px", fontWeight: 600 }}>{p.score.toFixed(1)}</td>
                            <td style={{ padding: "5px 8px" }}>${p.entry_price.toFixed(2)}</td>
                            <td style={{ padding: "5px 8px", color: t.short }}>${p.stop_loss.toFixed(2)}</td>
                            <td style={{ padding: "5px 8px", color: p.t1_hit ? t.gold : t.textDim }}>{p.t1_hit ? "HIT" : `$${p.target_1.toFixed(2)}`}</td>
                            <td style={{ padding: "5px 8px", color: p.t2_hit ? t.gold : t.textDim }}>{p.t2_hit ? "HIT" : `$${p.target_2.toFixed(2)}`}</td>
                            <td style={{ padding: "5px 8px" }}>{p.shares}</td>
                            <td style={{ padding: "5px 8px", fontSize: 9, fontWeight: 600, color: p.status === "ACTIVE" ? t.long : t.gold }}>{p.status}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </motion.div>
          )}

          {/* CLOSED TRADES */}
          {tab === "closed" && (
            <motion.div key="closed" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              {closedTrades.length === 0 ? (
                <div style={{ color: t.textMuted, textAlign: "center", padding: 40, fontSize: 12 }}>
                  No closed trades yet
                </div>
              ) : (
                <div>
                  <div style={{ overflowX: "auto", maxHeight: 240, overflowY: "auto", marginBottom: 16 }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
                      <thead>
                        <tr style={{ borderBottom: `1px solid ${t.border}`, position: "sticky", top: 0, background: t.bgCard }}>
                          {["Symbol", "Bias", "Pattern", "Entry", "Exit", "P&L", "Reason"].map(h => (
                            <th key={h} style={{ padding: "6px 8px", textAlign: "left", fontSize: 9, color: t.textMuted, fontWeight: 600 }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {closedTrades.map((ct, i) => {
                          const color = ct.pnl >= 0 ? t.long : t.short;
                          return (
                            <tr key={i} style={{ borderBottom: `1px solid ${t.border}`, cursor: "pointer" }} onClick={() => setChartSymbol(ct.symbol)}>
                              <td style={{ padding: "5px 8px", fontWeight: 700, color: t.accent, textDecoration: "underline" }}>{ct.symbol}</td>
                              <td style={{ padding: "5px 8px", color: ct.bias === "long" ? t.long : t.short, fontSize: 10, fontWeight: 600 }}>{ct.bias.toUpperCase()}</td>
                              <td style={{ padding: "5px 8px", color: t.textDim, fontSize: 10 }}>{ct.pattern.slice(0, 22)}</td>
                              <td style={{ padding: "5px 8px" }}>${ct.entry_price.toFixed(2)}</td>
                              <td style={{ padding: "5px 8px" }}>${ct.exit_price.toFixed(2)}</td>
                              <td style={{ padding: "5px 8px", fontWeight: 700, color }}>{ct.pnl >= 0 ? "+" : ""}${ct.pnl.toFixed(2)}</td>
                              <td style={{ padding: "5px 8px", color: t.textMuted, fontSize: 9 }}>{ct.exit_reason}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  {/* Pattern breakdown */}
                  <div style={{ fontSize: 11, fontWeight: 700, color: t.textMuted, marginBottom: 6 }}>BY PATTERN</div>
                  <PatternBreakdown trades={closedTrades} t={t} />
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
