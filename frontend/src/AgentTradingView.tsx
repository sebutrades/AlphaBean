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
  const values = points.map(p => p.total_r);
  const minV = Math.min(0, ...values);
  const maxV = Math.max(0.1, ...values);
  const range = maxV - minV || 1;

  const toX = (i: number) => PAD + (i / (points.length - 1)) * (W - PAD * 2);
  const toY = (v: number) => H - PAD - ((v - minV) / range) * (H - PAD * 2);
  const zeroY = toY(0);

  // Build path
  const pathParts = points.map((p, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(p.total_r).toFixed(1)}`);
  const linePath = pathParts.join(" ");

  // Fill area
  const fillPath = linePath + ` L${toX(points.length - 1).toFixed(1)},${zeroY.toFixed(1)} L${toX(0).toFixed(1)},${zeroY.toFixed(1)} Z`;

  const lastR = values[values.length - 1];
  const color = lastR >= 0 ? t.long : t.short;

  // Time labels
  const labels: { x: number; text: string }[] = [];
  const step = Math.max(1, Math.floor(points.length / 6));
  for (let i = 0; i < points.length; i += step) {
    const time = points[i].time.slice(11, 16); // HH:MM
    labels.push({ x: toX(i), text: time });
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
      <path d={linePath} fill="none" stroke={color} strokeWidth={2} />
      {/* Current dot */}
      <circle cx={toX(points.length - 1)} cy={toY(lastR)} r={4} fill={color} />
      {/* Y-axis labels */}
      {[minV, minV + range / 2, maxV].map((v, i) => (
        <text key={i} x={PAD - 5} y={toY(v)} textAnchor="end" fill={t.chartText} fontSize={10} dominantBaseline="middle">
          {v.toFixed(1)}R
        </text>
      ))}
      {/* X-axis labels */}
      {labels.map((l, i) => (
        <text key={i} x={l.x} y={H - 5} textAnchor="middle" fill={t.chartText} fontSize={9}>{l.text}</text>
      ))}
    </svg>
  );
}

// ── Agent Log Entry ─────────────────────────────────────────────────────────

function LogEntry({ event, t }: { event: AgentEvent; t: Theme }) {
  const time = event.timestamp.slice(11, 16);
  const d = event.data;

  const colors: Record<string, string> = {
    setup_detected: t.gold,
    agent_thinking: t.purple,
    agent_verdict: d.verdict === "CONFIRMED" ? t.long : d.verdict === "DENIED" ? t.short : t.gold,
    trade_open: t.long,
    trade_close: (d.realized_r ?? 0) > 0 ? t.long : t.short,
    position_update: t.textMuted,
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
      message = `=== DAY START: ${d.date} | ${d.symbols} symbols | ${d.total_bars} bars ===`;
      break;
    case "day_end":
      message = `=== DAY END | Trades: ${d.total_trades} | R: ${d.cumulative_r?.toFixed(2)} | Wins: ${d.wins} Losses: ${d.losses} ===`;
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
  const [dates, setDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [speed, setSpeed] = useState(3);
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [paused, setPaused] = useState(false);

  // Data
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [pnlHistory, setPnlHistory] = useState<PnlPoint[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [closedTrades, setClosedTrades] = useState<ClosedTrade[]>([]);
  const [progress, setProgress] = useState({ bar: 0, total: 1, time: "" });
  const [dayStats, setDayStats] = useState<Record<string, any>>({});

  const wsRef = useRef<WebSocket | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  // Fetch available dates
  useEffect(() => {
    fetch(`${API}/api/agent-trading/dates`)
      .then(r => r.json())
      .then(d => {
        if (d.dates?.length) {
          setDates(d.dates);
          setSelectedDate(d.dates[d.dates.length - 1]); // most recent
        }
      })
      .catch(() => {});
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

      case "setup_detected":
      case "agent_thinking":
      case "agent_verdict":
      case "day_start":
      case "day_end":
      case "error":
        setEvents(prev => [...prev.slice(-500), event]); // keep last 500
        if (event.type === "day_end") {
          setDayStats(d);
          setRunning(false);
        }
        break;

      case "trade_open":
        setEvents(prev => [...prev.slice(-500), event]);
        setPositions(prev => [...prev, d as unknown as Position]);
        break;

      case "trade_close":
        setEvents(prev => [...prev.slice(-500), event]);
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

  const startSim = () => {
    if (!connected) connect();
    // Reset state
    setEvents([]);
    setPnlHistory([]);
    setPositions([]);
    setClosedTrades([]);
    setDayStats({});

    // Wait for connection then send start
    const trySend = () => {
      const ws = wsRef.current;
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          action: "start",
          date: selectedDate,
          speed: speed,
          use_agents: false,
          capital: 100000,
          min_score: 50,
        }));
        setRunning(true);
        setPaused(false);
      } else {
        setTimeout(trySend, 200);
      }
    };
    if (!connected) {
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

  // Stats
  const totalR = pnlHistory.length ? pnlHistory[pnlHistory.length - 1].total_r : 0;
  const equity = pnlHistory.length ? pnlHistory[pnlHistory.length - 1].equity : 100000;
  const heat = pnlHistory.length ? pnlHistory[pnlHistory.length - 1].heat_pct : 0;
  const wins = closedTrades.filter(t => t.realized_r > 0).length;
  const dayPnl = closedTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
  const losses = closedTrades.filter(t => t.realized_r <= 0).length;
  const progressPct = progress.total > 0 ? (progress.bar / progress.total) * 100 : 0;

  return (
    <div style={{ padding: 0 }}>
      {/* Header Controls */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "12px 16px", background: t.bgCard, borderRadius: 12,
        border: `1px solid ${t.border}`, marginBottom: 12,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 14, fontWeight: 800, color: t.accent }}>AGENT TRADING</span>

          <select
            value={selectedDate}
            onChange={e => setSelectedDate(e.target.value)}
            disabled={running}
            style={{
              background: t.bg, color: t.text, border: `1px solid ${t.border}`,
              borderRadius: 6, padding: "4px 8px", fontSize: 12, cursor: "pointer",
            }}
          >
            {dates.map(d => <option key={d} value={d}>{d}</option>)}
          </select>

          {!running ? (
            <button
              onClick={startSim}
              disabled={!selectedDate}
              style={{
                background: t.accent, color: "#fff", border: "none", borderRadius: 6,
                padding: "5px 16px", fontSize: 12, fontWeight: 700, cursor: "pointer",
              }}
            >
              START
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
          {/* Speed control */}
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ fontSize: 10, color: t.textMuted }}>Speed:</span>
            {[1, 3, 5, 10, 25].map(s => (
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

          {/* Connection indicator */}
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
              Bar {progress.bar}/{progress.total} | {progress.time.slice(11, 16)}
            </span>
            <span style={{ fontSize: 10, color: t.textMuted }}>{progressPct.toFixed(0)}%</span>
          </div>
        </div>
      )}

      {/* Stats Row */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 8, marginBottom: 12,
      }}>
        {[
          { label: "EQUITY", value: `$${equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, color: t.text },
          { label: "DAY P&L", value: `${dayPnl >= 0 ? "+" : ""}$${dayPnl.toFixed(0)}`, color: dayPnl >= 0 ? t.long : t.short },
          { label: "TOTAL R", value: `${totalR >= 0 ? "+" : ""}${totalR.toFixed(2)}R`, color: totalR >= 0 ? t.long : t.short },
          { label: "W / L", value: `${wins} / ${losses}`, color: wins > losses ? t.long : wins < losses ? t.short : t.textDim },
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
            <div style={{ fontSize: 11, fontWeight: 700, color: t.textDim, marginBottom: 8 }}>P&L CURVE (R-Multiple)</div>
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
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
                <thead>
                  <tr style={{ borderBottom: `1px solid ${t.border}` }}>
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
    </div>
  );
}
