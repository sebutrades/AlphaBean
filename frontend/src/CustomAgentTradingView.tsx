import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";

const API = "http://localhost:8000";

type Theme = {
  bg: string; bgCard: string; bgHover: string; border: string; borderLight: string;
  text: string; textDim: string; textMuted: string;
  accent: string; accentLight: string;
  long: string; longBg: string; short: string; shortBg: string;
  gold: string; goldBg: string; purple: string;
  chartBg: string; chartGrid: string; chartText: string;
};

// ── Types ────────────────────────────────────────────────────────────────────

type Strategy = { name: string; direction: string; category: string; strategy_type: string };
type StrategyGroups = Record<string, string[]>;

type RunConfig = {
  name: string;
  description: string;
  starting_capital: number;
  use_agents: boolean;
  allowed_strategies: string[];
  min_composite_score: number;
  max_trades_per_scan: number;
  max_trades_per_day: number;
  playback_speed: number;
  sizing: { mode: string; base_risk_pct: number; max_heat_pct: number; max_positions: number; drawdown_reduction: boolean };
  models: { analyst: string; portfolio_manager: string; risk_manager: string; strategist: string };
  deliberation: { mode: string; max_candidates: number; max_api_calls_per_candidate: number };
};

type RunSummary = {
  run_id: string;
  name: string;
  status: string;
  created_at: string;
  progress?: { day: number; total: number };
  config_summary: { starting_capital: number; use_agents: boolean; deliberation_mode: string; strategies: number | string; sizing_mode: string };
  stats: Record<string, number>;
};

type PnlPoint = { timestamp: string; equity: number; cumulative_pnl: number; unrealized_pnl: number; total_pnl: number; positions: number; closed_trades: number; heat_pct: number; day_number: number };

type AgentLog = { type: string; agent?: string; symbol?: string; pattern?: string; verdict?: string; reasoning?: string; message?: string; action?: string };

type TradeEvent = { id: string; symbol: string; pattern: string; bias: string; entry: number; exit?: number; pnl?: number; realized_r?: number; outcome?: string; dollar_risk: number; size_modifier?: number; sizing?: string };

// ── Model options ────────────────────────────────────────────────────────────

const MODEL_OPTIONS = [
  { value: "qwen3:8b", label: "Qwen3 8B (free, local)" },
  { value: "claude-haiku-4-5-20251001", label: "Haiku 4.5 ($)" },
  { value: "claude-sonnet-4-6-20250514", label: "Sonnet 4.6 ($$)" },
  { value: "claude-opus-4-6-20250610", label: "Opus 4.6 ($$$)" },
];

const DELIB_MODES = [
  { value: "quick", label: "Quick", desc: "1 analyst, fast, ~3-7 API calls" },
  { value: "standard", label: "Standard", desc: "Analyst + Senior review, ~7-12 calls" },
  { value: "thorough", label: "Thorough", desc: "3 analysts vote + strategist, ~12-20 calls" },
];

const SIZING_MODES = [
  { value: "fixed", label: "Fixed", desc: "Always risk % of starting capital" },
  { value: "compound", label: "Compound", desc: "Risk % of current equity (grows with profits)" },
  { value: "adaptive", label: "Adaptive", desc: "Compound + scale by strategy win rate" },
];

const defaultConfig: RunConfig = {
  name: "",
  description: "",
  starting_capital: 100000,
  use_agents: true,
  allowed_strategies: [],
  min_composite_score: 50,
  max_trades_per_scan: 1,
  max_trades_per_day: 5,
  playback_speed: 10,
  sizing: { mode: "compound", base_risk_pct: 1.0, max_heat_pct: 6.0, max_positions: 10, drawdown_reduction: true },
  models: { analyst: "qwen3:8b", portfolio_manager: "qwen3:8b", risk_manager: "qwen3:8b", strategist: "qwen3:8b" },
  deliberation: { mode: "standard", max_candidates: 5, max_api_calls_per_candidate: 10 },
};

// ── SVG Equity Chart ─────────────────────────────────────────────────────────

function EquityChart({ points, t }: { points: PnlPoint[]; t: Theme }) {
  if (points.length < 2) return null;
  const W = 800, H = 200, PAD = 50;

  const vals = points.map(p => p.total_pnl);
  const minV = Math.min(0, ...vals);
  const maxV = Math.max(100, ...vals);
  const range = maxV - minV || 1;

  const toX = (i: number) => PAD + (i / (points.length - 1)) * (W - PAD * 2);
  const toY = (v: number) => H - PAD - ((v - minV) / range) * (H - PAD * 2);

  const path = points.map((p, i) => `${i === 0 ? "M" : "L"} ${toX(i).toFixed(1)} ${toY(p.total_pnl).toFixed(1)}`).join(" ");
  const lastPnl = vals[vals.length - 1];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 200, background: t.chartBg, borderRadius: 8 }}>
      {/* Zero line */}
      <line x1={PAD} x2={W - PAD} y1={toY(0)} y2={toY(0)} stroke={t.chartGrid} strokeDasharray="4,4" />
      {/* Y-axis labels */}
      {[minV, 0, maxV].map((v, i) => (
        <text key={i} x={PAD - 5} y={toY(v)} fill={t.chartText} fontSize={10} textAnchor="end" dominantBaseline="middle">
          ${(v / 1000).toFixed(1)}k
        </text>
      ))}
      {/* Line */}
      <path d={path} fill="none" stroke={lastPnl >= 0 ? t.long : t.short} strokeWidth={2} />
      {/* Current value label */}
      <text x={W - PAD + 5} y={toY(lastPnl)} fill={lastPnl >= 0 ? t.long : t.short} fontSize={11} dominantBaseline="middle">
        ${lastPnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
      </text>
    </svg>
  );
}

// ── Strategy Selector ────────────────────────────────────────────────────────

function StrategySelector({ strategies, groups, selected, onChange, t }: {
  strategies: Strategy[];
  groups: StrategyGroups;
  selected: string[];
  onChange: (s: string[]) => void;
  t: Theme;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const selectedSet = new Set(selected);
  const allSelected = selected.length === 0; // empty = all

  const toggle = (name: string) => {
    if (allSelected) {
      // First selection: select ONLY this one
      onChange([name]);
    } else if (selectedSet.has(name)) {
      const next = selected.filter(s => s !== name);
      onChange(next); // if empty, means "all"
    } else {
      onChange([...selected, name]);
    }
  };

  const toggleGroup = (group: string) => {
    const names = groups[group] || [];
    const allInGroup = names.every(n => selectedSet.has(n));
    if (allSelected || !allInGroup) {
      const base = allSelected ? [] : [...selected];
      const toAdd = names.filter(n => !selectedSet.has(n));
      onChange([...base, ...toAdd]);
    } else {
      onChange(selected.filter(s => !names.includes(s)));
    }
  };

  return (
    <div style={{ fontSize: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ color: t.textDim }}>
          {allSelected ? "All 51 strategies" : `${selected.length} selected`}
        </span>
        {!allSelected && (
          <button onClick={() => onChange([])} style={{ background: "none", border: "none", color: t.accent, cursor: "pointer", fontSize: 11 }}>
            Reset to All
          </button>
        )}
      </div>
      {Object.entries(groups).map(([group, names]) => {
        const count = allSelected ? names.length : names.filter(n => selectedSet.has(n)).length;
        return (
          <div key={group} style={{ marginBottom: 4 }}>
            <div
              onClick={() => setExpanded(expanded === group ? null : group)}
              style={{ display: "flex", justifyContent: "space-between", cursor: "pointer", padding: "4px 6px", background: t.bgHover, borderRadius: 4, color: t.text }}
            >
              <span>{group.replace("_", " ")} ({names.length})</span>
              <span style={{ color: count > 0 && !allSelected ? t.accent : t.textMuted }}>{count}/{names.length}</span>
            </div>
            {expanded === group && (
              <div style={{ padding: "4px 8px", display: "flex", flexWrap: "wrap", gap: 4 }}>
                <button
                  onClick={() => toggleGroup(group)}
                  style={{ fontSize: 10, padding: "2px 6px", background: t.accent, color: "#fff", border: "none", borderRadius: 3, cursor: "pointer" }}
                >
                  Toggle All
                </button>
                {names.map(name => {
                  const active = allSelected || selectedSet.has(name);
                  return (
                    <button
                      key={name}
                      onClick={() => toggle(name)}
                      style={{
                        fontSize: 10, padding: "2px 6px", borderRadius: 3, cursor: "pointer",
                        background: active ? t.accentLight : t.bgCard,
                        color: active ? t.accent : t.textMuted,
                        border: `1px solid ${active ? t.accent : t.border}`,
                      }}
                    >
                      {name}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Run History ──────────────────────────────────────────────────────────────

function RunHistory({ runs, onSelect, onDelete, t }: {
  runs: RunSummary[];
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  t: Theme;
}) {
  if (runs.length === 0) return <div style={{ color: t.textMuted, fontSize: 12, padding: 12 }}>No saved runs yet</div>;

  return (
    <div style={{ maxHeight: 300, overflowY: "auto" }}>
      {runs.map(run => {
        const stats = run.stats || {};
        const statusColor = run.status === "completed" ? t.long : run.status === "running" ? t.accent : run.status === "error" ? t.short : t.textMuted;
        return (
          <div
            key={run.run_id}
            onClick={() => onSelect(run.run_id)}
            style={{ padding: "8px 10px", borderBottom: `1px solid ${t.border}`, cursor: "pointer", fontSize: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}
          >
            <div>
              <div style={{ color: t.text, fontWeight: 600 }}>{run.name || run.run_id.slice(4, 19)}</div>
              <div style={{ color: t.textMuted, fontSize: 10 }}>
                <span style={{ color: statusColor }}>{run.status}</span>
                {" | "}
                {run.config_summary?.strategies === "all" ? "all strategies" : `${run.config_summary?.strategies} strategies`}
                {" | "}
                {run.config_summary?.deliberation_mode || "det"}
                {run.progress ? ` | Day ${run.progress.day}/${run.progress.total}` : ""}
              </div>
              {stats.total_trades !== undefined && (
                <div style={{ color: t.textDim, fontSize: 10 }}>
                  {stats.total_trades} trades | {stats.win_rate?.toFixed(0)}% win | P&L: ${stats.total_pnl?.toLocaleString()}
                </div>
              )}
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(run.run_id); }}
              style={{ background: "none", border: "none", color: t.textMuted, cursor: "pointer", fontSize: 14 }}
            >
              x
            </button>
          </div>
        );
      })}
    </div>
  );
}

// ── Agent Log ────────────────────────────────────────────────────────────────

function AgentLogView({ logs, t }: { logs: AgentLog[]; t: Theme }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [logs.length]);

  return (
    <div style={{ maxHeight: 300, overflowY: "auto", fontSize: 11, fontFamily: "monospace" }}>
      {logs.slice(-50).map((log, i) => {
        const color = log.verdict === "CONFIRMED" || log.verdict?.includes("APPROVE") ? t.long
          : log.verdict === "DENIED" || log.verdict?.includes("REJECT") ? t.short
          : log.type === "agent_thinking" ? t.accent
          : t.textDim;
        return (
          <div key={i} style={{ padding: "2px 6px", borderBottom: `1px solid ${t.border}`, color }}>
            <span style={{ color: t.textMuted }}>[{log.agent || "sys"}]</span>{" "}
            {log.symbol && <span style={{ fontWeight: 600 }}>{log.symbol} </span>}
            {log.verdict && <span style={{ fontWeight: 700 }}>{log.verdict} </span>}
            {log.message || log.reasoning?.slice(0, 200) || log.action || ""}
          </div>
        );
      })}
      <div ref={endRef} />
    </div>
  );
}

// ── Trade Log ────────────────────────────────────────────────────────────────

function TradeLog({ trades, t }: { trades: TradeEvent[]; t: Theme }) {
  if (trades.length === 0) return null;
  return (
    <div style={{ maxHeight: 250, overflowY: "auto", fontSize: 11 }}>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ color: t.textMuted, borderBottom: `1px solid ${t.border}` }}>
            <th style={{ textAlign: "left", padding: 4 }}>ID</th>
            <th style={{ textAlign: "left", padding: 4 }}>Symbol</th>
            <th style={{ textAlign: "left", padding: 4 }}>Pattern</th>
            <th style={{ textAlign: "right", padding: 4 }}>P&L</th>
            <th style={{ textAlign: "right", padding: 4 }}>R</th>
            <th style={{ textAlign: "center", padding: 4 }}>Result</th>
          </tr>
        </thead>
        <tbody>
          {trades.slice(-50).reverse().map((t2, i) => {
            const pnl = t2.pnl || 0;
            const color = pnl > 0 ? t.long : pnl < 0 ? t.short : t.textDim;
            return (
              <tr key={i} style={{ borderBottom: `1px solid ${t.border}` }}>
                <td style={{ padding: 4, color: t.textMuted }}>{t2.id}</td>
                <td style={{ padding: 4, color: t.text, fontWeight: 600 }}>{t2.symbol}</td>
                <td style={{ padding: 4, color: t.textDim }}>{t2.pattern}</td>
                <td style={{ padding: 4, color, textAlign: "right" }}>${pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                <td style={{ padding: 4, color, textAlign: "right" }}>{(t2.realized_r || 0).toFixed(2)}R</td>
                <td style={{ padding: 4, textAlign: "center", color }}>{t2.outcome || "-"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Saved Run Detail ─────────────────────────────────────────────────────────

function SavedRunView({ data, t, onBack }: { data: any; t: Theme; onBack: () => void }) {
  const stats = data.stats || {};
  const eqCurve = data.equity_curve || [];
  const trades = data.closed_trades || [];
  const strategies = data.strategy_summary || [];
  const totalPnl = stats.total_pnl || 0;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <button onClick={onBack} style={{ background: "none", border: `1px solid ${t.border}`, color: t.text, padding: "4px 12px", borderRadius: 6, cursor: "pointer" }}>
          Back
        </button>
        <span style={{ color: t.textMuted, fontSize: 12 }}>{data.run_id}</span>
      </div>

      {/* Stats row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8, marginBottom: 12 }}>
        {[
          { label: "Total P&L", value: `$${totalPnl.toLocaleString()}`, color: totalPnl >= 0 ? t.long : t.short },
          { label: "Trades", value: stats.total_trades || 0, color: t.text },
          { label: "Win Rate", value: `${(stats.win_rate || 0).toFixed(0)}%`, color: (stats.win_rate || 0) >= 50 ? t.long : t.short },
          { label: "Profit Factor", value: (stats.profit_factor || 0).toFixed(2), color: (stats.profit_factor || 0) >= 1.5 ? t.long : t.short },
          { label: "Avg R", value: (stats.avg_r || 0).toFixed(3), color: (stats.avg_r || 0) >= 0 ? t.long : t.short },
        ].map((s, i) => (
          <div key={i} style={{ background: t.bgCard, padding: "8px 12px", borderRadius: 6, textAlign: "center", border: `1px solid ${t.border}` }}>
            <div style={{ color: t.textMuted, fontSize: 10 }}>{s.label}</div>
            <div style={{ color: s.color, fontSize: 18, fontWeight: 700 }}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Equity curve from saved data */}
      {eqCurve.length > 1 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ color: t.textDim, fontSize: 11, marginBottom: 4 }}>Equity Curve ({eqCurve.length} days)</div>
          <EquityChartFromCurve data={eqCurve} t={t} />
        </div>
      )}

      {/* Strategy breakdown */}
      {strategies.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ color: t.textDim, fontSize: 11, marginBottom: 4 }}>Strategy Performance</div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
            <thead>
              <tr style={{ color: t.textMuted, borderBottom: `1px solid ${t.border}` }}>
                <th style={{ textAlign: "left", padding: 4 }}>Strategy</th>
                <th style={{ textAlign: "right", padding: 4 }}>Trades</th>
                <th style={{ textAlign: "right", padding: 4 }}>Win%</th>
                <th style={{ textAlign: "right", padding: 4 }}>Total R</th>
                <th style={{ textAlign: "right", padding: 4 }}>Size Mult</th>
              </tr>
            </thead>
            <tbody>
              {strategies.map((s: any, i: number) => (
                <tr key={i} style={{ borderBottom: `1px solid ${t.border}` }}>
                  <td style={{ padding: 4, color: t.text }}>{s.strategy}</td>
                  <td style={{ padding: 4, color: t.textDim, textAlign: "right" }}>{s.trades}</td>
                  <td style={{ padding: 4, color: s.win_rate >= 50 ? t.long : t.short, textAlign: "right" }}>{s.win_rate}%</td>
                  <td style={{ padding: 4, color: s.total_r >= 0 ? t.long : t.short, textAlign: "right" }}>{s.total_r.toFixed(2)}</td>
                  <td style={{ padding: 4, color: t.accent, textAlign: "right" }}>{s.size_multiplier}x</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Trade log */}
      <div style={{ color: t.textDim, fontSize: 11, marginBottom: 4 }}>Trades ({trades.length})</div>
      <TradeLog trades={trades} t={t} />
    </div>
  );
}

function EquityChartFromCurve({ data, t }: { data: any[]; t: Theme }) {
  const W = 800, H = 180, PAD = 50;
  const vals = data.map((d: any) => d.cumulative_pnl || 0);
  const minV = Math.min(0, ...vals);
  const maxV = Math.max(100, ...vals);
  const range = maxV - minV || 1;
  const toX = (i: number) => PAD + (i / (data.length - 1)) * (W - PAD * 2);
  const toY = (v: number) => H - PAD - ((v - minV) / range) * (H - PAD * 2);
  const path = vals.map((v: number, i: number) => `${i === 0 ? "M" : "L"} ${toX(i).toFixed(1)} ${toY(v).toFixed(1)}`).join(" ");
  const last = vals[vals.length - 1];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 180, background: t.chartBg, borderRadius: 8 }}>
      <line x1={PAD} x2={W - PAD} y1={toY(0)} y2={toY(0)} stroke={t.chartGrid} strokeDasharray="4,4" />
      {[minV, 0, maxV].map((v, i) => (
        <text key={i} x={PAD - 5} y={toY(v)} fill={t.chartText} fontSize={10} textAnchor="end" dominantBaseline="middle">
          ${(v / 1000).toFixed(1)}k
        </text>
      ))}
      <path d={path} fill="none" stroke={last >= 0 ? t.long : t.short} strokeWidth={2} />
    </svg>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────

export default function CustomAgentTradingView({ t }: { t: Theme }) {
  // Config state
  const [config, setConfig] = useState<RunConfig>({ ...defaultConfig });
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [groups, setGroups] = useState<StrategyGroups>({});
  const [configOpen, setConfigOpen] = useState(true);

  // Run state
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [pnlPoints, setPnlPoints] = useState<PnlPoint[]>([]);
  const [agentLogs, setAgentLogs] = useState<AgentLog[]>([]);
  const [closedTrades, setClosedTrades] = useState<TradeEvent[]>([]);
  const [dayInfo, setDayInfo] = useState<{ date: string; day_number: number; total_days?: number } | null>(null);
  const [stats, setStats] = useState<Record<string, number>>({});

  // Run history
  const [savedRuns, setSavedRuns] = useState<RunSummary[]>([]);
  const [viewingRun, setViewingRun] = useState<any>(null);
  const [apiKeySet, setApiKeySet] = useState<boolean | null>(null);

  // WebSocket
  const wsRef = useRef<WebSocket | null>(null);

  // Load strategies, runs, and API status on mount
  useEffect(() => {
    fetch(`${API}/api/custom-trading/strategies`)
      .then(r => r.json())
      .then(d => { setStrategies(d.strategies || []); setGroups(d.groups || {}); })
      .catch(() => {});
    fetch(`${API}/api/custom-trading/status`)
      .then(r => r.json())
      .then(d => setApiKeySet(d.api_key_set))
      .catch(() => {});
    loadRuns();
  }, []);

  const loadRuns = () => {
    fetch(`${API}/api/custom-trading/runs`)
      .then(r => r.json())
      .then(d => setSavedRuns(d.saved || []))
      .catch(() => {});
  };

  // Connect WebSocket to active run
  const connectWs = useCallback((runId: string) => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(`ws://localhost:8000/ws/custom-trading/${runId}`);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      const event = JSON.parse(e.data);
      handleEvent(event);
    };

    ws.onclose = () => {
      // Don't clear state — run continues server-side
    };

    ws.onerror = () => {};
  }, []);

  const handleEvent = useCallback((event: any) => {
    switch (event.type) {
      case "run_start":
        setDayInfo({ date: "", day_number: 0, total_days: event.total_days });
        break;
      case "day_start":
        setDayInfo({ date: event.date, day_number: event.day_number, total_days: dayInfo?.total_days });
        break;
      case "pnl":
        setPnlPoints(prev => {
          const next = [...prev, {
            timestamp: event.timestamp, equity: event.equity,
            cumulative_pnl: event.cumulative_pnl, unrealized_pnl: event.unrealized_pnl,
            total_pnl: event.total_pnl, positions: event.positions,
            closed_trades: event.closed_trades, heat_pct: event.heat_pct,
            day_number: event.day_number,
          }];
          return next.length > 2000 ? next.slice(-1500) : next;
        });
        setStats({
          total_pnl: event.total_pnl, cumulative_pnl: event.cumulative_pnl,
          equity: event.equity, positions: event.positions,
          closed_trades: event.closed_trades, heat_pct: event.heat_pct,
        });
        break;
      case "agent_thinking":
      case "agent_verdict":
        setAgentLogs(prev => {
          const next = [...prev, event];
          return next.length > 200 ? next.slice(-150) : next;
        });
        break;
      case "trade_open":
        setClosedTrades(prev => [...prev, event]);
        break;
      case "trade_close":
        setClosedTrades(prev => {
          const next = prev.map(t2 => t2.id === event.id ? { ...t2, ...event } : t2);
          // If not found (shouldn't happen), add it
          if (!prev.find(t2 => t2.id === event.id)) next.push(event);
          return next;
        });
        break;
      case "day_end":
        setDayInfo(prev => prev ? { ...prev, date: event.date, day_number: event.day_number } : null);
        break;
      case "run_end":
        setRunning(false);
        loadRuns();
        break;
    }
  }, [dayInfo?.total_days]);

  // Start a new run
  const startRun = async () => {
    try {
      const resp = await fetch(`${API}/api/custom-trading/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      const data = await resp.json();
      if (data.run_id) {
        setActiveRunId(data.run_id);
        setRunning(true);
        setPnlPoints([]);
        setAgentLogs([]);
        setClosedTrades([]);
        setConfigOpen(false);
        setViewingRun(null);
        connectWs(data.run_id);
      }
    } catch (e) {
      console.error("Failed to start run:", e);
    }
  };

  const stopRun = async () => {
    if (activeRunId) {
      await fetch(`${API}/api/custom-trading/stop/${activeRunId}`, { method: "POST" });
      setRunning(false);
    }
  };

  const pauseRun = async () => {
    if (activeRunId) await fetch(`${API}/api/custom-trading/pause/${activeRunId}`, { method: "POST" });
  };

  const resumeRun = async () => {
    if (activeRunId) await fetch(`${API}/api/custom-trading/resume/${activeRunId}`, { method: "POST" });
  };

  const setSpeed = (speed: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "set_speed", speed }));
    }
    setConfig(c => ({ ...c, playback_speed: speed }));
  };

  const viewRun = async (runId: string) => {
    const resp = await fetch(`${API}/api/custom-trading/runs/${runId}`);
    const data = await resp.json();
    if (data.run_id || data.stats) {
      setViewingRun(data);
    }
  };

  const deleteRun = async (runId: string) => {
    await fetch(`${API}/api/custom-trading/runs/${runId}`, { method: "DELETE" });
    loadRuns();
    if (viewingRun?.run_id === runId) setViewingRun(null);
  };

  // Cleanup on unmount (but DON'T stop the run!)
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const totalPnl = stats.total_pnl || 0;
  const closedCount = stats.closed_trades || 0;

  // ── Viewing a saved run ───────────────────────────────────────────────────

  if (viewingRun) {
    return (
      <div style={{ padding: 16, maxWidth: 1000, margin: "0 auto" }}>
        <SavedRunView data={viewingRun} t={t} onBack={() => setViewingRun(null)} />
      </div>
    );
  }

  // ── Main View ─────────────────────────────────────────────────────────────

  return (
    <div style={{ padding: 16, maxWidth: 1100, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div>
          <span style={{ fontSize: 20, fontWeight: 700, color: t.text }}>Custom Agent Trading</span>
          <span style={{ color: t.textMuted, fontSize: 12, marginLeft: 8 }}>Background-persistent runs</span>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {running && (
            <>
              <button onClick={pauseRun} style={btnStyle(t)}>Pause</button>
              <button onClick={resumeRun} style={btnStyle(t)}>Resume</button>
              <button onClick={stopRun} style={{ ...btnStyle(t), borderColor: t.short, color: t.short }}>Stop</button>
            </>
          )}
          <button onClick={() => setConfigOpen(!configOpen)} style={btnStyle(t)}>
            {configOpen ? "Hide Config" : "Show Config"}
          </button>
        </div>
      </div>

      {/* Stats bar (when running) */}
      {(running || closedCount > 0) && (
        <div style={{ display: "flex", gap: 12, marginBottom: 12, fontSize: 12 }}>
          {dayInfo && (
            <span style={{ color: t.textMuted }}>
              Day {dayInfo.day_number}{dayInfo.total_days ? `/${dayInfo.total_days}` : ""} | {dayInfo.date}
            </span>
          )}
          <span style={{ color: totalPnl >= 0 ? t.long : t.short, fontWeight: 700 }}>
            P&L: ${totalPnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
          <span style={{ color: t.textDim }}>Trades: {closedCount}</span>
          <span style={{ color: t.textDim }}>Positions: {stats.positions || 0}</span>
          <span style={{ color: t.textDim }}>Heat: {(stats.heat_pct || 0).toFixed(1)}%</span>
          {running && (
            <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
              {[1, 5, 10, 25, 50].map(s => (
                <button key={s} onClick={() => setSpeed(s)} style={{
                  fontSize: 10, padding: "2px 6px", borderRadius: 3, cursor: "pointer",
                  background: config.playback_speed === s ? t.accent : t.bgCard,
                  color: config.playback_speed === s ? "#fff" : t.textMuted,
                  border: `1px solid ${t.border}`,
                }}>
                  {s}x
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: configOpen ? "340px 1fr" : "1fr", gap: 12 }}>
        {/* Config panel */}
        {configOpen && (
          <div style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 8, padding: 12, maxHeight: "80vh", overflowY: "auto" }}>
            {/* Run name */}
            <Section label="Run Name" t={t}>
              <input
                value={config.name}
                onChange={e => setConfig(c => ({ ...c, name: e.target.value }))}
                placeholder="e.g., Bull Flag Only - Haiku"
                style={inputStyle(t)}
              />
            </Section>

            {/* Capital & Risk */}
            <Section label="Capital & Risk" t={t}>
              <Row t={t}>
                <LabeledInput label="Starting Capital" value={config.starting_capital} onChange={v => setConfig(c => ({ ...c, starting_capital: v }))} t={t} prefix="$" />
                <LabeledInput label="Risk per Trade %" value={config.sizing.base_risk_pct} onChange={v => setConfig(c => ({ ...c, sizing: { ...c.sizing, base_risk_pct: v } }))} t={t} step={0.25} />
              </Row>
              <Row t={t}>
                <LabeledInput label="Max Heat %" value={config.sizing.max_heat_pct} onChange={v => setConfig(c => ({ ...c, sizing: { ...c.sizing, max_heat_pct: v } }))} t={t} />
                <LabeledInput label="Max Positions" value={config.sizing.max_positions} onChange={v => setConfig(c => ({ ...c, sizing: { ...c.sizing, max_positions: v } }))} t={t} />
              </Row>
            </Section>

            {/* Sizing mode */}
            <Section label="Position Sizing" t={t}>
              {SIZING_MODES.map(m => (
                <label key={m.value} style={{ display: "flex", alignItems: "flex-start", gap: 6, marginBottom: 4, cursor: "pointer" }}>
                  <input
                    type="radio" checked={config.sizing.mode === m.value}
                    onChange={() => setConfig(c => ({ ...c, sizing: { ...c.sizing, mode: m.value } }))}
                  />
                  <div>
                    <div style={{ color: t.text, fontSize: 12, fontWeight: 600 }}>{m.label}</div>
                    <div style={{ color: t.textMuted, fontSize: 10 }}>{m.desc}</div>
                  </div>
                </label>
              ))}
            </Section>

            {/* Agent mode */}
            <Section label="Agent Mode" t={t}>
              <label style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8, cursor: "pointer" }}>
                <input type="checkbox" checked={config.use_agents} onChange={e => setConfig(c => ({ ...c, use_agents: e.target.checked }))} />
                <span style={{ color: t.text, fontSize: 12 }}>Enable AI Agents</span>
              </label>

              {config.use_agents && apiKeySet === false && (
                <div style={{ background: t.shortBg, border: `1px solid ${t.short}`, borderRadius: 6, padding: "6px 10px", marginBottom: 8, fontSize: 11, color: t.short }}>
                  ANTHROPIC_API_KEY not set — agents will fall back to deterministic mode.
                  Set the env var and restart the server to use agents.
                </div>
              )}

              {config.use_agents && (
                <>
                  {/* Deliberation mode */}
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ color: t.textMuted, fontSize: 10, marginBottom: 4 }}>Deliberation Depth</div>
                    {DELIB_MODES.map(m => (
                      <label key={m.value} style={{ display: "flex", alignItems: "flex-start", gap: 6, marginBottom: 4, cursor: "pointer" }}>
                        <input
                          type="radio" checked={config.deliberation.mode === m.value}
                          onChange={() => setConfig(c => ({ ...c, deliberation: { ...c.deliberation, mode: m.value } }))}
                        />
                        <div>
                          <div style={{ color: t.text, fontSize: 12, fontWeight: 600 }}>{m.label}</div>
                          <div style={{ color: t.textMuted, fontSize: 10 }}>{m.desc}</div>
                        </div>
                      </label>
                    ))}
                  </div>

                  {/* Model selection */}
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ color: t.textMuted, fontSize: 10, marginBottom: 4 }}>Model Selection</div>
                    {(["analyst", "portfolio_manager", "risk_manager", "strategist"] as const).map(role => (
                      <div key={role} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                        <span style={{ color: t.textDim, fontSize: 11 }}>{role.replace("_", " ")}</span>
                        <select
                          value={(config.models as any)[role]}
                          onChange={e => setConfig(c => ({ ...c, models: { ...c.models, [role]: e.target.value } }))}
                          style={{ ...inputStyle(t), width: 160, fontSize: 10, padding: "2px 4px" }}
                        >
                          {MODEL_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                        </select>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </Section>

            {/* Trading */}
            <Section label="Trading Rules" t={t}>
              <Row t={t}>
                <LabeledInput label="Min Score" value={config.min_composite_score} onChange={v => setConfig(c => ({ ...c, min_composite_score: v }))} t={t} />
                <LabeledInput label="Max Trades/Day" value={config.max_trades_per_day} onChange={v => setConfig(c => ({ ...c, max_trades_per_day: v }))} t={t} />
              </Row>
              <Row t={t}>
                <LabeledInput label="Max Trades/Scan" value={config.max_trades_per_scan} onChange={v => setConfig(c => ({ ...c, max_trades_per_scan: v }))} t={t} />
                <LabeledInput label="Speed" value={config.playback_speed} onChange={v => setConfig(c => ({ ...c, playback_speed: v }))} t={t} />
              </Row>
            </Section>

            {/* Strategy selection */}
            <Section label="Strategies" t={t}>
              <StrategySelector
                strategies={strategies}
                groups={groups}
                selected={config.allowed_strategies}
                onChange={s => setConfig(c => ({ ...c, allowed_strategies: s }))}
                t={t}
              />
            </Section>

            {/* Start button */}
            <button
              onClick={startRun}
              disabled={running}
              style={{
                width: "100%", padding: "10px 16px", marginTop: 8,
                background: running ? t.bgHover : t.accent,
                color: running ? t.textMuted : "#fff",
                border: "none", borderRadius: 6, fontSize: 14, fontWeight: 700,
                cursor: running ? "not-allowed" : "pointer",
              }}
            >
              {running ? "Running..." : "Start Run"}
            </button>

            {/* Run history */}
            <div style={{ marginTop: 16, borderTop: `1px solid ${t.border}`, paddingTop: 8 }}>
              <div style={{ color: t.textDim, fontSize: 11, marginBottom: 4 }}>
                Run History ({savedRuns.length})
                <button onClick={loadRuns} style={{ background: "none", border: "none", color: t.accent, cursor: "pointer", fontSize: 10, marginLeft: 8 }}>Refresh</button>
              </div>
              <RunHistory runs={savedRuns} onSelect={viewRun} onDelete={deleteRun} t={t} />
            </div>
          </div>
        )}

        {/* Main content */}
        <div>
          {/* Equity chart */}
          {pnlPoints.length > 2 && (
            <div style={{ marginBottom: 12 }}>
              <EquityChart points={pnlPoints} t={t} />
            </div>
          )}

          {/* Agent log */}
          {agentLogs.length > 0 && (
            <div style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 8, padding: 8, marginBottom: 12 }}>
              <div style={{ color: t.textDim, fontSize: 11, marginBottom: 4 }}>Agent Log ({agentLogs.length})</div>
              <AgentLogView logs={agentLogs} t={t} />
            </div>
          )}

          {/* Trade log */}
          {closedTrades.length > 0 && (
            <div style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 8, padding: 8, marginBottom: 12 }}>
              <div style={{ color: t.textDim, fontSize: 11, marginBottom: 4 }}>Trades ({closedTrades.filter(t2 => t2.outcome).length} closed)</div>
              <TradeLog trades={closedTrades.filter(t2 => t2.outcome)} t={t} />
            </div>
          )}

          {/* Empty state */}
          {!running && pnlPoints.length === 0 && (
            <div style={{ textAlign: "center", padding: 60, color: t.textMuted }}>
              <div style={{ fontSize: 32, marginBottom: 8 }}>Custom Agent Trading</div>
              <div style={{ fontSize: 14 }}>Configure your strategy, agents, and sizing on the left, then hit Start Run.</div>
              <div style={{ fontSize: 12, marginTop: 8, color: t.textDim }}>
                Runs persist in the background — navigate away and come back anytime.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Utility components ───────────────────────────────────────────────────────

function Section({ label, t, children }: { label: string; t: Theme; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ color: t.textDim, fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: 1, marginBottom: 4 }}>{label}</div>
      {children}
    </div>
  );
}

function Row({ t, children }: { t: Theme; children: React.ReactNode }) {
  return <div style={{ display: "flex", gap: 8, marginBottom: 4 }}>{children}</div>;
}

function LabeledInput({ label, value, onChange, t, prefix, step }: {
  label: string; value: number; onChange: (v: number) => void; t: Theme; prefix?: string; step?: number;
}) {
  return (
    <div style={{ flex: 1 }}>
      <div style={{ color: t.textMuted, fontSize: 10, marginBottom: 2 }}>{label}</div>
      <div style={{ display: "flex", alignItems: "center" }}>
        {prefix && <span style={{ color: t.textMuted, fontSize: 12, marginRight: 2 }}>{prefix}</span>}
        <input
          type="number" value={value} step={step || 1}
          onChange={e => onChange(parseFloat(e.target.value) || 0)}
          style={{ ...inputStyle(t), width: "100%" }}
        />
      </div>
    </div>
  );
}

function btnStyle(t: Theme): React.CSSProperties {
  return {
    background: "none", border: `1px solid ${t.border}`, color: t.text,
    padding: "4px 12px", borderRadius: 6, cursor: "pointer", fontSize: 12,
  };
}

function inputStyle(t: Theme): React.CSSProperties {
  return {
    background: t.bg, border: `1px solid ${t.border}`, color: t.text,
    padding: "4px 8px", borderRadius: 4, fontSize: 12, outline: "none",
    width: "100%",
  };
}
