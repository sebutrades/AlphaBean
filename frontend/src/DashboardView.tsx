import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";

const API = "http://localhost:8000";
const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" } },
};

// ── Types ────────────────────────────────────────────────────────────────────

interface EquityPoint { date: string; cumulative_r: number; daily_r: number; trade_count: number }
interface PatternAttr { pattern_name: string; total_r: number; trade_count: number; win_rate: number; avg_r: number; best_r: number; worst_r: number }
interface DailyPnl { date: string; r_total: number; wins: number; losses: number; trade_count: number }
interface Streaks { current_type: string; current_length: number; best_win_streak: number; worst_loss_streak: number }
interface DrawdownData { max_drawdown_r: number; current_drawdown_r: number; peak_r: number; series: { date: string; drawdown_r: number }[] }
interface SizingConfig { account_size: number; risk_per_trade_pct: number; max_portfolio_heat_pct: number; max_single_position_pct: number }
interface PortfolioHeat { total_risk_dollars: number; total_risk_pct: number; positions_count: number; long_exposure: number; short_exposure: number; net_exposure: number; can_add_trade: boolean; remaining_risk_budget: number }

interface DashboardData {
  equity_curve: EquityPoint[];
  pattern_attribution: PatternAttr[];
  daily_pnl: DailyPnl[];
  drawdown: DrawdownData;
  streaks: Streaks;
  summary: {
    total_trades: number; total_r: number; win_rate: number; profit_factor: number;
    avg_r: number; best_day_r: number; worst_day_r: number; trading_days: number;
  };
}

// ── Mini Chart (pure CSS, no charting lib needed) ────────────────────────────

function MiniBarChart({ data, color, negColor, height = 60, t }: { data: number[]; color: string; negColor: string; height?: number; t: any }) {
  if (!data.length) return null;
  const maxAbs = Math.max(...data.map(Math.abs), 0.01);
  const w = Math.max(4, Math.min(16, 300 / data.length));
  return (
    <div style={{ display: "flex", alignItems: "center", height, gap: 1 }}>
      {data.map((v, i) => {
        const h = Math.abs(v) / maxAbs * (height / 2);
        const isPos = v >= 0;
        return (
          <div key={i} style={{ display: "flex", flexDirection: "column", justifyContent: "center", height }}>
            <div style={{
              width: w, height: h, borderRadius: 2,
              background: isPos ? color : negColor,
              opacity: 0.85,
              marginTop: isPos ? height / 2 - h : height / 2,
            }} title={`${v >= 0 ? "+" : ""}${v.toFixed(2)}R`} />
          </div>
        );
      })}
    </div>
  );
}

function EquityCurve({ data, t }: { data: EquityPoint[]; t: any }) {
  if (data.length < 2) return <div style={{ color: t.textMuted, textAlign: "center", padding: 30 }}>Not enough data for equity curve</div>;
  const rs = data.map(d => d.cumulative_r);
  const maxR = Math.max(...rs);
  const minR = Math.min(...rs, 0);
  const range = maxR - minR || 1;
  const h = 140;
  const points = data.map((d, i) => ({
    x: (i / (data.length - 1)) * 100,
    y: h - ((d.cumulative_r - minR) / range) * h,
  }));
  const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
  const fillD = pathD + ` L ${points[points.length - 1].x.toFixed(1)} ${h} L 0 ${h} Z`;
  const finalR = rs[rs.length - 1];
  const lineColor = finalR >= 0 ? t.long : t.short;

  return (
    <div style={{ position: "relative" }}>
      <svg viewBox={`0 0 100 ${h}`} preserveAspectRatio="none" style={{ width: "100%", height: h }}>
        <defs>
          <linearGradient id="eqFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={lineColor} stopOpacity="0.25" />
            <stop offset="100%" stopColor={lineColor} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        <path d={fillD} fill="url(#eqFill)" />
        <path d={pathD} fill="none" stroke={lineColor} strokeWidth="0.6" />
      </svg>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: t.textMuted, marginTop: 4 }}>
        <span>{data[0].date}</span>
        <span>{data[data.length - 1].date}</span>
      </div>
    </div>
  );
}

// ── Stat Card ────────────────────────────────────────────────────────────────

function StatCard({ label, value, color, sub, t }: { label: string; value: string; color: string; sub?: string; t: any }) {
  return (
    <div style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 12, padding: "12px 14px", textAlign: "center" }}>
      <div style={{ fontSize: 10, color: t.textMuted, fontWeight: 700, marginBottom: 4, letterSpacing: 0.4 }}>{label.toUpperCase()}</div>
      <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 18, fontWeight: 800, color }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: t.textMuted, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

// ── Position Sizing Widget ───────────────────────────────────────────────────

function SizingWidget({ t }: { t: any }) {
  const [config, setConfig] = useState<SizingConfig | null>(null);
  const [heat, setHeat] = useState<PortfolioHeat | null>(null);
  const [calcEntry, setCalcEntry] = useState("");
  const [calcStop, setCalcStop] = useState("");
  const [calcBias, setCalcBias] = useState("long");
  const [calcResult, setCalcResult] = useState<any>(null);
  const [editing, setEditing] = useState(false);
  const [editConfig, setEditConfig] = useState<SizingConfig>({ account_size: 25000, risk_per_trade_pct: 1, max_portfolio_heat_pct: 6, max_single_position_pct: 15 });

  useEffect(() => {
    fetch(`${API}/api/sizing/config`).then(r => r.json()).then(d => { setConfig(d); setEditConfig(d); }).catch(() => {});
    fetch(`${API}/api/sizing/summary`).then(r => r.json()).then(d => { if (d.heat) setHeat(d.heat); }).catch(() => {});
  }, []);

  const saveConfig = async () => {
    await fetch(`${API}/api/sizing/config`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(editConfig) });
    setConfig(editConfig);
    setEditing(false);
  };

  const calculate = async () => {
    if (!calcEntry || !calcStop) return;
    try {
      const r = await fetch(`${API}/api/sizing/calculate?entry=${calcEntry}&stop=${calcStop}&bias=${calcBias}`);
      setCalcResult(await r.json());
    } catch {}
  };

  const inputStyle = { fontSize: 13, padding: "6px 10px", borderRadius: 8, border: `1px solid ${t.border}`, background: t.bgCard, color: t.text, fontFamily: "'JetBrains Mono',monospace", width: 100 };

  return (
    <motion.div variants={fadeUp} style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 14, padding: 16, marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <span style={{ fontSize: 14, fontWeight: 800, color: t.text }}>Position Sizing</span>
        <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => setEditing(!editing)}
          style={{ fontSize: 11, fontWeight: 700, padding: "4px 12px", borderRadius: 8, border: `1px solid ${t.border}`, background: editing ? t.accent + "15" : t.bgCard, color: editing ? t.accent : t.textDim, cursor: "pointer" }}>
          {editing ? "Cancel" : "Settings"}
        </motion.button>
      </div>

      {editing && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12, padding: 12, background: t.bg, borderRadius: 10 }}>
          {([["Account Size", "account_size"], ["Risk Per Trade %", "risk_per_trade_pct"], ["Max Heat %", "max_portfolio_heat_pct"], ["Max Position %", "max_single_position_pct"]] as const).map(([label, key]) => (
            <div key={key}>
              <div style={{ fontSize: 10, color: t.textMuted, fontWeight: 700, marginBottom: 2 }}>{label}</div>
              <input type="number" value={(editConfig as any)[key]} onChange={e => setEditConfig({ ...editConfig, [key]: parseFloat(e.target.value) || 0 })} style={inputStyle} />
            </div>
          ))}
          <div style={{ gridColumn: "1 / -1" }}>
            <motion.button whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }} onClick={saveConfig}
              style={{ fontSize: 12, fontWeight: 700, padding: "6px 16px", borderRadius: 8, border: "none", background: `linear-gradient(135deg, ${t.accent}, ${t.accentLight})`, color: "#fff", cursor: "pointer" }}>
              Save
            </motion.button>
          </div>
        </div>
      )}

      {/* Portfolio Heat */}
      {heat && config && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 6, marginBottom: 12 }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700 }}>ACCOUNT</div>
            <div style={{ fontSize: 14, fontWeight: 800, color: t.text, fontFamily: "'JetBrains Mono',monospace" }}>${config.account_size.toLocaleString()}</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700 }}>HEAT</div>
            <div style={{ fontSize: 14, fontWeight: 800, color: heat.total_risk_pct > config.max_portfolio_heat_pct ? t.short : t.long, fontFamily: "'JetBrains Mono',monospace" }}>
              {heat.total_risk_pct.toFixed(1)}%
            </div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700 }}>POSITIONS</div>
            <div style={{ fontSize: 14, fontWeight: 800, color: t.text, fontFamily: "'JetBrains Mono',monospace" }}>{heat.positions_count}</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700 }}>BUDGET LEFT</div>
            <div style={{ fontSize: 14, fontWeight: 800, color: heat.can_add_trade ? t.long : t.short, fontFamily: "'JetBrains Mono',monospace" }}>
              ${heat.remaining_risk_budget.toFixed(0)}
            </div>
          </div>
        </div>
      )}

      {/* Calculator */}
      <div style={{ display: "flex", gap: 8, alignItems: "end", flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700, marginBottom: 2 }}>ENTRY</div>
          <input type="number" step="0.01" value={calcEntry} onChange={e => setCalcEntry(e.target.value)} placeholder="50.00" style={inputStyle} />
        </div>
        <div>
          <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700, marginBottom: 2 }}>STOP</div>
          <input type="number" step="0.01" value={calcStop} onChange={e => setCalcStop(e.target.value)} placeholder="48.50" style={inputStyle} />
        </div>
        <div style={{ display: "flex", gap: 2, background: t.border, borderRadius: 8, padding: 3 }}>
          {["long", "short"].map(b => (
            <button key={b} onClick={() => setCalcBias(b)} style={{
              fontSize: 11, fontWeight: 700, padding: "4px 10px", borderRadius: 6, border: "none", cursor: "pointer",
              background: calcBias === b ? t.accent : "transparent", color: calcBias === b ? "#fff" : t.textDim,
            }}>{b.toUpperCase()}</button>
          ))}
        </div>
        <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }} onClick={calculate}
          style={{ fontSize: 12, fontWeight: 700, padding: "7px 16px", borderRadius: 8, border: "none", background: `linear-gradient(135deg, ${t.accent}, ${t.accentLight})`, color: "#fff", cursor: "pointer" }}>
          Calculate
        </motion.button>
      </div>

      {calcResult && !calcResult.error && (
        <div style={{ marginTop: 10, padding: 10, background: t.bg, borderRadius: 10, display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 6 }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700 }}>SHARES</div>
            <div style={{ fontSize: 16, fontWeight: 800, color: t.accent, fontFamily: "'JetBrains Mono',monospace" }}>{calcResult.shares}</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700 }}>$ RISK</div>
            <div style={{ fontSize: 14, fontWeight: 800, color: t.short, fontFamily: "'JetBrains Mono',monospace" }}>${calcResult.dollar_risk?.toFixed(0)}</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700 }}>POSITION</div>
            <div style={{ fontSize: 14, fontWeight: 800, color: t.text, fontFamily: "'JetBrains Mono',monospace" }}>${calcResult.position_value?.toFixed(0)}</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 9, color: t.textMuted, fontWeight: 700 }}>RISK %</div>
            <div style={{ fontSize: 14, fontWeight: 800, color: t.text, fontFamily: "'JetBrains Mono',monospace" }}>{calcResult.risk_pct_of_account?.toFixed(2)}%</div>
          </div>
          {calcResult.capped && <div style={{ gridColumn: "1 / -1", fontSize: 11, color: t.gold, textAlign: "center" }}>Position capped at max single position limit</div>}
          {calcResult.warnings?.map((w: string, i: number) => <div key={i} style={{ gridColumn: "1 / -1", fontSize: 11, color: t.gold, textAlign: "center" }}>{w}</div>)}
        </div>
      )}
    </motion.div>
  );
}

// ── Alert Config Widget ──────────────────────────────────────────────────────

function AlertWidget({ t }: { t: any }) {
  const [config, setConfig] = useState<any>(null);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<any>(null);

  useEffect(() => {
    fetch(`${API}/api/alerts/config`).then(r => r.json()).then(setConfig).catch(() => {});
  }, []);

  const saveConfig = async (updated: any) => {
    await fetch(`${API}/api/alerts/config`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(updated) });
    setConfig(updated);
  };

  const testAlerts = async () => {
    setTesting(true);
    try {
      const r = await fetch(`${API}/api/alerts/test`, { method: "POST" });
      setTestResult(await r.json());
    } catch {}
    setTesting(false);
  };

  if (!config) return null;

  const inputStyle = { fontSize: 12, padding: "6px 10px", borderRadius: 8, border: `1px solid ${t.border}`, background: t.bg, color: t.text, fontFamily: "'JetBrains Mono',monospace", width: "100%" };

  return (
    <motion.div variants={fadeUp} style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 14, padding: 16, marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <span style={{ fontSize: 14, fontWeight: 800, color: t.text }}>Alerts</span>
        <div style={{ display: "flex", gap: 6 }}>
          <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={testAlerts} disabled={testing}
            style={{ fontSize: 11, fontWeight: 700, padding: "4px 12px", borderRadius: 8, border: `1px solid ${t.border}`, background: t.bgCard, color: t.textDim, cursor: testing ? "wait" : "pointer" }}>
            {testing ? "Testing..." : "Test"}
          </motion.button>
          <button onClick={() => saveConfig({ ...config, enabled: !config.enabled })}
            style={{ fontSize: 11, fontWeight: 700, padding: "4px 12px", borderRadius: 8, border: "none", background: config.enabled ? t.long + "20" : t.border, color: config.enabled ? t.long : t.textMuted, cursor: "pointer" }}>
            {config.enabled ? "ON" : "OFF"}
          </button>
        </div>
      </div>

      <div style={{ display: "grid", gap: 8 }}>
        {/* Discord */}
        <div style={{ padding: 10, background: t.bg, borderRadius: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
            <span style={{ fontSize: 12, fontWeight: 700, color: t.text }}>Discord</span>
            <button onClick={() => saveConfig({ ...config, channels: { ...config.channels, discord: { ...config.channels.discord, enabled: !config.channels.discord.enabled } } })}
              style={{ fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 5, border: "none", background: config.channels.discord.enabled ? t.long + "20" : t.border, color: config.channels.discord.enabled ? t.long : t.textMuted, cursor: "pointer" }}>
              {config.channels.discord.enabled ? "ON" : "OFF"}
            </button>
          </div>
          <input type="text" placeholder="Webhook URL" value={config.channels.discord.webhook_url || ""}
            onChange={e => saveConfig({ ...config, channels: { ...config.channels, discord: { ...config.channels.discord, webhook_url: e.target.value } } })}
            style={inputStyle} />
        </div>

        {/* Telegram */}
        <div style={{ padding: 10, background: t.bg, borderRadius: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
            <span style={{ fontSize: 12, fontWeight: 700, color: t.text }}>Telegram</span>
            <button onClick={() => saveConfig({ ...config, channels: { ...config.channels, telegram: { ...config.channels.telegram, enabled: !config.channels.telegram.enabled } } })}
              style={{ fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 5, border: "none", background: config.channels.telegram.enabled ? t.long + "20" : t.border, color: config.channels.telegram.enabled ? t.long : t.textMuted, cursor: "pointer" }}>
              {config.channels.telegram.enabled ? "ON" : "OFF"}
            </button>
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <input type="text" placeholder="Bot Token" value={config.channels.telegram.bot_token || ""}
              onChange={e => saveConfig({ ...config, channels: { ...config.channels, telegram: { ...config.channels.telegram, bot_token: e.target.value } } })}
              style={{ ...inputStyle, flex: 1 }} />
            <input type="text" placeholder="Chat ID" value={config.channels.telegram.chat_id || ""}
              onChange={e => saveConfig({ ...config, channels: { ...config.channels, telegram: { ...config.channels.telegram, chat_id: e.target.value } } })}
              style={{ ...inputStyle, width: 120 }} />
          </div>
        </div>

        {/* Min Score */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 11, color: t.textMuted, fontWeight: 700 }}>Min Score:</span>
          <input type="number" value={config.min_score} onChange={e => saveConfig({ ...config, min_score: parseInt(e.target.value) || 70 })}
            style={{ ...inputStyle, width: 60 }} />
        </div>
      </div>

      {testResult && (
        <div style={{ marginTop: 8, fontSize: 11, color: testResult.success ? t.long : t.short }}>
          {testResult.success ? "Test sent successfully" : `Error: ${testResult.error || "Failed"}`}
        </div>
      )}
    </motion.div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN DASHBOARD VIEW
// ═══════════════════════════════════════════════════════════════════════════════

export default function DashboardView({ t }: { t: any }) {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"overview" | "patterns" | "sizing" | "alerts">("overview");

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [summaryRes, dailyRes] = await Promise.all([
        fetch(`${API}/api/performance/summary`),
        fetch(`${API}/api/performance/daily`),
      ]);
      const raw = await summaryRes.json();
      const dailyPnl: DailyPnl[] = await dailyRes.json().catch(() => []);

      if (raw.error) { setLoading(false); return; }

      // Compute fields the frontend needs from daily_pnl
      const tradingDays = dailyPnl.filter((d: DailyPnl) => d.trade_count > 0).length;
      const dayRs = dailyPnl.map((d: DailyPnl) => d.r_total);
      const bestDayR = dayRs.length ? Math.max(...dayRs) : 0;
      const worstDayR = dayRs.length ? Math.min(...dayRs) : 0;

      // Compute peak R from equity curve
      const eqCurve: EquityPoint[] = raw.equity_curve || [];
      const peakR = eqCurve.length ? Math.max(...eqCurve.map((p: EquityPoint) => p.cumulative_r)) : 0;

      // Map streaks: backend uses current_streak / "win"/"loss"/"none",
      // frontend expects current_length / "W"/"L"
      const rawStreaks = raw.streaks || {};
      const typeMap: Record<string, string> = { win: "W", loss: "L", none: "-" };

      const shaped: DashboardData = {
        equity_curve: eqCurve,
        pattern_attribution: raw.pattern_attribution || [],
        daily_pnl: dailyPnl,
        drawdown: {
          max_drawdown_r: raw.drawdown?.max_drawdown_r ?? 0,
          current_drawdown_r: raw.drawdown?.current_drawdown_r ?? 0,
          peak_r: peakR,
          series: raw.drawdown?.series ?? [],
        },
        streaks: {
          current_type: typeMap[rawStreaks.current_type] || "-",
          current_length: Math.abs(rawStreaks.current_streak ?? 0),
          best_win_streak: rawStreaks.best_win_streak ?? 0,
          worst_loss_streak: rawStreaks.worst_loss_streak ?? 0,
        },
        summary: {
          total_trades: raw.total_trades ?? 0,
          total_r: raw.total_r ?? 0,
          win_rate: raw.win_rate ?? 0,
          profit_factor: raw.profit_factor ?? 0,
          avg_r: raw.expectancy ?? 0,
          best_day_r: bestDayR,
          worst_day_r: worstDayR,
          trading_days: tradingDays,
        },
      };

      setData(shaped);
    } catch (err) {
      console.error("Dashboard fetch error:", err);
    }
    setLoading(false);
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  if (loading) return (
    <div style={{ textAlign: "center", padding: 60, color: t.textDim }}>
      <div style={{ fontSize: 24, marginBottom: 10 }}>Loading dashboard...</div>
    </div>
  );

  return (
    <motion.div initial="hidden" animate="visible" variants={fadeUp}>
      {/* Tab bar */}
      <div style={{ display: "flex", gap: 2, background: t.border, borderRadius: 8, padding: 3, marginBottom: 14, width: "fit-content" }}>
        {(["overview", "patterns", "sizing", "alerts"] as const).map(v => (
          <button key={v} onClick={() => setTab(v)} style={{
            fontSize: 12, fontWeight: 700, padding: "5px 14px", borderRadius: 6, border: "none", cursor: "pointer",
            background: tab === v ? t.accent : "transparent", color: tab === v ? "#fff" : t.textDim,
            transition: "all .15s",
          }}>{v.charAt(0).toUpperCase() + v.slice(1)}</button>
        ))}
      </div>

      {tab === "overview" && data && (
        <>
          {/* Summary stats */}
          <motion.div variants={fadeUp} style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 8, marginBottom: 14 }}>
            <StatCard label="Total R" value={`${data.summary.total_r >= 0 ? "+" : ""}${data.summary.total_r.toFixed(2)}R`} color={data.summary.total_r >= 0 ? t.long : t.short} t={t} />
            <StatCard label="Win Rate" value={`${data.summary.win_rate.toFixed(0)}%`} color={data.summary.win_rate >= 55 ? t.long : data.summary.win_rate >= 45 ? t.gold : t.short} t={t} />
            <StatCard label="Profit Factor" value={data.summary.profit_factor.toFixed(2)} color={data.summary.profit_factor >= 1.5 ? t.long : data.summary.profit_factor >= 1 ? t.gold : t.short} t={t} />
            <StatCard label="Avg R" value={`${data.summary.avg_r >= 0 ? "+" : ""}${data.summary.avg_r.toFixed(3)}R`} color={data.summary.avg_r >= 0 ? t.long : t.short} t={t} />
            <StatCard label="Total Trades" value={String(data.summary.total_trades)} color={t.text} sub={`${data.summary.trading_days} days`} t={t} />
            <StatCard label="Max DD" value={`${data.drawdown.max_drawdown_r.toFixed(2)}R`} color={t.short} sub={`Now: ${data.drawdown.current_drawdown_r.toFixed(2)}R`} t={t} />
          </motion.div>

          {/* Equity Curve */}
          <motion.div variants={fadeUp} style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 14, padding: 16, marginBottom: 12 }}>
            <div style={{ fontSize: 14, fontWeight: 800, color: t.text, marginBottom: 10 }}>Equity Curve (Cumulative R)</div>
            <EquityCurve data={data.equity_curve} t={t} />
          </motion.div>

          {/* Daily P&L Bar Chart */}
          <motion.div variants={fadeUp} style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 14, padding: 16, marginBottom: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <span style={{ fontSize: 14, fontWeight: 800, color: t.text }}>Daily P&L</span>
              <span style={{ fontSize: 11, color: t.textMuted }}>
                Best: <span style={{ color: t.long }}>+{data.summary.best_day_r.toFixed(2)}R</span>
                {" / "}Worst: <span style={{ color: t.short }}>{data.summary.worst_day_r.toFixed(2)}R</span>
              </span>
            </div>
            <MiniBarChart data={data.daily_pnl.map(d => d.r_total)} color={t.long} negColor={t.short} height={70} t={t} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: t.textMuted, marginTop: 4 }}>
              <span>{data.daily_pnl[0]?.date}</span>
              <span>{data.daily_pnl[data.daily_pnl.length - 1]?.date}</span>
            </div>
          </motion.div>

          {/* Streaks */}
          {data.streaks && (
            <motion.div variants={fadeUp} style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginBottom: 12 }}>
              <StatCard label="Current Streak" value={`${data.streaks.current_length} ${data.streaks.current_type}`}
                color={data.streaks.current_type === "W" ? t.long : data.streaks.current_type === "L" ? t.short : t.textDim} t={t} />
              <StatCard label="Best Win Streak" value={String(data.streaks.best_win_streak)} color={t.long} t={t} />
              <StatCard label="Worst Loss Streak" value={String(data.streaks.worst_loss_streak)} color={t.short} t={t} />
              <StatCard label="Peak R" value={`+${data.drawdown.peak_r.toFixed(2)}R`} color={t.long} t={t} />
            </motion.div>
          )}
        </>
      )}

      {tab === "patterns" && data && (
        <motion.div variants={fadeUp} style={{ background: t.bgCard, border: `1px solid ${t.border}`, borderRadius: 14, overflow: "hidden" }}>
          <div style={{ padding: "10px 16px", borderBottom: `1px solid ${t.border}`, fontSize: 14, fontWeight: 800, color: t.text }}>Pattern Attribution</div>
          <div style={{ padding: "6px 16px", borderBottom: `2px solid ${t.borderLight}`, display: "grid", gridTemplateColumns: "2fr repeat(6, 1fr)", gap: 8, fontSize: 10, fontWeight: 700, color: t.textMuted }}>
            <div>PATTERN</div><div style={{ textAlign: "right" }}>TOTAL R</div><div style={{ textAlign: "right" }}>TRADES</div>
            <div style={{ textAlign: "right" }}>WIN %</div><div style={{ textAlign: "right" }}>AVG R</div>
            <div style={{ textAlign: "right" }}>BEST</div><div style={{ textAlign: "right" }}>WORST</div>
          </div>
          {data.pattern_attribution
            .sort((a, b) => b.total_r - a.total_r)
            .map((p, i) => (
            <div key={p.pattern_name} style={{
              padding: "8px 16px", borderBottom: `1px solid ${t.border}`,
              display: "grid", gridTemplateColumns: "2fr repeat(6, 1fr)", gap: 8, fontSize: 12,
              background: i % 2 === 0 ? "transparent" : t.bg + "40",
            }}>
              <div style={{ fontWeight: 700, color: t.text }}>{p.pattern_name}</div>
              <div style={{ textAlign: "right", fontFamily: "'JetBrains Mono',monospace", fontWeight: 700, color: p.total_r >= 0 ? t.long : t.short }}>
                {p.total_r >= 0 ? "+" : ""}{p.total_r.toFixed(2)}R
              </div>
              <div style={{ textAlign: "right", color: t.textDim }}>{p.trade_count}</div>
              <div style={{ textAlign: "right", fontWeight: 700, color: p.win_rate >= 55 ? t.long : p.win_rate >= 45 ? t.gold : t.short }}>
                {p.win_rate.toFixed(0)}%
              </div>
              <div style={{ textAlign: "right", fontFamily: "'JetBrains Mono',monospace", color: p.avg_r >= 0 ? t.long : t.short }}>
                {p.avg_r >= 0 ? "+" : ""}{p.avg_r.toFixed(2)}
              </div>
              <div style={{ textAlign: "right", fontFamily: "'JetBrains Mono',monospace", color: t.long }}>+{p.best_r.toFixed(2)}</div>
              <div style={{ textAlign: "right", fontFamily: "'JetBrains Mono',monospace", color: t.short }}>{p.worst_r.toFixed(2)}</div>
            </div>
          ))}
          {data.pattern_attribution.length === 0 && (
            <div style={{ padding: 30, textAlign: "center", color: t.textMuted }}>No closed trades yet — patterns will appear here as you close positions.</div>
          )}
        </motion.div>
      )}

      {tab === "sizing" && <SizingWidget t={t} />}
      {tab === "alerts" && <AlertWidget t={t} />}
    </motion.div>
  );
}
