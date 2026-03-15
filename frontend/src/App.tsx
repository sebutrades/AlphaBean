import { useState, useMemo } from "react";

// ============================================================
// TYPES
// ============================================================
interface Setup {
  pattern_name: string;
  symbol: string;
  bias: string;
  timeframe: string;
  entry_price: number;
  stop_loss: number;
  target_price: number;
  risk_reward_ratio: number;
  confidence: number;
  detected_at: string;
  description: string;
  win_rate: number;
  max_attempts: number;
  exit_strategy: string;
  key_levels: Record<string, number>;
  ideal_time: string;
}

interface TopOppsResponse {
  in_play_source: string;
  in_play_date: string;
  symbols_scanned: string[];
  count: number;
  setups: Setup[];
}

// ============================================================
// API
// ============================================================
const API = "http://localhost:8000";

async function fetchScan(symbol: string, timeframe: string, daysBack: number): Promise<Setup[]> {
  const res = await fetch(`${API}/api/scan?symbol=${symbol}&timeframe=${timeframe}&days_back=${daysBack}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return (await res.json()).setups;
}

async function fetchTopOpps(): Promise<TopOppsResponse> {
  const res = await fetch(`${API}/api/top-opps?days_back=30`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return await res.json();
}

// ============================================================
// HELPERS
// ============================================================
function confLabel(c: number): { text: string; color: string } {
  if (c >= 0.80) return { text: "Very High — Multiple confirming signals", color: "#15803d" };
  if (c >= 0.65) return { text: "High — Strong pattern with good volume", color: "#16a34a" };
  if (c >= 0.50) return { text: "Moderate — Watch for confirmation", color: "#ca8a04" };
  return { text: "Speculative — Weak signals, use caution", color: "#dc2626" };
}

const F = "'Nunito', sans-serif";
const FM = "'Courier New', monospace";

// ============================================================
// BEAN LOGO
// ============================================================
function BeanLogo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" style={{ verticalAlign: "middle" }}>
      <ellipse cx="50" cy="52" rx="36" ry="40" fill="#8B5E3C" />
      <ellipse cx="50" cy="50" rx="34" ry="38" fill="#C4956A" />
      <ellipse cx="50" cy="48" rx="30" ry="34" fill="#D4A574" />
      <path d="M 35 30 Q 50 70 65 30" stroke="#8B5E3C" strokeWidth="3" fill="none" strokeLinecap="round" />
      <ellipse cx="40" cy="42" rx="4" ry="5" fill="#5C3A1E" />
      <ellipse cx="60" cy="42" rx="4" ry="5" fill="#5C3A1E" />
      <ellipse cx="41" cy="41" rx="1.5" ry="1.8" fill="#fff" />
      <ellipse cx="61" cy="41" rx="1.5" ry="1.8" fill="#fff" />
      <path d="M 44 55 Q 50 61 56 55" stroke="#5C3A1E" strokeWidth="2.5" fill="none" strokeLinecap="round" />
      <ellipse cx="33" cy="52" rx="5" ry="4" fill="#E8B89D" opacity="0.6" />
      <ellipse cx="67" cy="52" rx="5" ry="4" fill="#E8B89D" opacity="0.6" />
    </svg>
  );
}

// ============================================================
// SETUP ROW
// ============================================================
function SetupRow({ s, expanded, onToggle }: { s: Setup; expanded: boolean; onToggle: () => void }) {
  const isLong = s.bias === "long";
  const biasColor = isLong ? "#16a34a" : "#dc2626";
  const biasBg = isLong ? "#f0fdf4" : "#fef2f2";
  const conf = confLabel(s.confidence);

  return (
    <div onClick={onToggle} style={{ borderBottom: "1px solid #f0ebe4", padding: "14px 0", cursor: "pointer", background: expanded ? "#faf7f3" : "transparent", transition: "background 0.15s" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <div style={{ width: 72 }}>
          <div style={{ fontFamily: FM, fontSize: 15, fontWeight: 700, color: "#3d2c1e" }}>{s.symbol}</div>
          <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 100, background: biasBg, color: biasColor }}>{s.bias.toUpperCase()}</span>
        </div>
        <div style={{ flex: 1, minWidth: 120 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#3d2c1e", fontFamily: F }}>{s.pattern_name}</div>
          <div style={{ fontSize: 11, color: "#a1917e" }}>{s.timeframe}</div>
        </div>
        <div style={{ display: "flex", gap: 14 }}>
          {[
            { label: "ENTRY", val: s.entry_price, color: "#3d2c1e" },
            { label: "STOP", val: s.stop_loss, color: "#dc2626" },
            { label: "TARGET", val: s.target_price, color: "#16a34a" },
          ].map(({ label, val, color }) => (
            <div key={label} style={{ textAlign: "right" as const }}>
              <div style={{ fontSize: 10, color: "#a1917e", fontWeight: 600, letterSpacing: 0.5, fontFamily: F }}>{label}</div>
              <div style={{ fontFamily: FM, fontSize: 13, fontWeight: 600, color }}>${val.toFixed(2)}</div>
            </div>
          ))}
        </div>
        <div style={{ width: 50, textAlign: "center" as const }}>
          <div style={{ fontSize: 10, color: "#a1917e", fontFamily: F, fontWeight: 600 }}>R:R</div>
          <div style={{ fontFamily: FM, fontSize: 15, fontWeight: 800, color: s.risk_reward_ratio >= 2 ? "#16a34a" : "#3d2c1e" }}>{s.risk_reward_ratio.toFixed(1)}</div>
        </div>
        <div style={{ width: 50, textAlign: "center" as const }}>
          <div style={{ fontSize: 10, color: "#a1917e", fontFamily: F, fontWeight: 600 }}>CONF</div>
          <div style={{ fontFamily: FM, fontSize: 15, fontWeight: 800, color: conf.color }}>{Math.round(s.confidence * 100)}%</div>
        </div>
        <div style={{ width: 20, textAlign: "center" as const, fontSize: 14, color: "#c4b5a4", transform: expanded ? "rotate(180deg)" : "rotate(0)", transition: "transform 0.2s" }}>▾</div>
      </div>
      {expanded && (
        <div style={{ marginTop: 14, paddingTop: 14, borderTop: "1px solid #f0ebe4", display: "flex", gap: 24, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ fontSize: 10, fontWeight: 800, color: "#a1917e", letterSpacing: 1, textTransform: "uppercase" as const, marginBottom: 6, fontFamily: F }}>Description</div>
            <p style={{ fontSize: 12, color: "#5c4a3a", lineHeight: 1.7, margin: 0, fontFamily: F }}>{s.description}</p>
            <div style={{ marginTop: 10, padding: "8px 12px", borderRadius: 8, background: "#faf7f3", border: `1px solid ${conf.color}22` }}>
              <div style={{ fontSize: 10, fontWeight: 800, color: "#a1917e", letterSpacing: 1, marginBottom: 2, fontFamily: F }}>CONFIDENCE ASSESSMENT</div>
              <div style={{ fontSize: 12, color: conf.color, fontWeight: 700, fontFamily: F }}>{Math.round(s.confidence * 100)}% — {conf.text}</div>
            </div>
          </div>
          <div style={{ width: 220 }}>
            <div style={{ fontSize: 10, fontWeight: 800, color: "#a1917e", letterSpacing: 1, textTransform: "uppercase" as const, marginBottom: 8, fontFamily: F }}>Details</div>
            {[
              ["Win Rate", `${Math.round(s.win_rate * 100)}%`],
              ["Max Attempts", String(s.max_attempts)],
              ["Ideal Time", s.ideal_time || "—"],
            ].map(([label, val]) => (
              <div key={label} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                <span style={{ color: "#a1917e", fontFamily: F }}>{label}</span>
                <span style={{ fontFamily: FM, fontWeight: 600, color: "#3d2c1e" }}>{val}</span>
              </div>
            ))}
            <div style={{ marginTop: 8, fontSize: 10, color: "#a1917e", fontFamily: F }}>
              <span style={{ fontWeight: 800 }}>EXIT: </span>
              <span style={{ color: "#5c4a3a" }}>{s.exit_strategy || "—"}</span>
            </div>
            {Object.keys(s.key_levels).length > 0 && (
              <div style={{ marginTop: 10 }}>
                <div style={{ fontSize: 10, fontWeight: 800, color: "#a1917e", letterSpacing: 1, marginBottom: 4, fontFamily: F }}>KEY LEVELS</div>
                {Object.entries(s.key_levels).map(([k, v]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 2 }}>
                    <span style={{ color: "#a1917e", fontFamily: F }}>{k}</span>
                    <span style={{ fontFamily: FM, fontWeight: 600, color: "#3d2c1e" }}>{typeof v === "number" ? `$${v.toFixed(2)}` : v}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================
// MAIN APP
// ============================================================
function App() {
  const [tab, setTab] = useState<"scan" | "topOpps">("scan");
  const [symbol, setSymbol] = useState("AAPL");
  const [timeframe, setTimeframe] = useState("1d");
  const [daysBack, setDaysBack] = useState(30);
  const [setups, setSetups] = useState<Setup[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const [filterBias, setFilterBias] = useState("ALL");
  const [filterPattern, setFilterPattern] = useState("ALL");
  const [hasScanned, setHasScanned] = useState(false);
  const [topOppsInfo, setTopOppsInfo] = useState<{ symbols: string[]; source: string } | null>(null);

  const handleScan = async () => {
    setLoading(true); setError(""); setSetups([]); setHasScanned(true);
    setExpandedIdx(null); setFilterBias("ALL"); setFilterPattern("ALL");
    try {
      setSetups(await fetchScan(symbol, timeframe, daysBack));
    } catch (e: unknown) { setError(e instanceof Error ? e.message : "Unknown error"); }
    finally { setLoading(false); }
  };

  const handleTopOpps = async () => {
    setTab("topOpps"); setLoading(true); setError(""); setSetups([]);
    setExpandedIdx(null); setFilterBias("ALL"); setFilterPattern("ALL");
    setHasScanned(true);
    try {
      const data = await fetchTopOpps();
      setSetups(data.setups);
      setTopOppsInfo({ symbols: data.symbols_scanned, source: data.in_play_source });
    } catch (e: unknown) { setError(e instanceof Error ? e.message : "Unknown error"); }
    finally { setLoading(false); }
  };

  const patternNames = useMemo(() => {
    const names = [...new Set(setups.map(s => s.pattern_name))].sort();
    return ["ALL", ...names];
  }, [setups]);

  const filtered = useMemo(() => {
    let result = setups;
    if (filterBias !== "ALL") result = result.filter(s => s.bias === filterBias.toLowerCase());
    if (filterPattern !== "ALL") result = result.filter(s => s.pattern_name === filterPattern);
    return result;
  }, [setups, filterBias, filterPattern]);

  const longs = setups.filter(s => s.bias === "long").length;
  const shorts = setups.filter(s => s.bias === "short").length;

  const pillStyle = (active: boolean) => ({
    fontSize: 12, fontWeight: active ? 800 : 600, padding: "6px 16px", borderRadius: 20,
    cursor: "pointer" as const, border: "none", fontFamily: F,
    background: active ? "#3d2c1e" : "transparent", color: active ? "#fff" : "#8c7a68",
    transition: "all 0.15s",
  });

  const mainTabStyle = (active: boolean) => ({
    fontSize: 14, fontWeight: active ? 800 : 600, padding: "10px 24px", borderRadius: 12,
    cursor: "pointer" as const, border: active ? "2px solid #3d2c1e" : "2px solid transparent",
    fontFamily: F, background: active ? "#faf7f3" : "transparent", color: active ? "#3d2c1e" : "#8c7a68",
    transition: "all 0.15s",
  });

  return (
    <div style={{ background: "#fff", minHeight: "100vh", fontFamily: F, color: "#3d2c1e" }}>
      {/* Header */}
      <div style={{ padding: "18px 32px", borderBottom: "2px solid #f0ebe4", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <BeanLogo size={32} />
          <span style={{ fontSize: 22, fontWeight: 900, letterSpacing: -0.5 }}>AlphaBean</span>
          <span style={{ fontSize: 11, color: "#c4b5a4", fontWeight: 600 }}>v2.0</span>
        </div>
        {setups.length > 0 && (
          <div style={{ display: "flex", gap: 14, fontSize: 12, fontWeight: 700 }}>
            <span style={{ color: "#16a34a" }}>{longs} Long</span>
            <span style={{ color: "#dc2626" }}>{shorts} Short</span>
            <span style={{ color: "#8c7a68" }}>{setups.length} Total</span>
          </div>
        )}
      </div>

      <div style={{ maxWidth: 940, margin: "0 auto", padding: "24px 32px" }}>
        {/* Main Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
          <button style={mainTabStyle(tab === "scan")} onClick={() => { setTab("scan"); setSetups([]); setHasScanned(false); }}>Manual Scan</button>
          <button style={mainTabStyle(tab === "topOpps")} onClick={handleTopOpps}>Top Opportunities</button>
        </div>

        {/* Scan Controls (only in scan tab) */}
        {tab === "scan" && (
          <div style={{ display: "flex", gap: 10, alignItems: "end", marginBottom: 24, flexWrap: "wrap" }}>
            <div>
              <label style={{ fontSize: 11, color: "#8c7a68", display: "block", marginBottom: 4, fontWeight: 700, fontFamily: F }}>SYMBOL</label>
              <input type="text" value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())}
                onKeyDown={e => e.key === "Enter" && handleScan()} placeholder="AAPL"
                style={{ fontSize: 14, padding: "8px 14px", borderRadius: 10, border: "2px solid #e8dfd4", width: 100, fontFamily: FM, fontWeight: 700, outline: "none" }} />
            </div>
            <div>
              <label style={{ fontSize: 11, color: "#8c7a68", display: "block", marginBottom: 4, fontWeight: 700, fontFamily: F }}>TIMEFRAME</label>
              <select value={timeframe} onChange={e => setTimeframe(e.target.value)}
                style={{ fontSize: 14, padding: "8px 14px", borderRadius: 10, border: "2px solid #e8dfd4", background: "#fff", fontFamily: F, fontWeight: 600, outline: "none" }}>
                <option value="15min">15 min</option>
                <option value="1h">1 hour</option>
                <option value="1d">1 day</option>
              </select>
            </div>
            <div>
              <label style={{ fontSize: 11, color: "#8c7a68", display: "block", marginBottom: 4, fontWeight: 700, fontFamily: F }}>DAYS BACK</label>
              <input type="number" value={daysBack} onChange={e => setDaysBack(Number(e.target.value))}
                style={{ fontSize: 14, padding: "8px 14px", borderRadius: 10, border: "2px solid #e8dfd4", width: 70, fontFamily: FM, outline: "none" }} />
            </div>
            <button onClick={handleScan} disabled={loading || !symbol}
              style={{ fontSize: 14, fontWeight: 800, padding: "9px 28px", borderRadius: 10, border: "none", fontFamily: F,
                background: loading ? "#c4b5a4" : "#3d2c1e", color: "#fff", cursor: loading ? "wait" : "pointer", transition: "all 0.15s" }}>
              {loading ? "Scanning..." : "Scan"}
            </button>
          </div>
        )}

        {/* Top Opps info banner */}
        {tab === "topOpps" && topOppsInfo && !loading && (
          <div style={{ padding: "12px 16px", borderRadius: 10, background: "#faf7f3", border: "1px solid #e8dfd4", marginBottom: 16, fontSize: 12, color: "#8c7a68", fontFamily: F }}>
            <span style={{ fontWeight: 800 }}>Symbols scanned: </span>
            {topOppsInfo.symbols.join(", ")}
            <span style={{ marginLeft: 12, fontWeight: 600 }}>(Source: {topOppsInfo.source})</span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{ padding: "12px 16px", borderRadius: 10, background: "#fef2f2", color: "#dc2626", fontSize: 13, marginBottom: 16, fontFamily: F }}>
            <strong>Error:</strong> {error}
            <div style={{ marginTop: 4, fontSize: 11, color: "#8c7a68" }}>
              Make sure the backend is running: <code style={{ background: "#fff", padding: "1px 6px", borderRadius: 4, fontFamily: FM, fontSize: 11 }}>uvicorn backend.main:app --reload --port 8000</code>
            </div>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div style={{ textAlign: "center", padding: 48, color: "#c4b5a4" }}>
            <BeanLogo size={48} />
            <div style={{ fontSize: 15, marginTop: 12, fontWeight: 700, fontFamily: F }}>
              {tab === "topOpps" ? "Scanning all in-play symbols across 3 timeframes..." : "Fetching data and running 21 pattern detectors..."}
            </div>
            <div style={{ fontSize: 12, marginTop: 4 }}>
              {tab === "topOpps" ? "This scans ~30 symbol×timeframe combos. May take 2-5 minutes." : "This can take 10-20 seconds"}
            </div>
          </div>
        )}

        {/* Results */}
        {!loading && setups.length > 0 && (
          <>
            {/* Filters */}
            <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 16, flexWrap: "wrap" }}>
              <div style={{ display: "flex", gap: 4, background: "#f5f0ea", borderRadius: 24, padding: 3 }}>
                {["ALL", "LONG", "SHORT"].map(b => (
                  <button key={b} style={pillStyle(filterBias === b)} onClick={() => setFilterBias(b)}>{b}</button>
                ))}
              </div>
              {patternNames.length > 2 && (
                <select value={filterPattern} onChange={e => setFilterPattern(e.target.value)}
                  style={{ fontSize: 12, padding: "6px 12px", borderRadius: 20, border: "2px solid #e8dfd4", background: "#fff", fontFamily: F, fontWeight: 600, outline: "none", color: "#3d2c1e" }}>
                  {patternNames.map(p => <option key={p} value={p}>{p === "ALL" ? "All Patterns" : p}</option>)}
                </select>
              )}
              <span style={{ fontSize: 11, color: "#c4b5a4", fontWeight: 600 }}>
                Showing {filtered.length} of {setups.length}
              </span>
            </div>

            {/* Column headers */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderBottom: "2px solid #3d2c1e", fontSize: 10, fontWeight: 800, color: "#a1917e", letterSpacing: 0.8, textTransform: "uppercase" as const, fontFamily: F }}>
              <div style={{ width: 72 }}>Symbol</div>
              <div style={{ flex: 1 }}>Pattern</div>
              <div style={{ display: "flex", gap: 14 }}>
                <div style={{ width: 60, textAlign: "right" as const }}>Entry</div>
                <div style={{ width: 60, textAlign: "right" as const }}>Stop</div>
                <div style={{ width: 60, textAlign: "right" as const }}>Target</div>
              </div>
              <div style={{ width: 50, textAlign: "center" as const }}>R:R</div>
              <div style={{ width: 50, textAlign: "center" as const }}>Conf</div>
              <div style={{ width: 20 }} />
            </div>

            {filtered.map((s, i) => (
              <SetupRow key={`${s.symbol}-${s.pattern_name}-${i}`} s={s} expanded={expandedIdx === i} onToggle={() => setExpandedIdx(expandedIdx === i ? null : i)} />
            ))}
          </>
        )}

        {/* Empty */}
        {!loading && hasScanned && setups.length === 0 && !error && (
          <div style={{ textAlign: "center", padding: 48, color: "#c4b5a4" }}>
            <BeanLogo size={40} />
            <div style={{ fontSize: 16, marginTop: 12, fontWeight: 700, fontFamily: F }}>No patterns detected</div>
            <div style={{ fontSize: 12, marginTop: 4, maxWidth: 400, margin: "8px auto 0" }}>
              {tab === "topOpps"
                ? "None of the in-play symbols matched any of the 21 pattern detectors on any timeframe right now. Check back later."
                : `No patterns matched for ${symbol} on ${timeframe}. Try a different symbol, timeframe, or look-back period.`}
            </div>
          </div>
        )}

        {/* Initial */}
        {!loading && !hasScanned && tab === "scan" && (
          <div style={{ textAlign: "center", padding: 48, color: "#c4b5a4" }}>
            <BeanLogo size={40} />
            <div style={{ fontSize: 16, marginTop: 12, fontWeight: 700, fontFamily: F }}>Enter a ticker and hit Scan</div>
            <div style={{ fontSize: 12, marginTop: 4 }}>
              AlphaBean will fetch price data and run 21 pattern detectors (10 SMB scalps + 11 classical) to find trade setups.
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={{ textAlign: "center", marginTop: 40, padding: "16px 0", borderTop: "1px solid #f0ebe4" }}>
          <BeanLogo size={16} />
          <span style={{ fontSize: 11, color: "#c4b5a4", marginLeft: 6, fontWeight: 600 }}>AlphaBean v2.0 · 21 Detectors · 3 Timeframes</span>
        </div>
      </div>
    </div>
  );
}

export default App;