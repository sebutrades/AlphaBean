import { useState, useMemo, useEffect } from "react";

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
  backtest_score?: number;
}

interface BacktestStatus {
  has_results: boolean;
  generated: string | null;
  quarter: string | null;
  total_signals: number;
  patterns_tested: number;
}

const API = "http://localhost:8000";

// ── Bean Logo ──────────────────────────────────────────────
function BeanLogo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="none">
      <ellipse cx="50" cy="54" rx="36" ry="40" fill="#8B5E3C" />
      <ellipse cx="50" cy="52" rx="33" ry="37" fill="#A0714F" />
      <path d="M50 22C50 22 42 42 42 54C42 66 50 86 50 86" stroke="#6B4226" strokeWidth="3" strokeLinecap="round" opacity="0.5" />
      <ellipse cx="50" cy="50" rx="30" ry="34" fill="url(#beanGrad)" opacity="0.3" />
      <circle cx="38" cy="42" r="4" fill="#2D1B0E" />
      <circle cx="62" cy="42" r="4" fill="#2D1B0E" />
      <circle cx="39" cy="41" r="1.2" fill="white" />
      <circle cx="63" cy="41" r="1.2" fill="white" />
      <path d="M42 56Q50 64 58 56" stroke="#2D1B0E" strokeWidth="2.5" strokeLinecap="round" fill="none" />
      <path d="M26 30L20 20" stroke="#A0714F" strokeWidth="2" strokeLinecap="round" opacity="0.6" />
      <path d="M74 30L80 20" stroke="#A0714F" strokeWidth="2" strokeLinecap="round" opacity="0.6" />
      <path d="M18 22L22 16" stroke="#4CAF50" strokeWidth="2" strokeLinecap="round" />
      <path d="M16 24L12 20" stroke="#4CAF50" strokeWidth="2" strokeLinecap="round" />
      <path d="M82 22L78 16" stroke="#EF5350" strokeWidth="2" strokeLinecap="round" />
      <path d="M84 24L88 20" stroke="#EF5350" strokeWidth="2" strokeLinecap="round" />
      <defs>
        <radialGradient id="beanGrad" cx="0.3" cy="0.3">
          <stop offset="0%" stopColor="white" /><stop offset="100%" stopColor="transparent" />
        </radialGradient>
      </defs>
    </svg>
  );
}

// ── Score Bar ──────────────────────────────────────────────
function ScoreBar({ value, label, color }: { value: number; label: string; color: string }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 2 }}>
        <span style={{ color: "#a08060", fontWeight: 700 }}>{label}</span>
        <span style={{ fontFamily: "monospace", fontWeight: 800, color }}>{value.toFixed(0)}</span>
      </div>
      <div style={{ background: "#f0ebe3", borderRadius: 100, height: 6, overflow: "hidden" }}>
        <div style={{
          width: `${Math.min(100, Math.max(0, value))}%`,
          height: "100%", borderRadius: 100, background: color,
          transition: "width 0.4s ease",
        }} />
      </div>
    </div>
  );
}

// ── Setup Row ──────────────────────────────────────────────
function SetupRow({ s, expanded, onToggle }: { s: Setup; expanded: boolean; onToggle: () => void }) {
  const isLong = s.bias === "long";
  const btScore = s.backtest_score ?? 50;
  const btColor = btScore >= 70 ? "#2e7d32" : btScore >= 50 ? "#e65100" : "#c62828";

  return (
    <div onClick={onToggle} style={{
      borderBottom: "1px solid #f0ebe3", padding: "14px 0", cursor: "pointer",
      background: expanded ? "#faf8f5" : "transparent", transition: "background 0.15s",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <div style={{ width: 72 }}>
          <div style={{ fontFamily: "'Nunito', sans-serif", fontSize: 15, fontWeight: 800, color: "#3d2b1f" }}>{s.symbol}</div>
          <span style={{
            fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 100,
            background: isLong ? "#e8f5e9" : "#ffebee", color: isLong ? "#2e7d32" : "#c62828",
          }}>{s.bias.toUpperCase()}</span>
        </div>
        <div style={{ flex: 1, minWidth: 120 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#3d2b1f" }}>{s.pattern_name}</div>
          <div style={{ fontSize: 11, color: "#a08060" }}>{s.timeframe}</div>
        </div>
        <div style={{ display: "flex", gap: 14 }}>
          {[
            { label: "ENTRY", val: s.entry_price, color: "#3d2b1f" },
            { label: "STOP", val: s.stop_loss, color: "#c62828" },
            { label: "TARGET", val: s.target_price, color: "#2e7d32" },
          ].map(({ label, val, color }) => (
            <div key={label} style={{ textAlign: "right" }}>
              <div style={{ fontSize: 10, color: "#a08060", fontWeight: 600, letterSpacing: 0.5 }}>{label}</div>
              <div style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 700, color }}>${val.toFixed(2)}</div>
            </div>
          ))}
        </div>
        <div style={{ width: 50, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "#a08060" }}>R:R</div>
          <div style={{ fontFamily: "monospace", fontSize: 15, fontWeight: 800, color: s.risk_reward_ratio >= 2 ? "#2e7d32" : "#3d2b1f" }}>
            {s.risk_reward_ratio.toFixed(1)}
          </div>
        </div>
        <div style={{ width: 50, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "#a08060" }}>CONF</div>
          <div style={{ fontFamily: "monospace", fontSize: 15, fontWeight: 800, color: s.confidence >= 0.7 ? "#2e7d32" : "#3d2b1f" }}>
            {Math.round(s.confidence * 100)}%
          </div>
        </div>
        {/* Backtest score badge */}
        <div style={{ width: 42, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "#a08060" }}>BT</div>
          <div style={{
            fontFamily: "monospace", fontSize: 12, fontWeight: 800, color: btColor,
            background: btScore >= 70 ? "#e8f5e9" : btScore >= 50 ? "#fff3e0" : "#ffebee",
            borderRadius: 4, padding: "1px 4px",
          }}>{btScore.toFixed(0)}</div>
        </div>
        <div style={{ width: 20, textAlign: "center", fontSize: 12, color: "#c4a882", transform: expanded ? "rotate(180deg)" : "", transition: "transform 0.2s" }}>▾</div>
      </div>

      {expanded && (
        <div style={{ marginTop: 14, paddingTop: 14, borderTop: "1px solid #f0ebe3", display: "flex", gap: 24, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ fontSize: 10, fontWeight: 800, color: "#a08060", letterSpacing: 1, textTransform: "uppercase", marginBottom: 6 }}>Description</div>
            <p style={{ fontSize: 12, color: "#5d4037", lineHeight: 1.7, margin: 0 }}>{s.description}</p>

            {/* Confidence breakdown */}
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 10, fontWeight: 800, color: "#a08060", letterSpacing: 1, textTransform: "uppercase", marginBottom: 8 }}>Confidence breakdown</div>
              <ScoreBar value={s.confidence * 100} label="Pattern signal" color="#8B5E3C" />
              <ScoreBar value={btScore} label="Backtest history" color={btColor} />
              <ScoreBar value={s.win_rate * 100} label="Historical win rate" color="#5d4037" />
            </div>
          </div>
          <div style={{ width: 220 }}>
            <div style={{ fontSize: 10, fontWeight: 800, color: "#a08060", letterSpacing: 1, textTransform: "uppercase", marginBottom: 8 }}>Details</div>
            {[
              ["Win Rate", `${Math.round(s.win_rate * 100)}%`],
              ["Max Attempts", String(s.max_attempts)],
              ["Backtest Score", `${btScore.toFixed(0)}/100`],
              ["Ideal Time", s.ideal_time || "—"],
            ].map(([label, val]) => (
              <div key={label} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                <span style={{ color: "#a08060" }}>{label}</span>
                <span style={{ fontFamily: "monospace", fontWeight: 700, color: "#3d2b1f" }}>{val}</span>
              </div>
            ))}
            <div style={{ marginTop: 8, fontSize: 10, color: "#a08060" }}>
              <span style={{ fontWeight: 800 }}>EXIT: </span>
              <span style={{ color: "#5d4037" }}>{s.exit_strategy || "—"}</span>
            </div>
            {Object.keys(s.key_levels).length > 0 && (
              <div style={{ marginTop: 10 }}>
                <div style={{ fontSize: 10, fontWeight: 800, color: "#a08060", letterSpacing: 1, marginBottom: 4 }}>KEY LEVELS</div>
                {Object.entries(s.key_levels).map(([k, v]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 2 }}>
                    <span style={{ color: "#a08060" }}>{k}</span>
                    <span style={{ fontFamily: "monospace", fontWeight: 700, color: "#3d2b1f" }}>
                      {typeof v === "number" ? `$${v.toFixed(2)}` : v}
                    </span>
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

// ── Main App ───────────────────────────────────────────────
function App() {
  const [symbol, setSymbol] = useState("AAPL");
  const [timeframe, setTimeframe] = useState("1d");
  const [daysBack, setDaysBack] = useState(30);
  const [setups, setSetups] = useState<Setup[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const [filterBias, setFilterBias] = useState("ALL");
  const [filterPattern, setFilterPattern] = useState("ALL");
  const [patternNames, setPatternNames] = useState<string[]>([]);
  const [hasScanned, setHasScanned] = useState(false);
  const [sortBy, setSortBy] = useState<"confidence" | "rr" | "backtest">("confidence");
  const [btStatus, setBtStatus] = useState<BacktestStatus | null>(null);
  const [btLoading, setBtLoading] = useState(false);

  useEffect(() => {
    fetch(`${API}/api/patterns`).then(r => r.json()).then(d => setPatternNames(d.patterns || [])).catch(() => {});
    fetch(`${API}/api/backtest/status`).then(r => r.json()).then(d => setBtStatus(d)).catch(() => {});
  }, []);

  const handleScan = async () => {
    setLoading(true); setError(""); setSetups([]); setHasScanned(true); setExpandedIdx(null);
    try {
      const res = await fetch(`${API}/api/scan?symbol=${symbol}&timeframe=${timeframe}&days_back=${daysBack}`);
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setSetups(data.setups);
    } catch (e: unknown) { setError(e instanceof Error ? e.message : "Unknown error"); }
    finally { setLoading(false); }
  };

  const handleRunBacktest = async (step: "fetch" | "run") => {
    setBtLoading(true);
    try {
      const endpoint = step === "fetch" ? "/api/backtest/fetch-data?symbols_count=50" : "/api/backtest/run?symbols_count=50";
      const method = "POST";
      await fetch(`${API}${endpoint}`, { method });
      // Poll status
      setTimeout(async () => {
        const res = await fetch(`${API}/api/backtest/status`);
        setBtStatus(await res.json());
        setBtLoading(false);
      }, 3000);
    } catch { setBtLoading(false); }
  };

  const filtered = useMemo(() => {
    let result = setups;
    if (filterBias !== "ALL") result = result.filter(s => s.bias === filterBias.toLowerCase());
    if (filterPattern !== "ALL") result = result.filter(s => s.pattern_name === filterPattern);
    result = [...result].sort((a, b) => {
      if (sortBy === "confidence") return b.confidence - a.confidence;
      if (sortBy === "rr") return b.risk_reward_ratio - a.risk_reward_ratio;
      return (b.backtest_score ?? 50) - (a.backtest_score ?? 50);
    });
    return result;
  }, [setups, filterBias, filterPattern, sortBy]);

  const longs = setups.filter(s => s.bias === "long").length;
  const shorts = setups.filter(s => s.bias === "short").length;

  const pill = (active: boolean) => ({
    fontSize: 12, fontWeight: active ? 800 : 600 as const, padding: "6px 14px", borderRadius: 100,
    cursor: "pointer" as const, border: "none",
    background: active ? "#3d2b1f" : "transparent", color: active ? "#fff" : "#a08060",
    fontFamily: "'Nunito', sans-serif",
  });

  return (
    <div style={{ background: "#fdf8f0", minHeight: "100vh", fontFamily: "'Nunito', sans-serif", color: "#3d2b1f" }}>
      {/* Header */}
      <div style={{ padding: "16px 32px", borderBottom: "1px solid #e8dfd0", display: "flex", justifyContent: "space-between", alignItems: "center", background: "#faf5ea" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <BeanLogo size={34} />
          <span style={{ fontSize: 22, fontWeight: 900, letterSpacing: -0.5 }}>AlphaBean</span>
          <span style={{ fontSize: 11, color: "#a08060", fontWeight: 600 }}>v2.1</span>
        </div>
        <div style={{ display: "flex", gap: 14, fontSize: 12, fontWeight: 700, alignItems: "center" }}>
          {setups.length > 0 && (
            <>
              <span style={{ color: "#2e7d32" }}>{longs} Long</span>
              <span style={{ color: "#c62828" }}>{shorts} Short</span>
              <span style={{ color: "#a08060" }}>{setups.length} Total</span>
            </>
          )}
          {btStatus?.has_results && (
            <span style={{ fontSize: 10, background: "#e8f5e9", color: "#2e7d32", padding: "3px 8px", borderRadius: 100, fontWeight: 700 }}>
              Backtest: {btStatus.total_signals} signals
            </span>
          )}
        </div>
      </div>

      <div style={{ maxWidth: 960, margin: "0 auto", padding: "24px 32px" }}>
        {/* Backtest status banner */}
        {!btStatus?.has_results && (
          <div style={{
            padding: "12px 16px", borderRadius: 10, background: "#fff8e1",
            border: "1px solid #ffe082", marginBottom: 16, fontSize: 12,
            display: "flex", justifyContent: "space-between", alignItems: "center",
          }}>
            <div>
              <strong style={{ color: "#e65100" }}>Backtest data not loaded</strong>
              <span style={{ color: "#a08060", marginLeft: 8 }}>
                Run a backtest to see historical pattern performance scores.
              </span>
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <button
                onClick={() => handleRunBacktest("fetch")}
                disabled={btLoading}
                style={{ fontSize: 11, fontWeight: 700, padding: "5px 12px", borderRadius: 8, border: "none", background: "#e65100", color: "#fff", cursor: btLoading ? "wait" : "pointer", fontFamily: "'Nunito'" }}
              >{btLoading ? "Working..." : "1. Fetch Data"}</button>
              <button
                onClick={() => handleRunBacktest("run")}
                disabled={btLoading}
                style={{ fontSize: 11, fontWeight: 700, padding: "5px 12px", borderRadius: 8, border: "none", background: "#3d2b1f", color: "#fff", cursor: btLoading ? "wait" : "pointer", fontFamily: "'Nunito'" }}
              >{btLoading ? "Working..." : "2. Run Backtest"}</button>
            </div>
          </div>
        )}

        {/* Scan Controls */}
        <div style={{ display: "flex", gap: 10, alignItems: "end", marginBottom: 20, flexWrap: "wrap" }}>
          <div>
            <label style={{ fontSize: 11, color: "#a08060", display: "block", marginBottom: 4, fontWeight: 700 }}>SYMBOL</label>
            <input type="text" value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())}
              onKeyDown={e => e.key === "Enter" && handleScan()} placeholder="AAPL"
              style={{ fontSize: 14, padding: "8px 14px", borderRadius: 10, border: "1.5px solid #e8dfd0", width: 100, fontFamily: "'Nunito', monospace", fontWeight: 800, background: "#fff" }} />
          </div>
          <div>
            <label style={{ fontSize: 11, color: "#a08060", display: "block", marginBottom: 4, fontWeight: 700 }}>TIMEFRAME</label>
            <select value={timeframe} onChange={e => setTimeframe(e.target.value)}
              style={{ fontSize: 14, padding: "8px 14px", borderRadius: 10, border: "1.5px solid #e8dfd0", background: "#fff", fontFamily: "'Nunito'", fontWeight: 600 }}>
              <option value="15min">15 min</option>
              <option value="1h">1 hour</option>
              <option value="1d">1 day</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: 11, color: "#a08060", display: "block", marginBottom: 4, fontWeight: 700 }}>DAYS</label>
            <input type="number" value={daysBack} onChange={e => setDaysBack(Number(e.target.value))}
              style={{ fontSize: 14, padding: "8px 14px", borderRadius: 10, border: "1.5px solid #e8dfd0", width: 70, fontFamily: "monospace", background: "#fff" }} />
          </div>
          <button onClick={handleScan} disabled={loading || !symbol}
            style={{ fontSize: 14, fontWeight: 800, padding: "9px 28px", borderRadius: 10, border: "none", background: loading ? "#c4a882" : "#3d2b1f", color: "#fff", cursor: loading ? "wait" : "pointer", fontFamily: "'Nunito'" }}>
            {loading ? "Scanning..." : "Scan"}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div style={{ padding: "12px 16px", borderRadius: 10, background: "#ffebee", color: "#c62828", fontSize: 13, marginBottom: 16 }}>
            <strong>Error:</strong> {error}
            <div style={{ marginTop: 4, fontSize: 11, color: "#a08060" }}>
              Backend: <code style={{ background: "#fff", padding: "1px 6px", borderRadius: 4, fontSize: 11 }}>uvicorn backend.main:app --reload --port 8000</code>
            </div>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div style={{ textAlign: "center", padding: 48, color: "#a08060" }}>
            <BeanLogo size={48} />
            <div style={{ fontSize: 14, marginTop: 12, fontWeight: 700 }}>Scanning patterns...</div>
            <div style={{ fontSize: 12, marginTop: 4 }}>Fetching data and running 21 detectors</div>
          </div>
        )}

        {/* Results */}
        {!loading && setups.length > 0 && (
          <>
            <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap", alignItems: "center" }}>
              <div style={{ display: "flex", gap: 3, background: "#f0ebe3", borderRadius: 100, padding: 3 }}>
                {["ALL", "LONG", "SHORT"].map(b => (
                  <button key={b} style={pill(filterBias === b)} onClick={() => setFilterBias(b)}>{b}</button>
                ))}
              </div>
              <select value={filterPattern} onChange={e => setFilterPattern(e.target.value)}
                style={{ fontSize: 12, padding: "6px 12px", borderRadius: 100, border: "1.5px solid #e8dfd0", background: "#fff", fontFamily: "'Nunito'", fontWeight: 600, color: "#3d2b1f" }}>
                <option value="ALL">All Patterns</option>
                {patternNames.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
              <div style={{ marginLeft: "auto", display: "flex", gap: 4, alignItems: "center", fontSize: 11, color: "#a08060" }}>
                <span style={{ fontWeight: 700 }}>Sort:</span>
                {(["confidence", "rr", "backtest"] as const).map(s => (
                  <button key={s} onClick={() => setSortBy(s)}
                    style={{ ...pill(sortBy === s), fontSize: 11, padding: "4px 10px" }}>
                    {s === "confidence" ? "Conf" : s === "rr" ? "R:R" : "BT"}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderBottom: "2px solid #3d2b1f", fontSize: 10, fontWeight: 800, color: "#a08060", letterSpacing: 0.8, textTransform: "uppercase" as const }}>
              <div style={{ width: 72 }}>Symbol</div>
              <div style={{ flex: 1 }}>Pattern</div>
              <div style={{ display: "flex", gap: 14 }}>
                <div style={{ width: 60, textAlign: "right" as const }}>Entry</div>
                <div style={{ width: 60, textAlign: "right" as const }}>Stop</div>
                <div style={{ width: 60, textAlign: "right" as const }}>Target</div>
              </div>
              <div style={{ width: 50, textAlign: "center" as const }}>R:R</div>
              <div style={{ width: 50, textAlign: "center" as const }}>Conf</div>
              <div style={{ width: 42, textAlign: "center" as const }}>BT</div>
              <div style={{ width: 20 }} />
            </div>

            {filtered.map((s, i) => (
              <SetupRow key={i} s={s} expanded={expandedIdx === i} onToggle={() => setExpandedIdx(expandedIdx === i ? null : i)} />
            ))}
            {filtered.length === 0 && (
              <div style={{ textAlign: "center", padding: 32, color: "#a08060", fontSize: 13 }}>No setups match your filters.</div>
            )}
          </>
        )}

        {!loading && hasScanned && setups.length === 0 && !error && (
          <div style={{ textAlign: "center", padding: 48, color: "#a08060" }}>
            <BeanLogo size={48} />
            <div style={{ fontSize: 16, fontWeight: 700, marginTop: 12 }}>No patterns detected</div>
            <div style={{ fontSize: 12, marginTop: 4 }}>
              <strong>{symbol}</strong> on <strong>{timeframe}</strong> didn't match any of 21 detectors. Try a different symbol or timeframe.
            </div>
          </div>
        )}

        {!loading && !hasScanned && (
          <div style={{ textAlign: "center", padding: 48, color: "#a08060" }}>
            <BeanLogo size={56} />
            <div style={{ fontSize: 18, fontWeight: 800, marginTop: 12, color: "#3d2b1f" }}>Welcome to AlphaBean</div>
            <div style={{ fontSize: 13, marginTop: 6, maxWidth: 420, margin: "6px auto 0" }}>
              Enter a ticker and hit Scan. AlphaBean runs 21 pattern detectors on 15-min, hourly, or daily candles
              {btStatus?.has_results ? " — enriched with backtest performance data." : "."}
            </div>
          </div>
        )}

        <div style={{ textAlign: "center", marginTop: 40, padding: "16px 0", borderTop: "1px solid #f0ebe3" }}>
          <span style={{ fontSize: 11, color: "#c4a882" }}>AlphaBean v2.1 — 21 Detectors — Backtested</span>
        </div>
      </div>
    </div>
  );
}

export default App;