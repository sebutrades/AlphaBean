import { useState, useMemo, useEffect } from "react";

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

// ============================================================
// API
// ============================================================
const API_BASE = "http://localhost:8000";

async function fetchScan(symbol: string, timeframe: string, daysBack: number): Promise<Setup[]> {
  const res = await fetch(
    `${API_BASE}/api/scan?symbol=${symbol}&timeframe=${timeframe}&days_back=${daysBack}`
  );
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const data = await res.json();
  return data.setups;
}

// ============================================================
// COMPONENTS
// ============================================================

function SetupRow({ s, expanded, onToggle }: { s: Setup; expanded: boolean; onToggle: () => void }) {
  const isLong = s.bias === "long";
  const biasColor = isLong ? "#16a34a" : "#dc2626";
  const biasBg = isLong ? "#f0fdf4" : "#fef2f2";

  return (
    <div onClick={onToggle} style={{ borderBottom: "1px solid #f4f4f5", padding: "14px 0", cursor: "pointer", background: expanded ? "#fafafa" : "transparent" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <div style={{ width: 72 }}>
          <div style={{ fontFamily: "monospace", fontSize: 15, fontWeight: 700, color: "#18181b" }}>{s.symbol}</div>
          <span style={{ fontSize: 10, fontWeight: 600, padding: "2px 8px", borderRadius: 100, background: biasBg, color: biasColor }}>{s.bias.toUpperCase()}</span>
        </div>
        <div style={{ flex: 1, minWidth: 120 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: "#18181b" }}>{s.pattern_name}</div>
          <div style={{ fontSize: 11, color: "#a1a1aa" }}>{s.timeframe}</div>
        </div>
        <div style={{ display: "flex", gap: 14 }}>
          {[
            { label: "ENTRY", val: s.entry_price, color: "#18181b" },
            { label: "STOP", val: s.stop_loss, color: "#dc2626" },
            { label: "TARGET", val: s.target_price, color: "#16a34a" },
          ].map(({ label, val, color }) => (
            <div key={label} style={{ textAlign: "right" }}>
              <div style={{ fontSize: 10, color: "#a1a1aa", fontWeight: 500, letterSpacing: 0.5 }}>{label}</div>
              <div style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 600, color }}>${val.toFixed(2)}</div>
            </div>
          ))}
        </div>
        <div style={{ width: 50, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "#a1a1aa" }}>R:R</div>
          <div style={{ fontFamily: "monospace", fontSize: 15, fontWeight: 800, color: s.risk_reward_ratio >= 2 ? "#16a34a" : "#18181b" }}>{s.risk_reward_ratio.toFixed(1)}</div>
        </div>
        <div style={{ width: 50, textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "#a1a1aa" }}>CONF</div>
          <div style={{ fontFamily: "monospace", fontSize: 15, fontWeight: 800, color: s.confidence >= 0.7 ? "#16a34a" : "#18181b" }}>{Math.round(s.confidence * 100)}%</div>
        </div>
        <div style={{ width: 20, textAlign: "center", fontSize: 14, color: "#d4d4d8", transform: expanded ? "rotate(180deg)" : "rotate(0)", transition: "transform 0.2s" }}>▾</div>
      </div>
      {expanded && (
        <div style={{ marginTop: 14, paddingTop: 14, borderTop: "1px solid #f4f4f5", display: "flex", gap: 24, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: "#a1a1aa", letterSpacing: 1, textTransform: "uppercase", marginBottom: 6 }}>Description</div>
            <p style={{ fontSize: 12, color: "#52525b", lineHeight: 1.65, margin: 0 }}>{s.description}</p>
          </div>
          <div style={{ width: 220 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: "#a1a1aa", letterSpacing: 1, textTransform: "uppercase", marginBottom: 8 }}>Details</div>
            {[
              ["Win Rate", `${Math.round(s.win_rate * 100)}%`],
              ["Max Attempts", String(s.max_attempts)],
              ["Ideal Time", s.ideal_time || "—"],
            ].map(([label, val]) => (
              <div key={label} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                <span style={{ color: "#a1a1aa" }}>{label}</span>
                <span style={{ fontFamily: "monospace", fontWeight: 600, color: "#18181b" }}>{val}</span>
              </div>
            ))}
            <div style={{ marginTop: 8, fontSize: 10, color: "#a1a1aa" }}>
              <span style={{ fontWeight: 700 }}>EXIT: </span>
              <span style={{ color: "#52525b" }}>{s.exit_strategy || "—"}</span>
            </div>
            {Object.keys(s.key_levels).length > 0 && (
              <div style={{ marginTop: 10 }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: "#a1a1aa", letterSpacing: 1, marginBottom: 4 }}>KEY LEVELS</div>
                {Object.entries(s.key_levels).map(([k, v]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 2 }}>
                    <span style={{ color: "#a1a1aa" }}>{k}</span>
                    <span style={{ fontFamily: "monospace", fontWeight: 600, color: "#18181b" }}>{typeof v === "number" ? `$${v.toFixed(2)}` : v}</span>
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
  const [symbol, setSymbol] = useState("AAPL");
  const [timeframe, setTimeframe] = useState("1d");
  const [daysBack, setDaysBack] = useState(30);
  const [setups, setSetups] = useState<Setup[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const [filterBias, setFilterBias] = useState("ALL");
  const [hasScanned, setHasScanned] = useState(false);

  const handleScan = async () => {
    setLoading(true);
    setError("");
    setSetups([]);
    setHasScanned(true);
    try {
      const results = await fetchScan(symbol, timeframe, daysBack);
      setSetups(results);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const filtered = useMemo(() => {
    if (filterBias === "ALL") return setups;
    return setups.filter((s) => s.bias === filterBias.toLowerCase());
  }, [setups, filterBias]);

  const longs = setups.filter((s) => s.bias === "long").length;
  const shorts = setups.filter((s) => s.bias === "short").length;

  const tabStyle = (active: boolean) => ({
    fontSize: 12,
    fontWeight: active ? 700 : 500,
    padding: "6px 14px",
    borderRadius: 8,
    cursor: "pointer" as const,
    border: "none",
    background: active ? "#18181b" : "transparent",
    color: active ? "#fff" : "#71717a",
  });

  return (
    <div style={{ background: "#fff", minHeight: "100vh", fontFamily: "'Inter', -apple-system, system-ui, sans-serif", color: "#18181b" }}>
      {/* Header */}
      <div style={{ padding: "20px 32px", borderBottom: "1px solid #f4f4f5", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
          <span style={{ fontSize: 20, fontWeight: 800, letterSpacing: -0.5 }}>EdgeFinder</span>
          <span style={{ fontSize: 11, color: "#a1a1aa" }}>v1.0</span>
        </div>
        {setups.length > 0 && (
          <div style={{ display: "flex", gap: 12, fontSize: 12 }}>
            <span style={{ color: "#16a34a", fontWeight: 600 }}>{longs} Long</span>
            <span style={{ color: "#dc2626", fontWeight: 600 }}>{shorts} Short</span>
            <span style={{ color: "#71717a" }}>{setups.length} Total</span>
          </div>
        )}
      </div>

      <div style={{ maxWidth: 900, margin: "0 auto", padding: "24px 32px" }}>
        {/* Scan Controls */}
        <div style={{ display: "flex", gap: 10, alignItems: "end", marginBottom: 24, flexWrap: "wrap" }}>
          <div>
            <label style={{ fontSize: 11, color: "#71717a", display: "block", marginBottom: 4, fontWeight: 600 }}>SYMBOL</label>
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              onKeyDown={(e) => e.key === "Enter" && handleScan()}
              placeholder="AAPL"
              style={{ fontSize: 14, padding: "8px 14px", borderRadius: 8, border: "1px solid #e4e4e7", width: 100, fontFamily: "monospace", fontWeight: 700 }}
            />
          </div>
          <div>
            <label style={{ fontSize: 11, color: "#71717a", display: "block", marginBottom: 4, fontWeight: 600 }}>TIMEFRAME</label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              style={{ fontSize: 14, padding: "8px 14px", borderRadius: 8, border: "1px solid #e4e4e7", background: "#fff" }}
            >
              <option value="1min">1 min</option>
              <option value="5min">5 min</option>
              <option value="15min">15 min</option>
              <option value="1h">1 hour</option>
              <option value="1d">1 day</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: 11, color: "#71717a", display: "block", marginBottom: 4, fontWeight: 600 }}>DAYS BACK</label>
            <input
              type="number"
              value={daysBack}
              onChange={(e) => setDaysBack(Number(e.target.value))}
              style={{ fontSize: 14, padding: "8px 14px", borderRadius: 8, border: "1px solid #e4e4e7", width: 70, fontFamily: "monospace" }}
            />
          </div>
          <button
            onClick={handleScan}
            disabled={loading || !symbol}
            style={{
              fontSize: 14, fontWeight: 700, padding: "9px 28px", borderRadius: 8, border: "none",
              background: loading ? "#a1a1aa" : "#18181b", color: "#fff", cursor: loading ? "wait" : "pointer",
            }}
          >
            {loading ? "Scanning..." : "Scan"}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div style={{ padding: "12px 16px", borderRadius: 8, background: "#fef2f2", color: "#dc2626", fontSize: 13, marginBottom: 16 }}>
            <strong>Error:</strong> {error}
            <div style={{ marginTop: 4, fontSize: 11, color: "#71717a" }}>
              Make sure the backend is running: <code style={{ background: "#fff", padding: "1px 4px", borderRadius: 3 }}>uvicorn backend.main:app --reload --port 8000</code>
            </div>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div style={{ textAlign: "center", padding: 40, color: "#a1a1aa" }}>
            <div style={{ fontSize: 14, marginBottom: 8 }}>Fetching data from Massive.com and running 21 pattern detectors...</div>
            <div style={{ fontSize: 12 }}>This can take 10-20 seconds</div>
          </div>
        )}

        {/* Results */}
        {!loading && setups.length > 0 && (
          <>
            {/* Filter tabs */}
            <div style={{ display: "flex", gap: 4, background: "#f4f4f5", borderRadius: 10, padding: 3, marginBottom: 16, width: "fit-content" }}>
              {["ALL", "LONG", "SHORT"].map((b) => (
                <button key={b} style={tabStyle(filterBias === b)} onClick={() => setFilterBias(b)}>{b}</button>
              ))}
            </div>

            {/* Column headers */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderBottom: "2px solid #18181b", fontSize: 10, fontWeight: 700, color: "#a1a1aa", letterSpacing: 0.8, textTransform: "uppercase" as const }}>
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

            {/* Rows */}
            {filtered.map((s, i) => (
              <SetupRow key={i} s={s} expanded={expandedIdx === i} onToggle={() => setExpandedIdx(expandedIdx === i ? null : i)} />
            ))}
          </>
        )}

        {/* Empty state */}
        {!loading && hasScanned && setups.length === 0 && !error && (
          <div style={{ textAlign: "center", padding: 48, color: "#a1a1aa" }}>
            <div style={{ fontSize: 16, marginBottom: 8 }}>No patterns detected</div>
            <div style={{ fontSize: 12 }}>
              This is normal — it means the current price data for <strong>{symbol}</strong> on the <strong>{timeframe}</strong> timeframe
              doesn't match any of the 21 pattern detectors right now. Try a different symbol, timeframe, or look-back period.
            </div>
          </div>
        )}

        {/* Initial state */}
        {!loading && !hasScanned && (
          <div style={{ textAlign: "center", padding: 48, color: "#a1a1aa" }}>
            <div style={{ fontSize: 16, marginBottom: 8 }}>Enter a ticker and hit Scan</div>
            <div style={{ fontSize: 12 }}>
              EdgeFinder will fetch price data from Massive.com and run 21 pattern detectors
              (10 SMB scalps + 11 classical patterns) to find trade setups.
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={{ textAlign: "center", marginTop: 40, padding: "16px 0", borderTop: "1px solid #f4f4f5" }}>
          <span style={{ fontSize: 11, color: "#d4d4d8" }}>EdgeFinder v1.0 · 21 Pattern Detectors</span>
        </div>
      </div>
    </div>
  );
}

export default App;