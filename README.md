<p align="center">
  <img src="frontend/public/juicer-logo.png" alt="Juicer" width="120" />
</p>

<h1 align="center">Juicer</h1>

<p align="center">
  <strong>Squeeze the market for alpha.</strong><br/>
  An AI-powered quantitative trade scanner that detects 47 chart patterns across 5-minute and 15-minute timeframes, scores them with a 6-factor composite model, evaluates them with local AI, and surfaces the highest-conviction setups — all running locally, all free.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/patterns-47-6c5ce7?style=flat-square" alt="47 Patterns"/>
  <img src="https://img.shields.io/badge/scoring-6--factor-00d2a0?style=flat-square" alt="6-Factor Scoring"/>
  <img src="https://img.shields.io/badge/AI-Ollama%20Local-ff6b6b?style=flat-square" alt="Ollama AI"/>
  <img src="https://img.shields.io/badge/cost-$0-fdcb6e?style=flat-square" alt="$0 Cost"/>
  <img src="https://img.shields.io/badge/python-3.12-3572A5?style=flat-square" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/react-TypeScript-61DAFB?style=flat-square" alt="React + TypeScript"/>
</p>

---

## What is Juicer?

Juicer is a full-stack quantitative trading platform that scans stocks for actionable chart pattern setups. It fetches intraday price data, runs 47 pattern detectors built on structural geometry (not indicators), scores every setup with a multi-factor model trained on walk-forward backtests, and optionally evaluates each trade with a local LLM for a final gut-check.

It is designed for **intraday and swing traders** who want to find high-probability setups faster than they can manually scan charts.

### Key Principles

- **No future data leakage.** Every pattern is detected using only bars available at detection time. The backtest engine is strictly walk-forward.
- **Structure-first detection.** Patterns are identified from price geometry (swings, trendlines, support/resistance), not lagging indicators.
- **Fully local.** All AI runs on Ollama (Llama 3.1 8B). No paid API calls required. Your data never leaves your machine.
- **Backtested.** Every pattern has been validated across 300+ symbols with walk-forward testing. Edge scores are real, not theoretical.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        JUICER PLATFORM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│  │  Yahoo    │   │  Massive.com │   │    Google News RSS     │  │
│  │ Trending  │──▶│  (Polygon)   │──▶│    + Finnhub           │  │
│  │  50 tkrs  │   │  5m + 15m    │   │    Headlines           │  │
│  └──────────┘   └──────┬───────┘   └───────────┬────────────┘  │
│                         │                        │               │
│                         ▼                        │               │
│  ┌─────────────────────────────────────┐         │               │
│  │     STRUCTURAL PRIMITIVES           │         │               │
│  │  ┌─────────┐ ┌──────────┐ ┌─────┐  │         │               │
│  │  │ Zigzag  │ │Trendlines│ │ S/R │  │         │               │
│  │  │ Swings  │ │ Channels │ │Zones│  │         │               │
│  │  └─────────┘ └──────────┘ └─────┘  │         │               │
│  └──────────────────┬──────────────────┘         │               │
│                     ▼                            │               │
│  ┌─────────────────────────────────────┐         │               │
│  │     47 PATTERN CLASSIFIERS          │         │               │
│  │                                     │         │               │
│  │  Classical (16)  │ Candlestick (10) │         │               │
│  │  H&S, Triangles  │ Engulfing, Doji  │         │               │
│  │  Flags, Wedges   │ Stars, Hammer    │         │               │
│  │                  │                  │         │               │
│  │  SMB Scalps (11) │ Quant (10)       │         │               │
│  │  ORB, RubberBand │ Momentum, VWAP   │         │               │
│  │  Tidal Wave      │ Mean Reversion   │         │               │
│  └──────────────────┬──────────────────┘         │               │
│                     ▼                            │               │
│  ┌─────────────────────────────────────┐         │               │
│  │     6-FACTOR COMPOSITE SCORING      │         │               │
│  │                                     │         │               │
│  │  Pattern Confidence ────── 20%      │         │               │
│  │  Statistical Features ──── 25%      │         │               │
│  │  Strategy Hot Score ────── 20%      │         │               │
│  │  Market Regime Fit ─────── 15%      │         │               │
│  │  Backtest Edge ─────────── 10%      │         │               │
│  │  Volume Confirmation ───── 10%      │         │               │
│  │                                     │         │               │
│  │  Output: Composite Score 0-100      │         │               │
│  └──────────────────┬──────────────────┘         │               │
│                     ▼                            ▼               │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              OLLAMA AI EVALUATION                     │       │
│  │                                                       │       │
│  │  Setup + News + Regime + Backtest Stats               │       │
│  │         ▼                                             │       │
│  │  Llama 3.1 8B  ──▶  ✓ CONFIRMED / ⚠ CAUTION / ✗ DENIED  │  │
│  │                      + reasoning + key factors        │       │
│  └──────────────────────┬────────────────────────────────┘       │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              REACT FRONTEND (Vite + Tailwind)         │       │
│  │                                                       │       │
│  │  Opportunities ◀──▶ Scan ◀──▶ Chart ◀──▶ Track       │       │
│  │  Backtest Viewer    Filters    3 Lines    Live P&L    │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

### Pattern Detection Engine

Juicer runs **47 pattern detectors** organized into 4 categories:

| Category | Count | Examples |
|----------|-------|---------|
| **Classical** | 16 | Head & Shoulders, Double Top/Bottom, Triangles, Flags, Wedges, Cup & Handle, Rectangle, Pennant |
| **Candlestick** | 10 | Bullish/Bearish Engulfing, Morning/Evening Star, Hammer, Shooting Star, Doji, Three White Soldiers |
| **SMB Scalps** | 11 | ORB 15/30min, RubberBand, HitchHiker, Tidal Wave, Gap Give & Go, Fashionably Late, Spencer |
| **Quantitative** | 10 | Momentum Breakout, Vol Compression, Mean Reversion, VWAP Reversion, Donchian Breakout |

Every pattern is detected from **structural price geometry** — swings, trendlines, support/resistance clusters — not from lagging indicators. Each detection produces an entry price, stop loss, target price, and risk-reward ratio.

### 6-Factor Composite Scoring

Every detected setup is scored 0-100 using six independent factors:

| Factor | Weight | What it measures |
|--------|--------|-----------------|
| Pattern Confidence | 20% | How cleanly the geometric pattern matches ideal form |
| Statistical Features | 25% | 8 features: momentum, volatility, vol compression, RVOL, trend strength, range breakout, mean reversion z-score, regime score |
| Strategy Hot Score | 20% | Rolling 60-trade performance of this pattern type (win rate, profit factor, expectancy) |
| Market Regime Alignment | 15% | Whether the setup's bias matches the current regime (Bull, Bear, High-Vol, Mean-Reverting) |
| Backtest Edge | 10% | Historical edge score from the walk-forward backtest across 300 symbols |
| Volume Confirmation | 10% | Relative volume (RVOL) at the point of detection |

Setups scoring below **45** are automatically filtered out.

### AI Evaluation (Ollama)

The top 50% of setups (by composite score) are sent to a local **Llama 3.1 8B** model running on Ollama. The AI receives the setup details, recent news headlines, market regime, and backtest statistics, then returns:

- **Verdict:** ✓ CONFIRMED, ⚠ CAUTION, or ✗ DENIED
- **Reasoning:** A sentence explaining the AI's logic
- **Key Factors:** News sentiment, regime alignment, backtest strength

### Market Regime Detection

Juicer classifies the current market into one of four regimes using SPY data:

| Regime | Indicator | Best Strategies |
|--------|-----------|----------------|
| 🟢 **Trending Bull** | ATR expanding, price > SMA | Breakouts, trend pullbacks |
| 🔴 **Trending Bear** | ATR expanding, price < SMA | Short setups, breakdowns |
| 🟡 **High Volatility** | ATR spike above 1.5× normal | Gap plays, ORB, momentum |
| ⚪ **Mean Reverting** | ATR contracting, range-bound | Mean reversion, VWAP |

### SPY Correlation

Every scanned stock gets a live **relative strength/weakness** label compared to SPY, computed from today's 15-minute session returns:

- 💪 **Relative Strength** — Stock up while SPY is down
- 📉 **Relative Weakness** — Stock down while SPY is up
- **Outperforming / Underperforming / Holding Up / In Line**

### Walk-Forward Backtest

The included `run_backtest.py` script validates all 47 patterns across hundreds of symbols using strict walk-forward methodology:

1. At bar N, detect patterns using **only** bars 0 through N
2. Record entry, stop, and target from the detection
3. Walk bars N+1 forward: did target hit before stop?
4. Aggregate win rate, profit factor, expectancy, edge score per pattern
5. Save to `cache/backtest_results.json`

No future data is ever used. The backtest has been run across **300 symbols over 90 days**, producing **24,000+ signals** for validation.

### Live Trade Tracker

Click **+ Track** on any setup to add it to the live tracking panel. Juicer polls current prices every 5 minutes and shows:

- Current price vs entry price
- P&L in R-multiples (risk units)
- Status: ⏳ ACTIVE, 🎯 TARGET HIT, 🛑 STOPPED, or ⏸ WAITING (market closed)

### Per-Symbol Analytics

When viewing a chart, click **📊 Show Pattern Stats** to see how patterns have historically performed **on that specific stock**. Shows the highlighted pattern's stats plus the top 5 best-performing patterns on that ticker.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12, FastAPI, NumPy (vectorized math) |
| **Frontend** | React 18, TypeScript, Vite, Framer Motion |
| **Charts** | TradingView Lightweight Charts v4 |
| **Data** | Massive.com (Polygon.io) REST API |
| **AI** | Ollama + Llama 3.1 8B (local, free) |
| **News** | Google News RSS + Finnhub (free tier) |
| **Trending** | Yahoo Finance trending page scraper |

---

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- [Ollama](https://ollama.ai) installed with `llama3.1:8b` pulled
- A [Massive.com](https://massive.com) (Polygon) API key (free tier works)

### 1. Clone and install

```bash
git clone https://github.com/your-username/juicer.git
cd juicer

# Backend
pip install fastapi uvicorn numpy requests feedparser python-dotenv --break-system-packages

# Frontend
cd frontend
npm install
npm install framer-motion lightweight-charts
cd ..

# AI model
ollama pull llama3.1:8b
```

### 2. Configure

Create a `.env` file in the project root:

```env
MASSIVE_API_KEY=your_polygon_api_key_here
FINNHUB_API_KEY=your_finnhub_key_here   # optional, free at finnhub.io
```

### 3. Run the backtest (recommended first)

```bash
python run_backtest.py
```

This generates edge scores that power the composite scoring system. Takes ~20-30 minutes for 300 symbols. You can start with fewer:

```bash
python run_backtest.py --symbols AAPL,NVDA,TSLA,AMD,META --days 30
```

### 4. Start the platform

```bash
# Terminal 1: Backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev

# Terminal 3: Ollama (if not already running)
ollama serve
```

Open **http://localhost:5173** and you're live.

---

## User Guide

### Opportunities Page

When you open Juicer, the **Opportunities** tab loads automatically. It:

1. Fetches the top 50 trending tickers
2. Scans each through the 47-pattern engine
3. Keeps only the **top 2 setups per symbol** (highest composite score)
4. Removes anything scoring below 45
5. Sends the top 50% to Ollama for AI evaluation
6. Displays the ranked results

The **trending tickers** appear as scrollable cards at the top. Click any card to scan that symbol individually.

**Hot strategies** shows which pattern types have the best recent rolling performance.

### Scanning a Ticker

Switch to the **Scan** tab, type a symbol (e.g., `NVDA`), and press Enter or click **Squeeze 🧃**.

Juicer runs all 47 pattern detectors on both 5-minute and 15-minute bars simultaneously. If the same pattern fires on both timeframes, it's marked with **★ 5m & 15m** and gets a confidence boost.

### Reading a Setup

Each row shows:

| Element | Meaning |
|---------|---------|
| **Symbol + Bias** | The ticker and whether it's a LONG or SHORT setup |
| **Pattern Name** | Which of the 47 patterns was detected |
| **Category Badge** | Classical, Candle, SMB, or Quant |
| **Timeframe** | 5m, 15m, or ★ 5m & 15m (multi-timeframe) |
| **AI Verdict** | ✓ CONFIRMED, ⚠ CAUTION, or ✗ DENIED |
| **Correlation** | Relative strength/weakness vs SPY |
| **Trigger** | The exact price condition: "⚡ BUY if NVDA reaches $137.50" |
| **Entry / Stop / Target** | The trade's key levels |
| **R:R** | Risk-reward ratio (green if ≥ 2.0) |
| **Score** | Composite score 0-100 (green ≥ 65, gold ≥ 45) |

### Viewing Charts

Click any setup row to expand the candlestick chart. You'll see:

- **Purple dashed line** — Entry price
- **Red dashed line** — Stop loss
- **Green dashed line** — Target price

Below the chart, click **📊 Show Pattern Stats** to see how this pattern has historically performed on this specific stock.

### Tracking Trades

Click **+ Track** on any setup to add it to the live tracking panel at the top. The tracker shows real-time P&L in R-multiples, updated every 5 minutes during market hours. When the market is closed, it shows "⏸ WAITING" instead of false P&L calculations.

### Filtering and Sorting

Use the filter bar to narrow results:

- **ALL / LONG / SHORT** — Filter by trade direction
- **All / Classical / Candle / SMB / Quant** — Filter by pattern category
- **Score / R:R** — Sort by composite score or risk-reward ratio

### Backtest Viewer

Click **📊 Backtest** in the header to open the full backtest results modal. This shows all 47 patterns ranked by edge score, with columns for signals, win rate, profit factor, expectancy, and letter grade (A through F). Click column headers to re-sort.

### Light/Dark Mode

Toggle ☀️/🌙 in the top-right corner.

---

## Project Structure

```
juicer/
├── backend/
│   ├── structures/          # Phase 1: Zigzag swings, trendlines, S/R zones
│   ├── features/            # Phase 2: 8 statistical features + SPY correlation
│   ├── regime/              # Phase 3: 4-state market regime detector
│   ├── patterns/            # Phase 4: 47 pattern classifiers
│   ├── strategies/          # Phase 5: Rolling 60-trade strategy evaluator
│   ├── scoring/             # Phase 6: 6-factor composite scoring
│   ├── scanner/             # Phase 7: Orchestrator engine
│   ├── news/                # Phase 8: Finnhub + Google News RSS pipeline
│   ├── ai/                  # Phase 8: Ollama agent + trending detector
│   ├── analytics/           # Per-symbol pattern performance stats
│   ├── data/                # Schemas + Massive.com API client
│   └── main.py              # FastAPI server
├── frontend/
│   └── src/
│       └── App.tsx           # Complete React UI
├── run_backtest.py           # Walk-forward backtest script
├── fetch_symbols.py          # Fetches top N symbols by volume
├── cache/
│   ├── backtest_results.json # Aggregate backtest results
│   ├── backtest_by_symbol/   # Per-symbol analytics cache
│   ├── strategy_performance.json
│   ├── in_play.json          # Trending tickers cache (30 min)
│   └── correlation/          # SPY correlation cache
└── README.md
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Server status, pattern count, Ollama status |
| `/api/scan?symbol=AAPL&mode=today` | GET | Scan a symbol (5m + 15m, scored, AI evaluated) |
| `/api/top-opportunities` | GET | Full pipeline: trending → scan → score → AI → rank |
| `/api/in-play` | GET | Trending tickers (cached 30 min) |
| `/api/chart/{symbol}?timeframe=5min` | GET | Candlestick data for Lightweight Charts |
| `/api/regime` | GET | Current market regime based on SPY |
| `/api/correlation/{symbol}` | GET | Live SPY relative strength/weakness |
| `/api/hot-strategies` | GET | Top performing pattern types |
| `/api/backtest/results` | GET | Aggregate backtest statistics |
| `/api/backtest/patterns?sort=edge_score` | GET | All patterns sorted by metric |
| `/api/analytics/{symbol}?pattern=...` | GET | Per-symbol pattern performance |
| `/api/track-prices?symbols=AAPL,NVDA` | GET | Batch price fetch for live tracking |
| `/api/news/{symbol}` | GET | Headlines for a ticker |
| `/api/ollama/status` | GET | Check if Ollama is running |

---

## Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Minimum composite score | `main.py` | 45 |
| Max symbols in opportunities | `main.py` query param | 15 |
| Setups per symbol (opportunities) | `main.py` query param | 2 |
| AI evaluation | `main.py` query param | On |
| Trending cache TTL | `inplay_detector.py` | 30 minutes |
| Correlation freshness | `correlation.py` | Live (no cache) |
| Track price poll interval | `App.tsx` | 5 minutes |
| Market hours filter | `main.py` | 9:30-16:00 ET (active only when open) |

---

## Disclaimer

Juicer is a research and educational tool. It does not constitute financial advice. Past pattern performance does not guarantee future results. Always do your own research and manage your own risk. Trade at your own discretion.

---

<p align="center">
  <strong>Juicer v1.0</strong> — 47 Detectors • 6-Factor Scoring • Local AI • Squeeze the Market 🧃
</p>