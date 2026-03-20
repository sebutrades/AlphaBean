"""
fetch_symbols.py — Fetch top US stocks by trading volume

Uses Massive.com (Polygon) grouped daily endpoint to get all US stocks
for a recent trading day, sorts by volume, saves top N.

Usage:
  python fetch_symbols.py              # Top 300, saves to cache/
  python fetch_symbols.py --count 500  # Top 500
  python fetch_symbols.py --list       # Just print the cached list

Output: cache/top_symbols.json
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = Path("cache/top_symbols.json")

# Fallback: top ~300 US stocks by typical volume (March 2025)
# These cover mega-cap + high-volume mid-cap across all sectors
FALLBACK_SYMBOLS = [
    # Mega Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "AVGO", "ORCL",
    "ADBE", "CRM", "AMD", "INTC", "CSCO", "QCOM", "TXN", "MU", "AMAT", "LRCX",
    "KLAC", "MRVL", "ADI", "NXPI", "SNPS", "CDNS", "FTNT", "PANW", "CRWD", "ZS",
    "NET", "DDOG", "SNOW", "MDB", "PLTR", "PALANTIR", "DELL", "HPE", "IBM",
    # Semis Extended
    "SMCI", "ARM", "ON", "SWKS", "MCHP", "MPWR", "ENTG", "ASML", "TSM",
    # Consumer Tech / Internet
    "NFLX", "SHOP", "SQ", "PYPL", "COIN", "HOOD", "SOFI", "AFRM", "BILL", "TOST",
    "DASH", "UBER", "LYFT", "ABNB", "BKNG", "EXPE", "TRIP", "ETSY", "EBAY", "MELI",
    "SE", "GRAB", "NU", "PINS", "SNAP", "RDDT", "RBLX", "TTWO", "EA", "ATVI",
    # AI / Growth
    "AI", "PATH", "S", "IONQ", "RGTI", "QUBT", "BBAI", "SOUN", "VNET",
    # Biotech / Pharma
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "BMY", "AMGN", "GILD", "REGN",
    "VRTX", "BIIB", "MRNA", "BNTX", "ISRG", "DXCM", "IDXX", "ZTS", "ALGN", "HOLX",
    "ILMN", "EW", "SYK", "MDT", "ABT", "TMO", "DHR", "A", "BDX", "BAX",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "BLK", "ICE", "CME",
    "SPGI", "MCO", "AXP", "V", "MA", "COF", "DFS", "SYF", "ALLY",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "OXY", "DVN",
    "HAL", "FANG", "PXD", "HES", "APA", "EQT", "AR", "RRC",
    # Industrials
    "BA", "CAT", "DE", "HON", "GE", "RTX", "LMT", "NOC", "GD", "TDG",
    "FDX", "UPS", "UNP", "CSX", "NSC", "DAL", "UAL", "AAL", "LUV",
    # Consumer
    "COST", "WMT", "TGT", "HD", "LOW", "NKE", "LULU", "SBUX", "MCD", "CMG",
    "DPZ", "YUM", "QSR", "DRI", "WYNN", "LVS", "MGM", "CZR", "DKNG",
    "PG", "KO", "PEP", "MNST", "CELH", "PM", "MO", "STZ", "DEO", "BUD",
    # Auto / EV
    "GM", "F", "RIVN", "LCID", "NIO", "XPEV", "LI", "RACE",
    # Real Estate / REITs
    "AMT", "PLD", "CCI", "EQIX", "SPG", "O", "DLR", "PSA",
    # Crypto Adjacent
    "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF",
    # China ADRs
    "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI", "BILI", "TME",
    # ETFs (for market context)
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP",
    "GLD", "SLV", "TLT", "HYG", "VXX", "ARKK", "ARKW", "SMH", "SOXX", "XBI",
    "KWEB", "EEM", "FXI", "EWZ",
    # Additional high-volume
    "PARA", "WBD", "CMCSA", "DIS", "T", "VZ", "TMUS",
    "ROKU", "TTD", "ZM", "DOCU", "OKTA", "TWLO", "VEEV", "HUBS", "WDAY",
    "NOW", "TEAM", "ZI", "CFLT", "GTLB", "ESTC", "MNDY", "U",
    "ENPH", "SEDG", "FSLR", "RUN", "NOVA",
    "FCX", "NEM", "GOLD", "AEM",
    "CHWY", "W", "CAVA", "BROS", "SHAK",
    "SMMT", "ASTS", "LUNR", "RKLB",
    "APP", "DUOL", "INTA",
]


def fetch_top_symbols(count: int = 300) -> list[str]:
    """
    Fetch top US stocks by volume from Massive.com API.
    Falls back to hardcoded list if API fails.
    """
    print(f"\n  Fetching top {count} symbols by volume...")

    try:
        api_key = os.getenv("MASSIVE_API_KEY")
        if not api_key:
            raise ValueError("No API key")

        # Try grouped daily endpoint for a recent trading day
        import requests

        # Find most recent weekday
        today = datetime.now()
        check_date = today - timedelta(days=1)
        while check_date.weekday() >= 5:  # Skip weekends
            check_date -= timedelta(days=1)
        date_str = check_date.strftime("%Y-%m-%d")

        print(f"  Querying grouped daily for {date_str}...")
        url = f"https://api.polygon.io/v2/grouped/locale/us/market/stocks/{date_str}"
        resp = requests.get(url, params={"apiKey": api_key}, timeout=30)

        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                # Filter: only regular stocks/ETFs, price > $5, volume > 500k
                filtered = [
                    r for r in results
                    if r.get("v", 0) > 500000
                    and r.get("c", 0) > 5.0
                    and len(r.get("T", "")) <= 5  # Skip long tickers (warrants, etc)
                    and "." not in r.get("T", "")  # Skip preferred shares
                ]

                # Sort by volume descending
                filtered.sort(key=lambda x: x.get("v", 0), reverse=True)
                symbols = [r["T"] for r in filtered[:count]]

                print(f"  ✓ Found {len(symbols)} symbols from API (filtered from {len(results)} total)")
                return symbols
            else:
                print(f"  ✗ API returned empty results, using fallback list")
        else:
            print(f"  ✗ API returned {resp.status_code}, using fallback list")

    except Exception as e:
        print(f"  ✗ API error: {e}, using fallback list")

    # Fallback: deduplicated hardcoded list
    seen = set()
    unique = []
    for s in FALLBACK_SYMBOLS:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    symbols = unique[:count]
    print(f"  Using fallback list: {len(symbols)} symbols")
    return symbols


def save_symbols(symbols: list[str]):
    """Save symbol list to cache."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "symbols": symbols,
        "count": len(symbols),
        "fetched_at": datetime.now().isoformat(),
    }
    CACHE_PATH.write_text(json.dumps(data, indent=2))
    print(f"  Saved {len(symbols)} symbols to {CACHE_PATH}")


def load_symbols() -> list[str]:
    """Load cached symbol list."""
    if not CACHE_PATH.exists():
        return []
    try:
        data = json.loads(CACHE_PATH.read_text())
        return data.get("symbols", [])
    except (json.JSONDecodeError, KeyError):
        return []


def main():
    parser = argparse.ArgumentParser(description="Fetch top US stocks by volume")
    parser.add_argument("--count", type=int, default=300, help="Number of symbols (default: 300)")
    parser.add_argument("--list", action="store_true", help="Just show cached list")
    args = parser.parse_args()

    if args.list:
        symbols = load_symbols()
        if not symbols:
            print("No cached symbols. Run without --list first.")
            return
        print(f"\n  Cached symbols ({len(symbols)}):")
        for i in range(0, len(symbols), 10):
            chunk = symbols[i:i+10]
            print(f"    {', '.join(chunk)}")
        return

    symbols = fetch_top_symbols(args.count)
    save_symbols(symbols)

    print(f"\n  Top 20: {', '.join(symbols[:20])}")
    print(f"  Total: {len(symbols)} symbols ready for backtest")
    print(f"\n  Next: python run_backtest.py --from-cache --days 90")


if __name__ == "__main__":
    main()