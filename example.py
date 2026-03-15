from massive import RESTClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = api_key = os.getenv("MASSIVE_API_KEY")
client = RESTClient(api_key=API_KEY)

TICKER = "QQQ"

today = datetime.today()

# Date ranges
three_months_ago = (today - timedelta(days=90)).strftime("%Y-%m-%d")
one_month_ago    = (today - timedelta(days=30)).strftime("%Y-%m-%d")
three_days_ago   = (today - timedelta(days=5)).strftime("%Y-%m-%d")  # buffer for weekends
today_str        = today.strftime("%Y-%m-%d")

def fetch_bars(ticker, multiplier, timespan, from_date, to_date, label):
    print(f"\n=== {label} ===")
    bars = []
    for bar in client.list_aggs(
        ticker=ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_=from_date,
        to=to_date,
        adjusted=True,
        sort="asc",
        limit=50000
    ):
        bars.append(bar)
    print(f"Fetched {len(bars)} bars")
    for b in bars:
        print(b)
    return bars

if __name__ == "__main__":
    # 3 months of 1-hour bars
    bars_1h = fetch_bars(TICKER, 1, "hour", three_months_ago, today_str, "QQQ - 1H Bars (3 Months)")

    # 1 month of 15-minute bars
    bars_15m = fetch_bars(TICKER, 15, "minute", one_month_ago, today_str, "QQQ - 15Min Bars (1 Month)")

    # Last 3 trading days of 5-minute bars
    #bars_5m = fetch_bars(TICKER, 5, "minute", three_days_ago, today_str, "QQQ - 5Min Bars (3 Trading Days)")