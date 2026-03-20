"""news/ — Multi-source news aggregation pipeline."""
from backend.news.pipeline import (
    fetch_news, fetch_market_news, fetch_news_batch,
    format_headlines_for_llm, NewsItem,
)