import os
import time
import logging
from pathlib import Path
from typing import List
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import requests_cache
from dotenv import load_dotenv

# Load environment variables
from pathlib import Path

# Look for .env in the root trading directory
root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("data_ingestion")

# Setup request caching (caches identical API calls for 1 hour to prevent redundant requests)
requests_cache.install_cache("polygon_cache", expire_after=3600)

_DEFAULT_MARKET_DATA = Path(__file__).resolve().parent.parent / "market_data"
MARKET_DATA_DIR = Path(os.environ.get("MARKET_DATA_DIR", str(_DEFAULT_MARKET_DATA)))
DAILY_DIR = MARKET_DATA_DIR / "daily"
HOURLY_DIR = MARKET_DATA_DIR / "hourly"

DAILY_DIR.mkdir(parents=True, exist_ok=True)
HOURLY_DIR.mkdir(parents=True, exist_ok=True)

import sys
_TRADING_ROOT = Path(__file__).resolve().parent.parent
if str(_TRADING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRADING_ROOT))

from mean_reversion.config import MeanReversionConfig
from trend_following.config import TrendFollowingConfig
from volatility_breakout.config import VolatilityBreakoutConfig

mr_tickers = list(MeanReversionConfig().tickers)
tf_tickers = list(TrendFollowingConfig().tickers)
vb_tickers = list(VolatilityBreakoutConfig().tickers)

# Include RL tickers (any ticker with a trained model)
_models_dir = _TRADING_ROOT / "models"
rl_tickers = [
    f.stem.replace("_ppo", "")
    for f in sorted(_models_dir.glob("*_ppo.zip"))
] if _models_dir.exists() else []

# Combine all strategy tickers
TICKERS = sorted(set(mr_tickers + tf_tickers + vb_tickers + rl_tickers))

def fetch_yfinance_aggs(ticker: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fallback: fetch OHLCV data via yfinance when Polygon key is unavailable."""
    try:
        import yfinance as yf
        yf_interval = "1d" if interval == "day" else "1h"
        df = yf.download(ticker, start=start_date, end=end_date,
                         interval=yf_interval, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = [c.lower() for c in df.columns]
        if "adj close" in df.columns:
            df = df.rename(columns={"adj close": "close"})
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[cols]
    except Exception as e:
        logger.warning(f"yfinance fallback failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_polygon_aggs(ticker: str, multiplier: int, timespan: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch aggregates from Polygon.io."""
    if not POLYGON_API_KEY or POLYGON_API_KEY == "your_polygon_api_key_here":
        logger.warning("POLYGON_API_KEY not set or invalid. Skipping fetch.")
        return pd.DataFrame()

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY
    }
    
    all_results = []
    
    max_retries = 3
    retry_count = 0
    
    while url:
        try:
            resp = requests.get(url, params=params)
            if resp.status_code == 429:
                logger.warning("Rate limit hit, backing off for 10 seconds...")
                time.sleep(10)
                continue
            elif resp.status_code >= 500:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"Max retries reached for {url}. Server error {resp.status_code}.")
                    resp.raise_for_status()
                
                backoff = 2 ** retry_count
                logger.warning(f"Server error {resp.status_code} for {ticker}, retrying in {backoff} seconds...")
                time.sleep(backoff)
                continue
            
            # Reset retry count on success
            retry_count = 0
            
            resp.raise_for_status()
            data = resp.json()
            
            if "results" in data:
                all_results.extend(data["results"])
                
            if "next_url" in data:
                url = data["next_url"]
                params = {"apiKey": POLYGON_API_KEY} # next_url already contains other params
            else:
                break
                
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            break
            
    if not all_results:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_results)
    # Polygon returns timestamp in milliseconds
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "vw": "vwap"
    })
    
    # We drop 't' and 'n' (number of transactions)
    cols = ["date", "open", "high", "low", "close", "volume", "vwap"]
    df = df[cols].set_index("date").sort_index()
    return df

def update_ticker_data(ticker: str, interval: str, force_full: bool = False):
    """Update parquet cache incrementally for a given ticker and interval.

    Args:
        ticker: Ticker symbol
        interval: '1d' or '1h'
        force_full: If True, re-download full history ignoring existing cache
    """
    if interval == "1d":
        target_dir = DAILY_DIR
        multiplier, timespan = 1, "day"
        # 5 years max history for daily start
        default_start = (datetime.now(timezone.utc) - timedelta(days=5*365)).strftime("%Y-%m-%d")
    elif interval == "1h":
        target_dir = HOURLY_DIR
        multiplier, timespan = 1, "hour"
        # 5 years max history for hourly start (supported by Polygon Starter plan)
        default_start = (datetime.now(timezone.utc) - timedelta(days=5*365)).strftime("%Y-%m-%d")
    else:
        raise ValueError(f"Unsupported interval: {interval}")
        
    parquet_file = target_dir / f"{ticker}.parquet"
    
    if parquet_file.exists() and not force_full:
        existing_df = pd.read_parquet(parquet_file)
        if not existing_df.empty:
            last_date = existing_df.index[-1]
            # Start from the last cached date to pick up any new bars
            start_date = last_date.strftime("%Y-%m-%d")
            logger.debug(f"[{ticker} - {interval}] Incremental from {start_date} (have {len(existing_df)} rows)")
        else:
            start_date = default_start
            existing_df = pd.DataFrame()
    else:
        start_date = default_start
        existing_df = pd.DataFrame()
        
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    logger.info(f"[{ticker} - {interval}] Fetching from {start_date} to {end_date}...")
    new_df = pd.DataFrame()
    
    if POLYGON_API_KEY and POLYGON_API_KEY != "your_polygon_api_key_here":
        new_df = fetch_polygon_aggs(ticker, multiplier, timespan, start_date, end_date)
        
    if new_df.empty:
        logger.warning(f"[{ticker} - {interval}] Polygon returned no data (or key not set) — using yfinance fallback")
        new_df = fetch_yfinance_aggs(ticker, timespan, start_date, end_date)
    
    if not new_df.empty:
        # --- Data Validation Layer ---
        # 1. Drop rows where High < Low
        invalid_mask = new_df["high"] < new_df["low"]
        if invalid_mask.any():
            num_invalid = invalid_mask.sum()
            logger.warning(f"[{ticker} - {interval}] Dropping {num_invalid} rows where High < Low")
            new_df = new_df[~invalid_mask]
            
        # 2. Check for missing Close prices
        if new_df["close"].isna().any():
            num_missing = new_df["close"].isna().sum()
            logger.warning(f"[{ticker} - {interval}] Forward-filling {num_missing} missing Close prices")
            new_df["close"] = new_df["close"].ffill()

        if not existing_df.empty:
            # Combine and deduplicate
            combined_df = pd.concat([existing_df, new_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
        else:
            combined_df = new_df
            
        # Sort and write
        combined_df = combined_df.sort_index()
        combined_df.to_parquet(parquet_file)
        logger.info(f"[{ticker} - {interval}] Saved {len(combined_df)} total rows.")
        
        # Write freshness sidecar
        freshness_file = target_dir / f"{ticker}.freshness"
        with open(freshness_file, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())
            
    else:
        logger.info(f"[{ticker} - {interval}] No new data found.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Centralized Data Ingestion Pipeline")
    parser.add_argument("--interval", choices=["1d", "1h", "both"], default="both",
                        help="Interval to update (default: both)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Override ticker list (space-separated, e.g. --tickers AAPL MSFT)")
    parser.add_argument("--force-full", action="store_true",
                        help="Re-download full history instead of incremental update")
    args = parser.parse_args()

    tickers_to_run = args.tickers if args.tickers else TICKERS
    intervals_to_run = ["1d", "1h"] if args.interval == "both" else [args.interval]

    logger.info(f"Tickers: {len(tickers_to_run)}  Intervals: {intervals_to_run}  Force: {args.force_full}")

    for interval in intervals_to_run:
        logger.info(f"=== Starting pipeline for interval: {interval} ===")
        for i, ticker in enumerate(tickers_to_run, 1):
            logger.info(f"[{i}/{len(tickers_to_run)}] {ticker}")
            update_ticker_data(ticker, interval, force_full=args.force_full)

    logger.info("=== Data Ingestion Complete ===")

if __name__ == "__main__":
    main()
