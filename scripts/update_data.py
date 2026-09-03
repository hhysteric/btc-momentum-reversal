#!/usr/bin/env python3
"""
BTC Momentum–Reversal Data Pipeline
===================================
Theory: Jegadeesh, Luo, Subrahmanyam & Titman (2025, RFS)
"Short-Term Reversals and Longer-Term Momentum around the World"

Data sources (in fallback order):
  - Price : Binance spot klines (data-api.binance.vision -> api.binance.com)
            -> CryptoQuant price-ohlcv (needs CRYPTOQUANT_API_KEY env, last 365d)
  - Funding: Binance USD-M futures fundingRate API (fapi.binance.com)
            -> Binance public data dumps (premiumIndexKlines -> funding estimate)

Model mapping (paper -> indicator):
  Table 2 rho1 < 0      -> Rev = -log(C / C[revLen])          (short-term reversal)
  Table 2 rho2 ~= 0     -> dead zone (no signal when |comp| < deadZone)
  Table 2 rho3..12 > 0  -> Mom = log(C[skip] / C[skip+momLen]) (longer-term momentum)
  Proposition 2         -> momSkip (skip the most recent month)
  Prediction a          -> info shock attenuates Rev after large move + volume
  Prediction b          -> corr(Rev, Mom) > corrThr => regime penalty
  Prediction c          -> noise up => wRev up, wMom down
"""

import io
import json
import math
import os
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOL = "BTCUSDT"
SPOT_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]
FAPI_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
VISION_DUMP = "https://data.binance.vision/data/futures/um/{period}/premiumIndexKlines/BTCUSDT/8h"
CQ_BASE = "https://api.cryptoquant.com/v1/btc/market-data/price-ohlcv"
CQ_API_KEY = os.environ.get("CRYPTOQUANT_API_KEY", "")

FUNDING_START = datetime(2019, 9, 10, tzinfo=timezone.utc)  # BTCUSDT perp funding starts

HALVING_DATES = [
    datetime(2012, 11, 28, tzinfo=timezone.utc),
    datetime(2016, 7, 9, tzinfo=timezone.utc),
    datetime(2020, 5, 11, tzinfo=timezone.utc),
    datetime(2024, 4, 20, tzinfo=timezone.utc),
    datetime(2028, 4, 20, tzinfo=timezone.utc),  # estimated
]

# Default model parameters. If data/params.json exists (produced by
# scripts/backtest.py), its "best_params" override the daily defaults and are
# scaled to weekly / monthly.
DEFAULT_PARAMS = {
    "revLen": 30, "momLen": 270, "momSkip": 21,
    "normLen": 180, "entryThr": 0.5, "noiseGain": 0.3,
}

TIMEFRAMES = {
    "daily":   {"interval": "1d", "barsYear": 365, "scale": 1,  "label": "日线 Daily"},
    "weekly":  {"interval": "1w", "barsYear": 52,  "scale": 7,  "label": "周线 Weekly"},
    "monthly": {"interval": "1M", "barsYear": 12,  "scale": 30, "label": "月线 Monthly"},
}

# Fixed model constants
REV_W = 0.50
MOM_W = 0.50
ATR_LEN = 14
ATR_SLOW_MUL = 5
VOL_AVG_LEN = 50
CLAMP_Z = 3.0
CORR_THR = 0.20
PENALTY = 0.60
SHOCK_K1 = 2.5
SHOCK_K2 = 2.0
SHOCK_WIN = 10
SHOCK_ATTEN = 0.40
DEAD_ZONE = 0.25
HALV_WIN = 540
HALV_BOOST = 1.20
CASCADE_K = 3.5
CASCADE_VOL_K = 2.5

# Noise proxy weights: ATR% / volume ratio / vol regime / |funding|
NOISE_W = {"atr": 0.25, "vol": 0.25, "regime": 0.20, "funding": 0.30}
NOISE_W_NOFUND = {"atr": 0.40, "vol": 0.30, "regime": 0.30}

UA = {"User-Agent": "btc-momentum-reversal/1.0"}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get(url: str, timeout: int = 30, headers: dict = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def http_get_json(url: str, timeout: int = 30, headers: dict = None):
    return json.loads(http_get(url, timeout, headers))


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def fetch_spot_klines(interval: str, limit: int = 1000) -> list:
    """Fetch full kline history, paginating backwards. Tries each endpoint."""
    last_err = None
    for base in SPOT_ENDPOINTS:
        try:
            all_data, end_time = [], None
            for _ in range(25):
                url = f"{base}?symbol={SYMBOL}&interval={interval}&limit={limit}"
                if end_time is not None:
                    url += f"&endTime={end_time}"
                data = http_get_json(url, timeout=30)
                if not data:
                    break
                all_data = data + all_data
                end_time = data[0][0] - 1
                if len(data) < limit:
                    break
                time.sleep(0.25)
            if all_data:
                print(f"    spot source OK: {base.split('/')[2]}")
                return all_data
        except Exception as e:
            last_err = e
            print(f"    [warn] {base.split('/')[2]} failed: {e}")
    raise RuntimeError(f"all spot endpoints failed: {last_err}")


def fetch_cq_ohlcv() -> list:
    """CryptoQuant fallback: last 365 daily candles -> kline-like arrays."""
    if not CQ_API_KEY:
        return []
    try:
        d = http_get_json(f"{CQ_BASE}?window=day&limit=365",
                          headers={"Authorization": f"Bearer {CQ_API_KEY}"})
        rows = d.get("result", {}).get("data", [])
        out = []
        for r in sorted(rows, key=lambda x: x["date"]):
            ts = int(datetime.strptime(r["date"], "%Y-%m-%d")
                     .replace(tzinfo=timezone.utc).timestamp() * 1000)
            out.append([ts, str(r["open"]), str(r["high"]), str(r["low"]),
                        str(r["close"]), str(r["volume"])])
        print(f"    CryptoQuant fallback: {len(out)} candles")
        return out
    except Exception as e:
        print(f"    [warn] CryptoQuant fallback failed: {e}")
        return []


def get_price_data(interval: str) -> dict:
    try:
        raw = fetch_spot_klines(interval)
    except RuntimeError:
        raw = fetch_cq_ohlcv() if interval == "1d" else []
    if not raw:
        raise RuntimeError(f"no price data for interval {interval}")
    n = len(raw)
    return {
        "ts": [int(k[0]) for k in raw],
        "open": [float(k[1]) for k in raw],
        "high": [float(k[2]) for k in raw],
        "low": [float(k[3]) for k in raw],
        "close": [float(k[4]) for k in raw],
        "volume": [float(k[5]) for k in raw],
        "n": n,
    }


# ---------------------------------------------------------------------------
# Funding rate
# ---------------------------------------------------------------------------

def fetch_funding_fapi() -> list:
    """Full funding history from USD-M futures API. List of {ts, rate}."""
    out, start = [], int(FUNDING_START.timestamp() * 1000)
    for _ in range(60):
        url = f"{FAPI_FUNDING}?symbol={SYMBOL}&limit=1000&startTime={start}"
        data = http_get_json(url, timeout=15)
        if not data:
            break
        out += [{"ts": int(x["fundingTime"]), "rate": float(x["fundingRate"])} for x in data]
        start = data[-1]["fundingTime"] + 1
        if len(data) < 1000:
            break
        time.sleep(0.25)
    return out


def funding_from_premium(premium: float) -> float:
    """Approximate funding rate from premium index (Binance formula):
    F = P + clamp(I - P, -0.05%, +0.05%), I = 0.01% per 8h."""
    return premium + max(-0.0005, min(0.0005, 0.0001 - premium))


def fetch_funding_dumps(known_dates: set) -> list:
    """Fallback: monthly + daily premiumIndexKlines zips from data.binance.vision.
    Returns list of {ts, rate} (estimated funding, 8h granularity)."""
    now = datetime.now(timezone.utc)
    records = []

    def grab(period: str, stamp: str):
        url = f"{VISION_DUMP.format(period=period)}/BTCUSDT-8h-{stamp}.zip"
        blob = http_get(url, timeout=30)
        zf = zipfile.ZipFile(io.BytesIO(blob))
        name = zf.namelist()[0]
        for line in zf.read(name).decode().splitlines():
            parts = line.split(",")
            if not parts or not parts[0].isdigit():
                continue  # header row
            ts = int(parts[0])
            if ts > 10**14:  # microseconds -> milliseconds
                ts //= 1000
            records.append({"ts": ts, "rate": funding_from_premium(float(parts[4]))})

    # Monthly files for completed months not yet in cache
    y, m = FUNDING_START.year, FUNDING_START.month
    while (y, m) < (now.year, now.month):
        covered = any(d.startswith(f"{y:04d}-{m:02d}") for d in known_dates)
        if not covered:
            try:
                grab("monthly", f"{y:04d}-{m:02d}")
            except Exception as e:
                print(f"    [warn] monthly dump {y}-{m:02d}: {e}")
            time.sleep(0.1)
        m += 1
        if m == 13:
            y, m = y + 1, 1

    # Daily files for the last 40 days (fills current month + dump lag)
    for i in range(40, -1, -1):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        if day in known_dates:
            continue
        try:
            grab("daily", day)
        except Exception:
            pass  # most days simply have no file yet
        time.sleep(0.05)

    return records


def update_funding_cache(cache_path: Path) -> dict:
    """Returns {date_str: avg_daily_funding}. Tries fapi, then dumps."""
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            cache = {}
    print(f"  funding cache: {len(cache)} daily entries")

    fresh = []
    try:
        fresh = fetch_funding_fapi()
        if fresh:
            print(f"  fapi funding: {len(fresh)} records")
    except Exception as e:
        print(f"  [warn] fapi unavailable ({e}); using premium-index dumps")
    if not fresh:
        try:
            fresh = fetch_funding_dumps(set(cache.keys()))
            if fresh:
                print(f"  dump funding(est): {len(fresh)} new records")
        except Exception as e:
            print(f"  [warn] dump fallback failed: {e}")

    if fresh:
        daily = defaultdict(list)
        for r in fresh:
            day = datetime.fromtimestamp(r["ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            daily[day].append(r["rate"])
        for day, rates in daily.items():
            cache[day] = sum(rates) / len(rates)
        cache_path.write_text(json.dumps(cache, separators=(",", ":")))
    return cache


# ---------------------------------------------------------------------------
# Indicator math (O(n) rolling statistics)
# ---------------------------------------------------------------------------

def rolling_mean(arr, window):
    n = len(arr)
    out, s, cnt = [None] * n, 0.0, 0
    for i in range(n):
        if arr[i] is not None:
            s += arr[i]; cnt += 1
        if i >= window:
            old = arr[i - window]
            if old is not None:
                s -= old; cnt -= 1
        if i >= window - 1 and cnt > 0:
            out[i] = s / cnt
    return out


def rolling_std(arr, window):
    n = len(arr)
    out, s, s2, cnt = [None] * n, 0.0, 0.0, 0
    for i in range(n):
        if arr[i] is not None:
            s += arr[i]; s2 += arr[i] * arr[i]; cnt += 1
        if i >= window:
            old = arr[i - window]
            if old is not None:
                s -= old; s2 -= old * old; cnt -= 1
        if i >= window - 1 and cnt >= 2:
            mu = s / cnt
            out[i] = math.sqrt(max(s2 / cnt - mu * mu, 0.0))
    return out


def rolling_corr(x, y, window):
    n = len(x)
    out = [None] * n
    sx = sy = sxy = sx2 = sy2 = 0.0
    cnt = 0
    for i in range(n):
        if x[i] is not None and y[i] is not None:
            sx += x[i]; sy += y[i]; sxy += x[i] * y[i]
            sx2 += x[i] * x[i]; sy2 += y[i] * y[i]; cnt += 1
        if i >= window:
            ox, oy = x[i - window], y[i - window]
            if ox is not None and oy is not None:
                sx -= ox; sy -= oy; sxy -= ox * oy
                sx2 -= ox * ox; sy2 -= oy * oy; cnt -= 1
        if i >= window - 1 and cnt >= 10:
            mx, my = sx / cnt, sy / cnt
            vx, vy = sx2 / cnt - mx * mx, sy2 / cnt - my * my
            if vx > 0 and vy > 0:
                out[i] = (sxy / cnt - mx * my) / math.sqrt(vx * vy)
    return out


def zscore(arr, window):
    mu, sd = rolling_mean(arr, window), rolling_std(arr, window)
    return [(a - m) / s if a is not None and m is not None and s and s > 0 else None
            for a, m, s in zip(arr, mu, sd)]


def clamp(v, lo, hi):
    return None if v is None else max(lo, min(hi, v))


def true_range(h, lo, c):
    n = len(h)
    tr = [h[0] - lo[0]]
    for i in range(1, n):
        tr.append(max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])))
    return tr


def rma(arr, period):
    n = len(arr)
    out = [None] * n
    if n < period:
        return out
    out[period - 1] = sum(arr[:period]) / period
    alpha = 1.0 / period
    for i in range(period, n):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _r(v, digits=4):
    return None if v is None else round(v, digits)


# ---------------------------------------------------------------------------
# JLST composite
# ---------------------------------------------------------------------------

def compute_jlst(data: dict, p: dict, funding_aligned: list) -> dict:
    n = data["n"]
    close, high, low, volume, ts = data["close"], data["high"], data["low"], data["volume"], data["ts"]

    revLen, momLen, momSkip = p["revLen"], p["momLen"], p["momSkip"]
    barsYear, normLen = p["barsYear"], p["normLen"]
    corrLen = p.get("corrLen", normLen)
    noiseGain = p.get("noiseGain", 0.3)
    entryThr = p.get("entryThr", 0.5)
    deadZone = p.get("deadZone", DEAD_ZONE)

    # 1. log returns
    log_ret = [None] + [math.log(close[i] / close[i - 1])
                        if close[i] > 0 and close[i - 1] > 0 else None
                        for i in range(1, n)]

    # 2. raw reversal & momentum
    rev_raw = [None] * n
    for i in range(revLen, n):
        if close[i] > 0 and close[i - revLen] > 0:
            rev_raw[i] = -math.log(close[i] / close[i - revLen])
    mom_raw = [None] * n
    skip_total = momSkip + momLen
    for i in range(skip_total, n):
        if close[i - momSkip] > 0 and close[i - skip_total] > 0:
            mom_raw[i] = math.log(close[i - momSkip] / close[i - skip_total])

    # 3. vol scaling + z-scores
    vol_ann = [v * math.sqrt(barsYear) if v is not None else None
               for v in rolling_std(log_ret, normLen)]
    rev_span, mom_span = math.sqrt(revLen / barsYear), math.sqrt(momLen / barsYear)
    rev_scaled = [r / ((va or 1.0) * rev_span) if r is not None else None
                  for r, va in zip(rev_raw, vol_ann)]
    mom_scaled = [m / ((va or 1.0) * mom_span) if m is not None else None
                  for m, va in zip(mom_raw, vol_ann)]
    rev_s = [clamp(v, -CLAMP_Z, CLAMP_Z) for v in zscore(rev_scaled, normLen)]
    mom_s = [clamp(v, -CLAMP_Z, CLAMP_Z) for v in zscore(mom_scaled, normLen)]

    # 4. noise proxy (Prediction c) with funding component
    tr = true_range(high, low, close)
    atr_fast, atr_slow = rma(tr, ATR_LEN), rma(tr, ATR_LEN * ATR_SLOW_MUL)
    avg_vol = rolling_mean(volume, VOL_AVG_LEN)
    atr_pct = [a / c if a is not None and c > 0 else None for a, c in zip(atr_fast, close)]
    vol_regime = [a / b if a is not None and b and b > 0 else None
                  for a, b in zip(atr_fast, atr_slow)]
    vol_ratio = [v / av if v is not None and av and av > 0 else None
                 for v, av in zip(volume, avg_vol)]
    nz_atr = zscore(atr_pct, normLen)
    nz_vol = zscore(vol_ratio, normLen)
    nz_regime = zscore(vol_regime, normLen)
    nz_funding = zscore([abs(f) if f is not None else None for f in funding_aligned], normLen)
    has_funding = any(f is not None for f in nz_funding)

    noise_z = []
    for i in range(n):
        a = nz_atr[i] or 0.0
        v = nz_vol[i] or 0.0
        r = nz_regime[i] or 0.0
        if has_funding and nz_funding[i] is not None:
            raw = (NOISE_W["atr"] * a + NOISE_W["vol"] * v +
                   NOISE_W["regime"] * r + NOISE_W["funding"] * nz_funding[i])
        else:
            raw = (NOISE_W_NOFUND["atr"] * a + NOISE_W_NOFUND["vol"] * v +
                   NOISE_W_NOFUND["regime"] * r)
        noise_z.append(clamp(raw, -3.0, 3.0))

    # 5. dynamic weights
    w_rev, w_mom = [], []
    for nz in noise_z:
        nz = nz or 0.0
        wr = max(0.0, REV_W * (1.0 + noiseGain * nz))
        wm = max(0.0, MOM_W * (1.0 - noiseGain * nz * 0.5))
        ws = max(wr + wm, 1e-6)
        w_rev.append(wr / ws)
        w_mom.append(wm / ws)

    # 6. info shock (Prediction a)
    has_vol = any(v and v > 0 for v in volume)
    shock_event = [False] * n
    for i in range(1, n):
        lr = abs(log_ret[i]) if log_ret[i] is not None else 0.0
        ap = atr_pct[i] if atr_pct[i] is not None else 999.0
        move_ok = lr > SHOCK_K1 * ap if ap < 999 else False
        vol_ok = (volume[i] or 0.0) > SHOCK_K2 * (avg_vol[i] or 1.0) if has_vol else True
        shock_event[i] = move_ok and vol_ok
    post_shock, last_shock = [False] * n, -10**9
    for i in range(n):
        if shock_event[i]:
            last_shock = i
        post_shock[i] = (i - last_shock) <= SHOCK_WIN
    rev_eff = [r * SHOCK_ATTEN if r is not None and post_shock[i] else r
               for i, r in enumerate(rev_s)]

    # 7. halving-cycle modulation
    post_halv, days_since_halv = [False] * n, [None] * n
    for i, t in enumerate(ts):
        dt = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
        last_h = next((hd for hd in reversed(HALVING_DATES) if dt >= hd), None)
        if last_h:
            days = (dt - last_h).days
            days_since_halv[i] = days
            post_halv[i] = 0 <= days <= HALV_WIN
    mom_eff = [m * (HALV_BOOST if post_halv[i] else 1.0) if m is not None else None
               for i, m in enumerate(mom_s)]

    # 8. cascade detection
    cascade = [False] * n
    for i in range(1, n):
        lr = abs(log_ret[i]) if log_ret[i] is not None else 0.0
        ap = atr_pct[i] if atr_pct[i] is not None else 999.0
        move_ok = lr > CASCADE_K * ap if ap < 999 else False
        vol_ok = (volume[i] or 0.0) > CASCADE_VOL_K * (avg_vol[i] or 1.0) if has_vol else True
        cascade[i] = move_ok and vol_ok

    # 9. regime confidence (Prediction b)
    corr_rm = rolling_corr(rev_s, mom_s, corrLen)
    log_ret_lag = [None] + log_ret[:-1]
    ac1 = rolling_corr(log_ret, log_ret_lag, corrLen)
    regime_f, regime_state = [1.0] * n, ["normal"] * n
    for i, cr in enumerate(corr_rm):
        if cr is not None:
            if cr > CORR_THR:
                regime_f[i] = PENALTY
                regime_state[i] = "resonance"
            elif cr < -CORR_THR:
                regime_state[i] = "complementary"

    # 10. composite + signals
    comp = [None] * n
    for i in range(n):
        if mom_eff[i] is not None and rev_eff[i] is not None:
            comp[i] = (w_mom[i] * mom_eff[i] + w_rev[i] * rev_eff[i]) * regime_f[i]

    long_sig, short_sig = [False] * n, [False] * n
    signal_state = ["watch"] * n
    for i in range(1, n):
        c, cp = comp[i], comp[i - 1]
        if c is None or cp is None:
            continue
        in_dead = abs(c) < deadZone
        if c >= entryThr and cp < entryThr and not in_dead:
            long_sig[i] = True
        if c <= -entryThr and cp > -entryThr and not in_dead:
            short_sig[i] = True
        signal_state[i] = ("bullish" if c >= entryThr else
                           "bearish" if c <= -entryThr else
                           "dead_zone" if in_dead else "watch")

    li = n - 1
    last_funding = next((f for f in reversed(funding_aligned) if f is not None), None)
    summary = {
        "last_update": datetime.fromtimestamp(ts[li] / 1000, tz=timezone.utc).isoformat(),
        "price": close[li],
        "composite": _r(comp[li]),
        "rev": _r(rev_eff[li]), "mom": _r(mom_eff[li]),
        "w_rev": _r(w_rev[li]), "w_mom": _r(w_mom[li]),
        "noise_z": _r(noise_z[li]), "corr_rm": _r(corr_rm[li]), "ac1": _r(ac1[li]),
        "vol_regime": _r(vol_regime[li]),
        "post_shock": post_shock[li], "post_halv": post_halv[li],
        "days_since_halv": days_since_halv[li], "cascade": cascade[li],
        "regime": regime_state[li], "signal_state": signal_state[li],
        "funding_rate": _r(last_funding, 6),
        "has_funding": has_funding,
    }

    start = next((i for i, c in enumerate(comp) if c is not None), 0)

    events = []
    for i in range(max(start, 1), n):
        if long_sig[i] or short_sig[i] or cascade[i]:
            events.append({
                "date": datetime.fromtimestamp(ts[i] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                "type": "long" if long_sig[i] else ("short" if short_sig[i] else "cascade"),
                "price": _r(close[i], 2), "composite": _r(comp[i]),
                "rev": _r(rev_eff[i]), "mom": _r(mom_eff[i]),
            })

    return {
        "params": {**p, "corrThr": CORR_THR, "penalty": PENALTY,
                   "deadZone": deadZone, "hasFunding": has_funding},
        "summary": summary,
        "data": {
            "timestamps": ts[start:],
            "open": [_r(v, 2) for v in data["open"][start:]],
            "high": [_r(v, 2) for v in high[start:]],
            "low": [_r(v, 2) for v in low[start:]],
            "close": [_r(v, 2) for v in close[start:]],
            "volume": [_r(v) for v in volume[start:]],
            "composite": [_r(v) for v in comp[start:]],
            "rev": [_r(v) for v in rev_eff[start:]],
            "mom": [_r(v) for v in mom_eff[start:]],
            "w_rev": [_r(v) for v in w_rev[start:]],
            "w_mom": [_r(v) for v in w_mom[start:]],
            "noise_z": [_r(v) for v in noise_z[start:]],
            "corr_rm": [_r(v) for v in corr_rm[start:]],
            "ac1": [_r(v) for v in ac1[start:]],
            "regime_state": regime_state[start:],
            "post_shock": post_shock[start:],
            "post_halv": post_halv[start:],
            "cascade": cascade[start:],
            "long_sig": long_sig[start:],
            "short_sig": short_sig[start:],
            "signal_state": signal_state[start:],
            "funding_rate": [_r(v, 6) for v in funding_aligned[start:]],
        },
        "signals_history": events[-200:],
    }


def scale_params(base: dict, scale: int, bars_year: int) -> dict:
    return {
        "revLen": max(1, round(base["revLen"] / scale)),
        "momLen": max(2, round(base["momLen"] / scale)),
        "momSkip": max(1, round(base["momSkip"] / scale)),
        "normLen": max(6, round(base["normLen"] / scale)),
        "corrLen": max(6, round(base["normLen"] / scale)),
        "entryThr": base["entryThr"],
        "noiseGain": base["noiseGain"],
        "deadZone": DEAD_ZONE,
        "barsYear": bars_year,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("BTC Momentum-Reversal Data Pipeline (JLST 2025)")
    print(f"UTC: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Optimized params (from backtest) or defaults
    params_path = data_dir / "params.json"
    base = dict(DEFAULT_PARAMS)
    if params_path.exists():
        try:
            best = json.loads(params_path.read_text()).get("best_params")
            if best:
                base.update({k: best[k] for k in DEFAULT_PARAMS if k in best})
                print(f"loaded optimized params from params.json: {base}")
        except json.JSONDecodeError:
            pass

    # Funding rate
    funding_daily = update_funding_cache(data_dir / "btc_funding.json")
    print(f"  funding daily entries: {len(funding_daily)}")

    for tf, cfg in TIMEFRAMES.items():
        print(f"\n>> {cfg['label']}")
        data = get_price_data(cfg["interval"])
        print(f"  candles: {data['n']} "
              f"({datetime.fromtimestamp(data['ts'][0]/1000, tz=timezone.utc):%Y-%m-%d} -> "
              f"{datetime.fromtimestamp(data['ts'][-1]/1000, tz=timezone.utc):%Y-%m-%d})")

        funding_aligned = [
            funding_daily.get(datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d"))
            for t in data["ts"]
        ]

        p = dict(base, barsYear=cfg["barsYear"], corrLen=base["normLen"], deadZone=DEAD_ZONE) \
            if cfg["scale"] == 1 else scale_params(base, cfg["scale"], cfg["barsYear"])
        print(f"  params: revLen={p['revLen']} momLen={p['momLen']} momSkip={p['momSkip']} "
              f"normLen={p['normLen']} entryThr={p['entryThr']} noiseGain={p['noiseGain']}")

        result = compute_jlst(data, p, funding_aligned)
        out = data_dir / f"btc_{tf}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        s = result["summary"]
        print(f"  wrote {out.name} ({out.stat().st_size/1024:.0f} KB)")
        print(f"  price=${s['price']:,.2f}  comp={s['composite']}  rev={s['rev']}  mom={s['mom']}")
        print(f"  state={s['signal_state']}  regime={s['regime']}  "
              f"funding={s['funding_rate']}  signals={len(result['signals_history'])}")

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "Binance BTCUSDT spot + USD-M funding",
        "model": "JLST (2025, RFS) reversal-momentum composite, BTC-adapted",
        "timeframes": list(TIMEFRAMES.keys()),
        "funding_days": len(funding_daily),
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nDone -> {data_dir}")


if __name__ == "__main__":
    main()
