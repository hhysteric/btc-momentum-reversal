#!/usr/bin/env python3
"""
JLST Reversal–Momentum grid-search backtest (BTC, daily).

Searches revLen x momLen x momSkip x entryThr x noiseGain, scores each combo
by train-period (hit_rate - 0.5) x Sharpe averaged over 7d/14d/30d forward
horizons, then reports out-of-sample test stats. Writes data/params.json.

Usage:
    python scripts/backtest.py            # full grid
    python scripts/backtest.py --quick    # small sanity grid
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from update_data import (get_price_data, update_funding_cache, HALVING_DATES,
                         HALV_WIN, HALV_BOOST, ATR_LEN, ATR_SLOW_MUL, VOL_AVG_LEN,
                         SHOCK_K1, SHOCK_K2, SHOCK_WIN, SHOCK_ATTEN,
                         CORR_THR, PENALTY, CLAMP_Z, REV_W, MOM_W,
                         NOISE_W, NOISE_W_NOFUND, DEAD_ZONE)

TRAIN_END = pd.Timestamp("2023-12-31", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
HORIZONS = [7, 14, 30]
NORM_LEN = 180
BARS_YEAR = 365

GRID = {
    "revLen": [14, 21, 30, 45, 60],
    "momLen": [90, 180, 270, 330, 365],
    "momSkip": [7, 14, 21, 30],
    "entryThr": [0.5, 0.75, 1.0],
    "noiseGain": [0.3, 0.5],
}
QUICK_GRID = {
    "revLen": [21, 30],
    "momLen": [180, 270],
    "momSkip": [14, 21],
    "entryThr": [0.5, 0.75],
    "noiseGain": [0.3],
}


def rolling_z(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(win).mean()
    sd = s.rolling(win).std(ddof=0)
    return ((s - mu) / sd.replace(0, np.nan)).clip(-CLAMP_Z, CLAMP_Z)


def rolling_corr(a: pd.Series, b: pd.Series, win: int) -> pd.Series:
    return a.rolling(win).corr(b)


def build_context(close: pd.Series, high: pd.Series, low: pd.Series,
                  volume: pd.Series, funding: pd.Series):
    """Param-independent building blocks."""
    ctx = {}
    log_ret = np.log(close / close.shift(1))
    ctx["log_ret"] = log_ret
    ctx["vol_ann"] = log_ret.rolling(NORM_LEN).std(ddof=0) * math.sqrt(BARS_YEAR)

    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()],
                   axis=1).max(axis=1)
    atr_fast = tr.ewm(alpha=1 / ATR_LEN, adjust=False).mean()
    atr_slow = tr.ewm(alpha=1 / (ATR_LEN * ATR_SLOW_MUL), adjust=False).mean()
    atr_pct = atr_fast / close
    avg_vol = volume.rolling(VOL_AVG_LEN).mean()

    nz_atr = rolling_z(atr_pct, NORM_LEN)
    nz_vol = rolling_z(volume / avg_vol, NORM_LEN)
    nz_regime = rolling_z(atr_fast / atr_slow, NORM_LEN)
    nz_fund = rolling_z(funding.abs(), NORM_LEN)
    if nz_fund.notna().any():
        noise = (NOISE_W["atr"] * nz_atr.fillna(0) + NOISE_W["vol"] * nz_vol.fillna(0)
                 + NOISE_W["regime"] * nz_regime.fillna(0)
                 + NOISE_W["funding"] * nz_fund.fillna(0))
        noise[nz_fund.isna()] = (NOISE_W_NOFUND["atr"] * nz_atr.fillna(0)
                                 + NOISE_W_NOFUND["vol"] * nz_vol.fillna(0)
                                 + NOISE_W_NOFUND["regime"] * nz_regime.fillna(0))[nz_fund.isna()]
    else:
        noise = (NOISE_W_NOFUND["atr"] * nz_atr.fillna(0)
                 + NOISE_W_NOFUND["vol"] * nz_vol.fillna(0)
                 + NOISE_W_NOFUND["regime"] * nz_regime.fillna(0))
    ctx["noise_z"] = noise.clip(-3, 3)

    # info shock -> reversal attenuation
    move = log_ret.abs() > SHOCK_K1 * atr_pct
    volx = volume > SHOCK_K2 * avg_vol
    shock = (move & volx).astype(int)
    post = shock.rolling(SHOCK_WIN + 1).max().fillna(0).astype(bool)
    ctx["rev_atten"] = pd.Series(np.where(post, SHOCK_ATTEN, 1.0), index=close.index)

    # halving boost
    days = pd.Series(close.index, index=close.index)
    boost = pd.Series(1.0, index=close.index)
    for hd in HALVING_DATES:
        mask = (days >= hd) & (days < hd + pd.Timedelta(days=HALV_WIN))
        boost[mask] = HALV_BOOST
    ctx["mom_boost"] = boost
    return ctx


def eval_signals(comp: pd.Series, close: pd.Series, thr: float, mask: pd.Series):
    """Threshold-crossing signals -> forward-return stats per horizon."""
    sig = pd.Series(0, index=comp.index)
    cross_up = (comp >= thr) & (comp.shift(1) < thr) & (comp.abs() >= DEAD_ZONE)
    cross_dn = (comp <= -thr) & (comp.shift(1) > -thr) & (comp.abs() >= DEAD_ZONE)
    sig[cross_up] = 1
    sig[cross_dn] = -1
    sig = sig[sig != 0]
    out = {}
    for h in HORIZONS:
        fwd = close.shift(-h) / close - 1.0
        r = (sig * fwd.reindex(sig.index)).dropna()
        r = r[mask.reindex(r.index).fillna(False)]
        if len(r) < 5:
            out[h] = {"n": int(len(r)), "hit": None, "sharpe": None, "avg_ret": None}
            continue
        out[h] = {
            "n": int(len(r)),
            "hit": round(float((r > 0).mean()), 4),
            "avg_ret": round(float(r.mean()), 6),
            "sharpe": round(float(r.mean() / (r.std(ddof=0) + 1e-12)
                                  * math.sqrt(BARS_YEAR / h)), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    grid = QUICK_GRID if args.quick else GRID

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"

    print("fetching daily price data ...")
    d = get_price_data("1d")
    idx = pd.to_datetime(pd.Series(d["ts"], dtype="int64"), unit="ms", utc=True)
    close = pd.Series(d["close"], index=idx, dtype=float)
    high = pd.Series(d["high"], index=idx, dtype=float)
    low = pd.Series(d["low"], index=idx, dtype=float)
    volume = pd.Series(d["volume"], index=idx, dtype=float)

    print("loading funding cache ...")
    funding_daily = update_funding_cache(data_dir / "btc_funding.json")
    funding = pd.Series({pd.Timestamp(k, tz="UTC"): v for k, v in funding_daily.items()},
                        dtype=float).reindex(idx)

    ctx = build_context(close, high, low, volume, funding)
    vol_ann = ctx["vol_ann"].replace(0, np.nan).fillna(1.0)
    log_ret = ctx["log_ret"]

    train_mask = pd.Series((idx <= TRAIN_END).to_numpy(), index=idx)
    test_mask = pd.Series((idx >= TEST_START).to_numpy(), index=idx)

    # Precompute z-scored reversal per revLen and momentum per (momLen, momSkip)
    rev_map = {}
    for rl in grid["revLen"]:
        raw = -np.log(close / close.shift(rl))
        scaled = raw / (vol_ann * math.sqrt(rl / BARS_YEAR))
        rev_map[rl] = (rolling_z(scaled, NORM_LEN) * ctx["rev_atten"]).clip(-CLAMP_Z, CLAMP_Z)
    mom_map = {}
    for ml in grid["momLen"]:
        for sk in grid["momSkip"]:
            raw = np.log(close.shift(sk) / close.shift(sk + ml))
            scaled = raw / (vol_ann * math.sqrt(ml / BARS_YEAR))
            mom_map[(ml, sk)] = (rolling_z(scaled, NORM_LEN) * ctx["mom_boost"]).clip(-CLAMP_Z, CLAMP_Z)

    results = []
    combos = len(grid["revLen"]) * len(grid["momLen"]) * len(grid["momSkip"]) \
        * len(grid["entryThr"]) * len(grid["noiseGain"])
    print(f"grid: {combos} combos | train<= {TRAIN_END:%Y-%m-%d} | test>= {TEST_START:%Y-%m-%d}")

    for rl in grid["revLen"]:
        rev = rev_map[rl]
        for (ml, sk), mom in mom_map.items():
            corr_rm = rolling_corr(rev, mom, NORM_LEN)
            regime_f = pd.Series(np.where(corr_rm > CORR_THR, PENALTY, 1.0), index=idx)
            for ng in grid["noiseGain"]:
                wr = (REV_W * (1 + ng * ctx["noise_z"])).clip(lower=0)
                wm = (MOM_W * (1 - ng * ctx["noise_z"] * 0.5)).clip(lower=0)
                ws = (wr + wm).replace(0, np.nan)
                comp = ((wm / ws) * mom + (wr / ws) * rev) * regime_f
                for thr in grid["entryThr"]:
                    tr_stats = eval_signals(comp, close, thr, train_mask)
                    scored = [s for s in tr_stats.values() if s["hit"] is not None]
                    if not scored:
                        continue
                    score = float(np.mean([(s["hit"] - 0.5) * max(s["sharpe"], 0)
                                           for s in scored]))
                    results.append({
                        "params": {"revLen": rl, "momLen": ml, "momSkip": sk,
                                   "entryThr": thr, "noiseGain": ng},
                        "score": round(score, 6),
                        "train": tr_stats,
                        "comp": comp, "thr": thr,
                    })

    results.sort(key=lambda r: r["score"], reverse=True)
    top = results[:5]

    # Out-of-sample stats for the top combos
    top_out = []
    for r in top:
        te = eval_signals(r["comp"], close, r["thr"], test_mask)
        top_out.append({"params": r["params"], "score": r["score"],
                        "train": r["train"], "test": te})

    best = top_out[0] if top_out else None
    payload = {
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "train_period": f"{idx.iloc[0]:%Y-%m-%d} to {TRAIN_END:%Y-%m-%d}",
        "test_period": f"{TEST_START:%Y-%m-%d} to {idx.iloc[-1]:%Y-%m-%d}",
        "normLen": NORM_LEN,
        "combos": len(results),
        "best_params": {**best["params"], "normLen": NORM_LEN} if best else None,
        "best_train": best["train"] if best else None,
        "best_test": best["test"] if best else None,
        "top5": top_out,
    }
    out = data_dir / "params.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    if best:
        print(f"best: {best['params']} score={best['score']}")
        for h in HORIZONS:
            t, e = best["train"][h], best["test"][h]
            print(f"  {h:>2}d  train hit={t['hit']} sharpe={t['sharpe']} n={t['n']}"
                  f"  |  test hit={e['hit']} sharpe={e['sharpe']} n={e['n']}")


if __name__ == "__main__":
    main()
