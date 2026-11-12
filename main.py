#!/usr/bin/env python3
"""pumpfun-top5-forecast-01: forecast the top 5 pump.fun coins by market cap.

Reads /input/task.json + /input/history.ndjson (10-minute market-cap marks of
every pump.fun coin launched since t0) and writes /output/forecast.json:
  - probabilities: P(coin is at place k) for k = 1..5
  - top5: the most likely top 5 in order
  - sizes: per place, the distribution of log10(cap in SOL) over 0.05-wide bins

Method (stdlib only, deterministic):
  * per coin, fit log10-cap trend over its recent marks, shrunk toward the
    global median drift (pump.fun coins mostly decay after their spike);
  * Monte-Carlo the future: for each trial draw one of the answer-read
    horizons, simulate every coin's log-cap with drift + diffusion and a
    small collapse ("death") chance for ungraduated bonding-curve coins;
  * place probabilities from simulated rank frequencies, scaled down to
    leave room for coins born after the cut (which the scorer covers with
    the unnamed share);
  * size distributions from the simulated k-th largest caps, mixed with a
    broad log-normal and floored so no bin is ever near-zero.
"""
import json
import math
import os
import random
import sys
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
N_TRIALS = 4000
# Share of each place's mass left unnamed (coins born after the cut, or
# known coins too unlikely to name: the scorer shares it among them).
Q_UNNAMED = [0.55, 0.60, 0.65, 0.70, 0.70]
MIN_NAME_P = 0.003
MAX_NAMED_PER_PLACE = 20
SIZE_FLOOR = 0.0003
SIZE_MIX_BROAD = 0.20
SIZE_BROAD_SD = 0.9
# Newborn runners often take the top places: a share of each place's size
# distribution is a broad log-normal around a typical fresh-runner cap.
SIZE_MIX_NEWBORN = 0.15
SIZE_NEWBORN_SD = 0.6
SIZE_NEWBORN_MEAN = [3.9, 3.75, 3.6, 3.5, 3.4]  # log10 SOL per place
# Collapse ("death") chance per horizon for ungraduated / graduated coins.
P_DEATH_CURVE = 0.12
P_DEATH_POOL = 0.04
DEATH_LO = 0.8
DEATH_HI = 2.0
DRIFT_CLAMP = (-0.25, 0.08)   # log10 cap per hour
# Momentum persists only a few hours; after that coins decay on average.
DRIFT_PERSIST_H = 3.0
MU_LONG = -0.06               # log10 cap per hour, long-run decay
SIGMA_FLOOR = 0.12
SIGMA_CAP = 0.50


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def erfc(x):
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
        t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
        t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return r if x >= 0 else 2.0 - r


def norm_sf(x, mean, sd):
    if x == -math.inf:
        return 1.0
    if x == math.inf:
        return 0.0
    return 0.5 * erfc((x - mean) / (sd * math.sqrt(2.0)))


def logcap_sol(lamports):
    return math.log10(int(lamports)) - 9.0


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def linfit(xs, ys):
    """Least-squares slope/intercept; returns (slope, intercept, resid_sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    inter = my - slope * mx
    if n >= 3:
        ss = sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ys))
        sd = math.sqrt(ss / max(1, n - 2))
    else:
        sd = 0.0
    return slope, inter, sd


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    n_bins = len(edges) + 1

    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise SystemExit("the history holds no ok mark")
    marks.sort(key=lambda m: m["at"])
    t_last = parse_time(marks[-1]["at"])
    t0 = parse_time(marks[0]["at"])

    # Answer-read horizons in hours from the last mark (from task.json).
    horizons = []
    for r in task.get("reads", []):
        h = (parse_time(r) - t_last).total_seconds() / 3600.0
        if h > 0:
            horizons.append(h)
    if not horizons:
        horizons = [10.0]

    # Per-coin series: minutes since t0 -> log10 cap (SOL).
    series = {}
    last_info = {}
    for m in marks:
        t_min = (parse_time(m["at"]) - t0).total_seconds() / 60.0
        for row in m["rows"]:
            mint = row["mint"]
            try:
                lc = logcap_sol(row["market_cap_lamports"])
            except (KeyError, TypeError, ValueError):
                continue
            series.setdefault(mint, []).append((t_min, lc))
            last_info[mint] = row

    last_rows = sorted(marks[-1]["rows"], key=lambda r: r.get("rank", 10 ** 9))
    if len(last_rows) < places:
        # Fall back: union of the last few marks, latest cap per coin.
        seen = {}
        for m in reversed(marks):
            for row in m["rows"]:
                seen.setdefault(row["mint"], row)
        last_rows = sorted(seen.values(),
                           key=lambda r: logcap_sol(r["market_cap_lamports"]),
                           reverse=True)
    universe = [r["mint"] for r in last_rows]
    if len(universe) < places:
        raise SystemExit("fewer than %d coins in the chart" % places)

    t_last_min = (t_last - t0).total_seconds() / 60.0
    # Recent window for drift fits: last 120 minutes of history.
    win_start = t_last_min - 120.0

    # Global drift prior: median recent slope across coins with >=3 points.
    slopes = []
    per_coin = {}
    for mint in universe:
        pts = [(t, lc) for (t, lc) in series.get(mint, []) if t >= win_start]
        if len(pts) < 2:
            pts = series.get(mint, [])[-6:]
        xs = [p[0] / 60.0 for p in pts]  # hours
        ys = [p[1] for p in pts]
        slope, _inter, resid = linfit(xs, ys)
        per_coin[mint] = (slope, resid, len(pts))
        if len(pts) >= 3:
            slopes.append(slope)
    mu_drift = median(slopes) if slopes else -0.05
    # On average pump.fun coins decay; never let the prior be growth.
    mu_drift = min(mu_drift, -0.01)
    mu_drift = max(DRIFT_CLAMP[0], min(DRIFT_CLAMP[1], mu_drift))

    resid_all = [c[1] for c in per_coin.values() if c[1] > 0]
    mu_resid = median(resid_all) if resid_all else 0.2

    coins = []
    for mint in universe:
        row = last_info[mint]
        l_last = logcap_sol(row["market_cap_lamports"])
        slope, resid, npts = per_coin[mint]
        # Shrink the slope hard toward the (decaying) global drift.
        w = min(npts, 6) / (min(npts, 6) + 6.0)
        drift = w * slope + (1.0 - w) * mu_drift
        drift = max(DRIFT_CLAMP[0], min(DRIFT_CLAMP[1], drift))
        sigma = resid if resid > 0 else mu_resid
        sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))
        # Very young coins: slope is unreliable, widen sigma.
        try:
            age_min = (t_last - parse_time(row["created_at"])).total_seconds() / 60.0
        except (KeyError, TypeError, ValueError):
            age_min = 60.0
        if age_min < 30:
            sigma = min(SIGMA_CAP, sigma * 1.3)
        venue = row.get("venue", "curve")
        complete = bool(row.get("complete"))
        p_death = P_DEATH_POOL if (venue == "pool" or complete) else P_DEATH_CURVE
        coins.append({
            "mint": mint, "l_last": l_last, "drift": drift, "sigma": sigma,
            "p_death": p_death,
        })

    rng = random.Random(SEED)
    place_counts = [[0] * len(coins) for _ in range(places)]
    place_caps = [[] for _ in range(places)]
    rank_sum = [0.0] * len(coins)
    in_top5 = [0] * len(coins)
    n = len(coins)

    for _ in range(N_TRIALS):
        h = horizons[rng.randrange(len(horizons))]
        sq = math.sqrt(h)
        sims = []
        for idx, c in enumerate(coins):
            if rng.random() < c["p_death"]:
                l = rng.uniform(DEATH_LO, DEATH_HI)
            else:
                # Momentum for the first few hours, then long-run decay.
                l = (c["l_last"] + c["drift"] * min(h, DRIFT_PERSIST_H)
                     + MU_LONG * max(0.0, h - DRIFT_PERSIST_H)
                     + rng.gauss(0.0, c["sigma"] * sq))
            l = max(0.5, min(5.5, l))
            sims.append((l, idx))
        sims.sort(key=lambda t: (-t[0], coins[t[1]]["mint"]))
        for k in range(places):
            idx = sims[k][1]
            place_counts[k][idx] += 1
            place_caps[k].append(sims[k][0])
            if k == 0:
                pass
        for rank, (_l, idx) in enumerate(sims[:places]):
            rank_sum[idx] += rank
            in_top5[idx] += 1
        # ranks beyond top 5 still count for mean rank of top-5 hopefuls only;
        # coins never in top 5 keep rank_sum 0 -> handled by in_top5 below.

    # Mean rank only over trials where the coin made top 5; others rank last.
    mean_rank = []
    for idx in range(n):
        if in_top5[idx]:
            mean_rank.append(rank_sum[idx] / in_top5[idx] + (1 - in_top5[idx] / N_TRIALS) * places)
        else:
            mean_rank.append(float(places))

    # ---- probabilities ----
    probabilities = []
    for k in range(places):
        q = Q_UNNAMED[k] if k < len(Q_UNNAMED) else 0.65
        scored = []
        for idx, c in enumerate(coins):
            p = (place_counts[k][idx] / N_TRIALS) * (1.0 - q)
            if p >= MIN_NAME_P:
                scored.append((p, c["mint"]))
        scored.sort(key=lambda t: (-t[0], t[1]))
        for p, mint in scored[:MAX_NAMED_PER_PLACE]:
            p6 = math.floor(p * 1_000_000) / 1_000_000
            if p6 > 0:
                probabilities.append({"mint": mint, "place": k + 1, "p": p6})

    # ---- top5: lowest mean rank, tie-break by top-5 frequency then last cap ----
    order = sorted(range(n), key=lambda i: (mean_rank[i], -in_top5[i],
                                            -coins[i]["l_last"], coins[i]["mint"]))
    top5 = [coins[i]["mint"] for i in order[:places]]

    # ---- sizes: simulated k-th largest caps, mixed with a broad log-normal ----
    sizes = []
    for k in range(places):
        caps = sorted(place_caps[k])
        med = caps[len(caps) // 2]
        hist = [0.0] * n_bins
        for v in caps:
            b = 0
            while b < len(edges) and v >= edges[b]:
                b += 1
            hist[b] += 1.0
        hist = [h / N_TRIALS for h in hist]
        broad = []
        for b in range(n_bins):
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == n_bins - 1 else edges[b]
            broad.append(norm_sf(lo, med, SIZE_BROAD_SD) - norm_sf(hi, med, SIZE_BROAD_SD))
        tot = sum(broad)
        broad = [x / tot for x in broad]
        nb_mean = SIZE_NEWBORN_MEAN[k] if k < len(SIZE_NEWBORN_MEAN) else 3.4
        newborn = []
        for b in range(n_bins):
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == n_bins - 1 else edges[b]
            newborn.append(norm_sf(lo, nb_mean, SIZE_NEWBORN_SD)
                           - norm_sf(hi, nb_mean, SIZE_NEWBORN_SD))
        tot = sum(newborn)
        newborn = [x / tot for x in newborn]
        w_hist = 1.0 - SIZE_MIX_BROAD - SIZE_MIX_NEWBORN
        mixed = [w_hist * hist[b] + SIZE_MIX_BROAD * broad[b]
                 + SIZE_MIX_NEWBORN * newborn[b] for b in range(n_bins)]
        mixed = [max(SIZE_FLOOR, p) for p in mixed]
        tot = sum(mixed)
        mixed = [p / tot for p in mixed]
        for b in range(n_bins):
            p6 = math.floor(mixed[b] * 1_000_000) / 1_000_000
            if p6 > 0:
                sizes.append({"place": k + 1, "bin": b, "p": p6})

    out = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
