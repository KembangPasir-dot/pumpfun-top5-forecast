# pumpfun-top5-forecast

Entry for the make.fun bounty
[pumpfun-top5-forecast-01](https://make.fun/bounty/pumpfun-top5-forecast-01):
forecast which 5 pump.fun coins launched since the opening will be the
largest by market cap 24 hours out, with honest probabilities and
market-cap distributions.

## Method

`main.py` (Python 3.13, standard library only, deterministic) reads
`/input/task.json` and `/input/history.ndjson` (10-minute market-cap marks)
and writes `/output/forecast.json`:

- Per coin, fit the recent log10-cap trend, shrunk hard toward a decaying
  global drift (pump.fun coins mostly fade after their spike; momentum is
  only trusted for the first few hours).
- Monte-Carlo the future over the answer-read horizons: drift + diffusion
  plus a small collapse chance for ungraduated bonding-curve coins.
- Place probabilities come from simulated rank frequencies, scaled down to
  leave room for coins born after the cut (the scorer covers those with the
  unnamed share).
- Size distributions come from the simulated k-th largest caps, mixed with
  a broad log-normal and a fresh-runner component, floored so no bin is
  ever near-zero.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/forecast.json
```

## Backtest (practice rehearsal)

Train on the first 40 minutes of the kit's history, forecast 20 minutes
ahead: rank log-score −21.75 vs the persistence baseline's −24.93
(top-5 agreement 3/5 for both). The size score trails at this short
horizon because the model is calibrated for the real 7–13 hour horizon,
where caps move far more than in 20 minutes.
