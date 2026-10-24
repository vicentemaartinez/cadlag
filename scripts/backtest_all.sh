#!/bin/sh
# Backtest de las cinco grandes ligas europeas (2016-17 a 2025-26).
for c in eng.1 esp.1 ita.1 ger.1 fra.1; do
  python3 scripts/backtest.py $c 2016 4000 > outputs/backtest/$c.log 2>&1
done
