#!/usr/bin/env bash
# get_data.sh - download the research days from Tardis's free first-of-month
# sample (no API key) and turn each into an engine .evt + an OFI csv.
#
# Binance USDS-M futures, BTCUSDT, 1st of each month Apr..Oct 2026,
# only the fixed UTC window [00:00, 02:00) of each day.
#
# Why a window and not the full day: one full day of L2 is 0.4-0.9 GB
# compressed and tens of GB as text. Tardis files are in time order, so we
# stream: curl | gunzip | awk stops reading at 02:00, and curl stops downloading.
# The same window every day keeps days comparable (same time-of-day effects).
#
# Output per day D (all in data/, which git ignores):
#   data/D.evt       engine binary file (32-byte events)
#   data/ofi_D.csv   mdr_ofi output: one row per message, e1..e10, reset_flag
# and results/data_checksums.txt with a sha256 per .evt file.
#
# usage: scripts/get_data.sh            (all 7 days)
#        scripts/get_data.sh 2026-10-01 (one day)
set -euo pipefail

BASE=https://datasets.tardis.dev/v1/binance-futures
WINDOW_HOURS=2
BIN=engine/build                         # engine built from the pinned submodule
DAYS=("$@")
[ ${#DAYS[@]} -eq 0 ] && DAYS=(2026-04-01 2026-05-01 2026-06-01 2026-07-01 2026-08-01 2026-09-01 2026-10-01)

mkdir -p data results
[ -x "$BIN/mdr_convert" ] && [ -x "$BIN/mdr_ofi" ] || { echo "build the engine first: make engine"; exit 1; }

# Stream one dataset for one day, keep the header and rows with
# local_timestamp (column 4, microseconds UTC) before the window end.
window() {   # $1 = dataset, $2 = YYYY/MM/DD, $3 = end_us, $4 = out file
    { curl -sfL "$BASE/$1/$2/BTCUSDT.csv.gz" | gzip -dc |
      awk -F, -v E="$3" 'NR==1 {print; next} $4+0 >= E+0 {exit} {print}'; } > "$4" || true
    # awk exits early on purpose; curl/gzip then die of a broken pipe, so ignore that exit code
    local n; n=$(($(wc -l < "$4") - 1))
    [ "$n" -gt 0 ] || { echo "  ERROR: no rows for $1 $2"; exit 1; }
    echo "  $1: $n rows"
}

for D in "${DAYS[@]}"; do
    TAG=${D//-/}                                         # 2026-10-01 -> 20261001
    START_US=$(( $(date -u -d "$D 00:00:00" +%s) * 1000000 ))
    END_US=$(( START_US + WINDOW_HOURS * 3600 * 1000000 ))
    echo "== $D  window 00:00-0${WINDOW_HOURS}:00 UTC"

    window incremental_book_L2 "${D//-//}" "$END_US" "data/l2_$TAG.csv"
    window trades              "${D//-//}" "$END_US" "data/trades_$TAG.csv"

    "$BIN/mdr_convert" "data/l2_$TAG.csv" "data/trades_$TAG.csv" "data/$TAG.evt"
    rm -f "data/l2_$TAG.csv" "data/trades_$TAG.csv"     # text is huge; the .evt has everything
    "$BIN/mdr_ofi" "data/$TAG.evt" "data/ofi_$TAG.csv" array
done

( cd data && sha256sum ./*.evt ) > results/data_checksums.txt
echo "checksums -> results/data_checksums.txt"
