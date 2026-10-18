.RECIPEPREFIX = >
.PHONY: engine data features figures

# Build the pinned engine (the submodule) so results match its exact commit
engine:
> cmake -S engine -B engine/build -G Ninja -DCMAKE_BUILD_TYPE=Release
> cmake --build engine/build

# Download the 7 research days and produce data/*.evt + data/ofi_*.csv
data: engine
> scripts/get_data.sh

# Clean per-day parquet tables + results/day_summary.csv
features:
> .venv/bin/python scripts/build_features.py

# Redraw every README plot from the committed results/*.csv
figures:
> .venv/bin/python scripts/figures.py
