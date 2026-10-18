.RECIPEPREFIX = >
.PHONY: engine data features figures

# Build the pinned engine (the submodule) so results match its exact commit
engine:
> cmake -S engine -B engine/build -G Ninja -DCMAKE_BUILD_TYPE=Release
> cmake --build engine/build

# Download the 7 research days and produce data/*.evt + data/ofi_*.csv
data: engine
> scripts/get_data.sh

features:
> @echo "TODO: build per-day feature tables from data/ofi_*.csv"

figures:
> @echo "TODO: regenerate all plots"
