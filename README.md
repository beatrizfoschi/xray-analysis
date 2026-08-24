# xray-analysis

Python scripts for X-ray diffraction and emission data analysis, developed at synchrotron facilities.

## Repository Structure

```
xray-analysis/
├── laue/           # Laue microdiffraction: peak search, segmentation, fitting, tracking
│   └── satellite/  # MQW satellite peaks: detection, metrics, period (own tests + docs)
├── emission/       # Emission analysis: NMF decomposition, XEOL spectral fitting, LED statistics
├── utils/          # Shared plotting and statistics helpers
├── pipelines/      # Batch processing and SLURM job scripts
└── data/           # Placeholder for local data; contents are not tracked
```

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/beatrizfoschi/xray-analysis
cd xray-analysis
uv sync                 # add --extra dev to get pytest
```

`uv sync` builds the environment from `uv.lock` and installs the repository
itself, so the documented imports resolve from anywhere — no `sys.path` setup:

```python
from laue.spot_metrics import analyze_spot
from laue.satellite.scan_pipeline import run_satellite_pipeline
```

Run things through `uv run`, which uses the project environment without
activating it:

```bash
uv run pytest
uv run python laue/sapphire_peak_filter.py --help
```

If the clone lives in a cloud-synced folder (OneDrive, Proton Drive, Dropbox),
put the environment somewhere else — a synced `.venv` is slow and gets its files
renamed underneath you:

```bash
export UV_PROJECT_ENVIRONMENT=~/venvs/xray-analysis   # Windows: C:\venvs\xray-analysis
```

`requirements.txt` is generated from `uv.lock`, for environments where uv is not
available (cluster nodes, `pip install -r`). Regenerate it whenever the
dependencies change — never edit it by hand:

```bash
uv export --no-hashes --no-emit-project --output-file requirements.txt
```

Two constraints come from the dependency chain. `lauexplore` pins
`lauetools==3.1.44`, which caps the interpreter at **Python < 3.13**. LaueTools
in turn requires wxPython, which publishes no Linux wheels; since none of the
LaueTools modules used here import `wx`, `pyproject.toml` drops it on Linux so
that cluster installs do not have to build it from source.

## Usage

Scripts are designed to run on synchrotron computing clusters. See individual
script docstrings for usage instructions and required arguments. Modules
carrying a shebang (e.g. `laue/sapphire_peak_filter.py`) also run directly as
scripts.

## Related repositories

- [xeol-viewer](https://github.com/beatrizfoschi/xeol-viewer) — PyQt6 viewer for
  XEOL datasets, split out because it has its own dependencies and release cycle.

## Author

Beatriz Foschiani — [beatrizfoschi](https://github.com/beatrizfoschi)
