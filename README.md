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

## Usage

Scripts are designed to run on synchrotron computing clusters. See individual script docstrings for usage instructions and required arguments.

`laue`, `emission`, `utils` and `pipelines` are packages, so the documented
imports (`from laue.spot_metrics import analyze_spot`) require the repository
root on `sys.path`:

```python
import sys; sys.path.insert(0, '/path/to/xray-analysis')
```

Modules carrying a shebang (e.g. `laue/sapphire_peak_filter.py`) also run
directly as scripts, in which case no path setup is needed.

## Related repositories

- [xeol-viewer](https://github.com/beatrizfoschi/xeol-viewer) — PyQt6 viewer for
  XEOL datasets, split out because it has its own dependencies and release cycle.

## Author

Beatriz Foschiani — [beatrizfoschi](https://github.com/beatrizfoschi)
