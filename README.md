# xray-data-analysis

Python and shell scripts for X-ray diffraction data analysis developed at the ESRF synchrotron.

## Repository Structure

```
xray-data-analysis/
├── analysis/       # Peak fitting, integration, and quantitative analysis
├── pipelines/      # Batch processing and workflow automation (SLURM jobs)
├── visualization/  # Plotting and pattern visualization
├── utils/          # Shared utility functions
├── notebooks/      # Jupyter notebooks
└── data/           # Small reference files: masks, calibrations
```

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Scripts are designed to run on the ESRF computing cluster. See individual script docstrings for usage instructions and required arguments.

## Author

Beatriz Foschiani — [beatrizfoschi](https://github.com/beatrizfoschi)
