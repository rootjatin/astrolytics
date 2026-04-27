# JWST / MAST Notebook Guide

This package includes a detailed Jupyter notebook for public JWST analytics from MAST.

## Files
- `jwst_mast_analytics_report_notebook.ipynb` — the notebook
- notebook-generated outputs will appear in:
  - `mast_downloads/`
  - `figures/`
  - `output/`

## Main capabilities
- query public JWST observations from MAST
- inspect observation metadata
- inspect product metadata
- download a curated FITS subset
- identify usable 2D science HDUs
- compute background/noise statistics
- measure bright-source centroid and aperture photometry
- compute radial profiles
- inspect detector row/column structure
- run source segmentation
- compare filters
- align images with WCS
- export CSV / JSON / TXT reports

## Good first targets
- SMACS 0723
- NGC 346
- M16
- Carina
- NGC 628

## Typical workflow
1. open the notebook
2. set `MAST_TARGET`
3. run cells top to bottom
4. inspect plots and tables
5. use exported CSV/JSON/TXT outputs for your final report

## Important note
This notebook is an analysis framework for public data exploration.  
It is useful for rapid astronomy analytics and reporting, but it is not a substitute for a full instrument-team science reduction workflow.
