# Notebook improvement notes

I improved the pasted JWST/MAST notebook by making it more runnable and less placeholder-driven.

## Main issues fixed

- Replaced hard-coded `mast_data/example_i2d.fits` with local/downloaded `*_i2d.fits*` discovery.
- Added a single `CONFIG` cell for target, radius, instruments, product suffixes, detection parameters, and export paths.
- Added safe `auto_download=False` behavior so the user can inspect products before downloading large FITS files.
- Added FITS extension inspection before analysis.
- Added robust handling of missing `ERR`, `DQ`, `AREA`, and variance extensions.
- Added a DQ-mask strategy switch rather than blindly masking all nonzero DQ values.
- Added background subtraction, segmentation, aperture photometry, AB magnitude conversion, QA summaries, and export cells that share consistent variable names.
- Guarded optional sections for SEP, multiband reprojected forced photometry, and external catalog matching.
- Added provenance exports: environment manifest, QA JSON, FITS header provenance, catalogue files, and segmentation FITS.

## Remaining science caveats

- Aperture corrections are still placeholders and should be derived per filter/field.
- Star/galaxy classification is intentionally pedagogical, not publication-grade.
- Multiband colour science should include PSF matching and correlated-noise checks.
- Completeness should be validated with injection/recovery tests before publication-quality claims.
