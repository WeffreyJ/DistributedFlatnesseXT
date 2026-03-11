# Figures Tree

This folder is the stable manuscript-facing figure package for the repo.

## Subfolders

- `matched_geometry/`
  - figures for the matched / geometry-aware main-text results
- `unmatchedness/`
  - figures for the Tier-2 unmatchedness and cross-family section
- `protocol/`
  - reserved for operating-envelope / admissibility / vignette figures if they return to the manuscript
- `appendix/`
  - reserved for overflow or supporting figures
- `_staging/`
  - temporary build area if needed; should stay mostly empty

## Built now

The build script generates the figures already needed by the current draft and writes a traceable manifest:

- `figures/manifest.json`

## Reserved for later

`protocol/` and `appendix/` are reserved so Overleaf paths do not churn later if those sections regain figures.

## Overleaf upload

Upload the whole `figures/` folder to Overleaf.
The intended LaTeX usage is:

```latex
\includegraphics[width=0.48\textwidth]{figures/matched_geometry/operator_gap_identity.png}
```

The point of this tree is to keep those paths stable across draft revisions.
