# Publication Figure Skill

This file defines the default figure standard for analysis outputs that may be
used in progress reviews, paper drafts, rebuttals, slides, or appendices.

## Goal

- produce reviewer-defensible figures by default
- keep figures readable at manuscript scale without manual post-processing
- preserve enough metadata to regenerate each figure later

## Default Rule

- when a task produces comparison-ready quantitative results, generate figures
  in publication-ready style unless the user explicitly asks for raw outputs
- do not treat figures as cosmetic; they are part of the scientific artifact

## Figure Bundle

Each substantive figure should emit:

- a vector export: `.pdf` preferred, `.svg` acceptable
- a raster export: `.png` at 600 dpi minimum
- a source-data file: `.csv`
- a figure manifest: `.json` with script, inputs, outputs, timestamp, and git
  commit hash when available
- a short caption draft: `.md`

Default local path:
- `artifacts/figures/<phase>/<figure_id>/`

## Style Rules

- use consistent sizing across comparable figures
- use explicit legends; do not rely on the caption alone to decode colors or
  line styles
- include axis labels and units when applicable
- keep comparable panels on the same axis scale unless a clear exception is
  documented in the caption
- use a colorblind-safe palette and ensure the figure remains readable in
  grayscale
- keep confidence intervals, uncertainty bands, and reference lines visible
- place legends outside the data region when possible
- avoid 3D effects, decorative backgrounds, and chartjunk

## Typography And Geometry

- panel label size: 12 to 14 pt
- axis title size: 9 to 10 pt
- tick label size: 8 to 9 pt
- legend text size: 8 to 9 pt
- line width: at least 1.5 pt
- marker size: at least 5 pt when markers are used

## Project-Level Consistency

- keep semantic groups visually stable across figures:
  - `Gold` / `X*`: black or dark gray
  - `X_rule`: blue
  - model-derived estimates: distinct contrasting colors
  - synthetic or perturbation curves: gray family or clearly differentiated
    dashed styles
- when multiple panels show the same estimand, preserve ordering and color
  mapping

## Validation Before Delivery

- inspect the figure at manuscript-like scale before considering it complete
- verify that no labels, legends, or tick marks are clipped
- verify that lines and error bars remain distinguishable after export
- verify that legends match the plotted order and labels exactly
- verify that the caption states what cohort, metric, and uncertainty are shown

## Do Not

- do not hand-edit final figure values in presentation software
- do not use screenshots as the final figure artifact
- do not ship figures without legends, units, or readable text
- do not let different figures silently redefine colors for the same dataset
