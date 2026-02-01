# Constraint Satisfaction Active Learning

This repository contains the campaign scripts we used to study constrained
multi-objective optimisation for refractory high-entropy alloys. The workflow
is dataset driven: a single CSV describing the alloy design space is ingested,
Gaussian-process models are trained for every objective and constraint, and the
scripts repeatedly select the next alloy to evaluate while logging metrics and
plots.

## Repository layout

- `campaign_const_w_priors.py` – feasibility-first active learning where each
  objective is modelled as a residual around a prior value. This is useful when
  we trust physics-based estimates (e.g. Thermo-Calc) but still want the
  surrogate to learn deviations.
- `campaign_const_NO_priors.py` – the same feasibility-first policy without the
  informative priors (pure data-driven residuals are set to zero).
- `campaign_pehvi.py` – probability of expected hypervolume improvement (pEHVI)
  guided search. The acquisition combines pEHVI in the maximisation space with
  a probability of satisfying the BCC constraint.
- `constraint-satisfaction.yml` – a lightweight conda environment specification
  (Python 3.11, scikit-learn, SciPy stack).
- `LICENSE` – MIT licence terms.

All three campaign scripts share the same helper functions, data layout, and
plotting utilities. Running any of them will create per-seed output folders
containing CSV logs and Matplotlib figures (affine simplex plots, predictive
maps, feasibility probabilities, etc.).

## Data requirements

The scripts look for `design_space.xlsx` (preferred) or `design_space.csv` in
this directory. The chosen file must supply:

- Objective/prior columns: `YS 600 C PRIOR`, `YS 25C PRIOR`,
  `PROP 25C Density (g/cm3)`, `Density Avg`, `Pugh_Ratio_PRIOR`,
  `PROP ST (K)`, `Tm Avg`.
- Phase and chemistry columns:
  - All columns containing both “600C” and “BCC” (their sum is used to build
    the single-phase indicator).
  - Either `VEC` or `VEC Avg` (for the BCC prior flag).
  - Elemental fractions for `Nb`, `Mo`, `Ta`, `V`, `W`, and `Cr`. They can
    appear anywhere in the file; the script validates their presence.

Any additional descriptors are ignored. Rows with missing values in the
required fields are dropped before the campaign starts.

## Quick start

```bash
# 1. Create the environment
conda env create -f constraint-satisfaction.yml
conda activate constraint-satisfaction

# 2. Place / verify design_space.xlsx (or .csv) in this directory

# 3. Launch a campaign (e.g. constraints-first with priors)
python campaign_const_w_priors.py
```

By default each script launches many seeds in parallel using
`ProcessPoolExecutor`. You can override this with the CLI options below.

## CLI usage

All `campaign_*.py` scripts accept the same CLI options:

```bash
# Single seed, shorter run
python campaign_const_w_priors.py --seeds 0 --iterations 30 --workers 1

# Multiple seeds (comma list and ranges both supported)
python campaign_const_NO_priors.py --seeds 1,3,5-8 --iterations 50 --workers 6

# Explicit data path and output locations
python campaign_pehvi.py --data-path /path/to/design_space.csv \
  --results-dir results_opt --plots-dir plots
```

Common flags:
- `--seeds` supports comma-separated values and ranges (e.g. `1,2,5-10`).
- `--iterations`, `--workers` control campaign length and parallelism.
- `--data-path` overrides the default `design_space.xlsx`/`.csv` lookup.
- `--results-dir`, `--plots-dir` change output locations.
- `--density-thresh`, `--ys-thresh`, `--pugh-thresh`, `--st-thresh`, `--vec-thresh`
  override feasibility thresholds (otherwise auto-detected for scaled vs raw data).

`campaign_pehvi.py` adds:
- `--fixed-range-scope {ALL,BCC_ONLY}` to control fixed-range HV scaling
  (default `ALL`, affects reported HV only).

Original defaults (before CLI overrides):
- `--seeds`: `1-200`
- `--iterations`: `100`
- `--workers`: `25`
- `--data-path`: auto-detect `design_space.xlsx` or `design_space.csv` in the repo
- `--results-dir`: `results_const_prior` / `results_const_no_prior` / `results_opt`
- `--plots-dir`: `plots_seed_###` subfolders in the current working directory
- Raw-data thresholds: solidus > 2473 K, density < 9.0 g cm⁻³, yield strength > 700 MPa,
  Pugh ratio > 2.5, VEC >= 6.87
- Scaled-data thresholds (auto-detected): solidus > 0.3340611, density < 0.2189121,
  yield strength > 0.2732669, Pugh ratio > 0.3420824, VEC >= 1.0

## Outputs

Each seed writes a CSV under the corresponding `results_*` directory containing
per-iteration metrics:

- predicted means/standard deviations,
- feasibility probabilities (per constraint and joint),
- dominated hypervolume (raw and fixed-range scaled),
- acquisition values and Pareto status flags.

Plots are saved under `plots_seed_###/` and include:

- the affine simplex progress plot (points coloured by constraint probability),
- per-objective prediction heatmaps (ST, density, YS, Pugh, p_BCC),
- iteration summaries showing which alloys were measured or rejected.

## Choosing a campaign

| Script                        | Key idea                                                      | Outputs directory        |
| ----------------------------- | ------------------------------------------------------------- | ------------------------ |
| `campaign_const_w_priors.py`  | Feasibility-first selection using informative priors          | `results_const_prior`    |
| `campaign_const_NO_priors.py` | Same as above but without priors (pure GP learning)           | `results_const_no_prior` |
| `campaign_pehvi.py`           | Optimisation-first search via closed-form pEHVI × p_bcc       | `results_opt`            |

All configurations share the same feasibility thresholds:
solidus > 2473 K, density < 9.0 g cm⁻³, yield strength > 700 MPa,
Pugh ratio > 2.5, and single-phase BCC at 600 °C.

## Sanitised design space

For distribution we ship `design_space_sanitized.csv`, which mirrors the schema
expected by the campaign scripts while obscuring Thermo-Calc outputs. The
sanitiser performs the following transformations:

1. **Column pruning.** Only the fields the code actually touches are retained:
   Nb/Mo/Ta/V/W/Cr fractions, the prior/objective columns listed in the scripts,
   and the 600 °C BCC phase-fraction columns used to build the single-phase flag.
2. **Element obfuscation.** The six elemental fractions are renamed to
   `element_01` … `element_06`.
3. **Scaling.** Every column that starts with `PROP` or `EQUIL`, as well as the
   prior columns (e.g. `YS 600 C PRIOR`, `PROP ST (K)`, `VEC Avg`), is min–max
   scaled into `[0, 1]`.
4. **Noise injection.** A ±1 % multiplicative jitter (clipped to sensible bounds)
   is applied to all numeric columns to mask the exact Thermo-Calc outputs.
5. **Threshold handling.** The campaign scripts automatically detect whether the
   scaled dataset is in use and swap to the corresponding scaled thresholds for
   density, yield strength, Pugh ratio, solidus temperature, and VEC.

If you have access to the original `design_space.xlsx` you can regenerate the
sanitised file locally, but end users only need the CSV shipped here.

## Tips

- The scripts cap BLAS/OpenMP threads to 1 for reproducibility. Remove the
  `os.environ.setdefault("..._NUM_THREADS", "1")` lines inside `_run_seed`
  if you prefer multi-threaded linear algebra.
- To resume or append to an existing campaign, keep the existing results
  folder – the scripts will append to the per-seed CSV if it already exists.
- For large design spaces consider pre-filtering to BCC-feasible rows to reduce
  the cost of GP training in the early iterations.

Feel free to adapt the scripts for new alloys or alternative constraint sets –
most of the logic is localised inside `prepare_dataframe` and `build_models`.
Pull requests are welcome!
