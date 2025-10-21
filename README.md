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
`ProcessPoolExecutor`. If you only need a single trajectory, edit the `seeds`,
`iterations`, and `workers` variables in the `__main__` block at the bottom of
the script before running it. (For quick checks we typically set
`seeds = [0]`, `iterations = 30`, `workers = 1`.)

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

## Sanitising the design space for sharing

The raw `design_space.xlsx` file contains Thermo-Calc derived fields. Before
distributing it externally, run:

```bash
python sanitize_design_space.py --input design_space.xlsx --output design_space_sanitized.csv
```

This will:

1. Rename elemental columns to anonymous placeholders (`element_01`, …).
2. Min–max scale every column whose name starts with `PROP` or `EQUIL` into the
   `[0, 1]` range.

The resulting CSV preserves the structure expected by the campaign scripts
while obscuring direct Thermo-Calc outputs.

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
