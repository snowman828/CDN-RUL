# Reproduction guide

Every command below is run from the repository root. `PY` is your Python
interpreter with `requirements.txt` installed.

## 0. Data

See `data/README.md`. C-MAPSS files go in `data/raw/`, N-CMAPSS DS02-006 as
`data/raw/N-CMAPSS_DS02-006.h5`.

## 1. Tables 1-2 -- accuracy and ablation (five seeds)

```bash
$PY src/run_multiseed.py            # CDN-RUL, seeds 42/2024/7/123/99
$PY src/run_baseline_multiseed.py   # strongest baseline, identical protocol
$PY src/aggregate_ablation.py       # Table 2
```

Baselines must be run under the *same* seeds: a single-run baseline number is not
a fixed constant, and comparing a multi-seed method against one over-attributes
variance to the method (this changed two conclusions during the project).

## 2. Table 3 -- split-conformal coverage and width

```bash
$PY src/conformal_eval.py
```

Studentized split conformal, quantile taken on the validation engines,
`mc_samples = 50`. Writes `results/physdec/conformal_stats.json`.

Note the protocol point the paper makes: two pipelines can store predictive
variance in different conventions (one pre-scaled by a calibration factor, the
other raw plus a factor). A naive PICP comparison between them can invert a
calibration verdict; conformal treatment removes the ambiguity because both
models receive identical treatment.

## 3. Section 4.6 -- engine-level intervals and effective sample size

```bash
$PY src/bootstrap_ds02_engine.py            # all five seeds, ~45 s each
$PY src/bootstrap_ds02_engine.py 42         # one seed
```

Writes `results/physdec/ds02_engine_bootstrap.json`. Reports per-engine coverage,
a cluster bootstrap over engines, a moving-block bootstrap that preserves
within-engine autocorrelation, the integrated autocorrelation time, and the
effective sample size.

## 4. Table 4 and Figure 5 -- group-conditional conformal

```bash
$PY src/conformal_group.py           # -> conformal_group_stats.json
$PY src/conformal_group_compare.py   # reads it; -> conformal_group_compare.json
$PY src/plot_fig5_group_conformal.py
```

`conformal_group.py` must run first: `conformal_group_compare.py` reuses
`conformal_group_stats.json` when it exists. Running only the compare step on a
clean tree makes it recompute those statistics inline instead, which is slower
and leaves the two artifacts with no common provenance.

## 4b. Section 4.6 -- remedy test, placebo control, under-coverage

```bash
$PY src/test_section46_remedy.py         # -> remedy_test.json
$PY src/diagnose_remedy_mechanism.py     # -> remedy_mechanism.json
$PY src/conformal_undercov_diagnosis.py  # -> undercov_diagnosis.json
```

Each takes an optional list of seeds and defaults to all five (42 2024 7 123 99).

`test_section46_remedy.py` reports the shift-weighted recalibration as a NEGATIVE
result: the mean coverage change across seeds is within noise of zero. The
placebo control in `diagnose_remedy_mechanism.py` is what turns "no effect" into
an identity -- matched-coverage interval widths agree to four decimals
(0.2636 = 0.2636), so the null is the absence of a mechanism rather than a
miscalibrated test. `conformal_undercov_diagnosis.py` produces the ratio of
nominal to empirical coverage per subset that Section 4.6 quotes.

## 5. Table 6 -- maintenance decision

```bash
$PY src/maintenance_decision.py
```

Expected-profit model with a closed-form truncated-normal `E[min(RUL, t)]`.
Reports the interval-aware policy's value over the point policy across three
cost regimes (71% / 34% / 31% of the run-to-failure loss); the paper reports the
full range rather than the favourable setting.

## 6. Figures 1-4

```bash
$PY src/make_figures.py
```

Writes vector PDF + 300-dpi PNG into the directory named by `FIG_DIR` in
`src/fig_style.py` if you want them elsewhere).

## Verifying without retraining

Checkpoints are not distributed. The evaluation scripts that do not require
training (`bootstrap_ds02_engine.py` does require checkpoints) can be re-run
against the JSON artifacts in `results/physdec/` to check the tabulated numbers.
