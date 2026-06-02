# S2S Experiments — Reproducible Simulation

Companion code for: *Semantic Signal–Assisted Decision Support for Reverse
Logistics Resource Recovery Under Uncertainty* (IEEE Access 2026).

## Design Decisions (settled before implementation)

### 1. How is "true yield" revealed?

Two-stage loop in `pipeline.py`:
1. **Predict**: extractor produces φ(m), σ(m) from text
2. **Decide**: engine chooses inspection depth + disposition using Beta(α', β')
3. **Realize**: actual component yields are drawn from ground-truth Beta(α_true, β_true)
4. **Measure**: TRV computed from realized yields minus costs incurred

The system is judged on realized TRV, not expected TRV.

### 2. What does opt-only baseline do exactly?

`opt-only` sets φ=1.0 (no yield adjustment) AND inspects every asset fully.
This is **identical** to `configs/ablation/remove_both.yaml`. Both use the
same code path (`pipeline.py` with `extractor=NullExtractor, policy=InspectAll`).

### 3. What controls ±std in Table IV?

Each seed controls three things via a single `np.random.default_rng(seed)`:
- **Batch composition**: which assets are generated (asset types, ages, conditions)
- **Per-asset yield draws**: realized Beta samples at decision time
- **Demand realization**: downstream demand fulfilled

Set once at the top of `pipeline.py`. All downstream RNG is derived from
child generators split from this master RNG. Documented here and enforced
in `tests/test_smoke.py::TestSeedControl`.

### 4. Decoupled note generation

A key integrity property: in all three scenarios, the technician note / review
text is generated from a **noisy observation** of the ground-truth condition
(via `src/data_generators/noise.py`), not from the condition itself. This
decouples text from yield so that extractor↔yield correlations reflect genuine
signal recovery. See `noise.observed_condition()` for the omission / mislabel
noise model. The decoupling parameters (p_omit=0.15, p_mislabel=0.25) can be
overridden per scenario via `config["note_noise"]`.

### 5. XGBoost baseline

`StructuredOnlyExtractor` learns an `age_bracket → mean(true_yield)` mapping on
a held-out 2000-asset training population (seed 99999, independent of all
evaluation seeds) and applies it via the asset's `age_bracket` field. Text is
ignored. Because the synthetic generators produce age-independent yields, the
trained mapping is nearly flat (~0.55 for all brackets), correctly representing
what a structured-feature model without text can achieve on this benchmark.

## Quick Start

```bash
pip install -r requirements.txt

# Smoke test (10 assets, <1 second)
python -m pytest tests/ -v

# Full experiment — 30 seeds, all baselines (~8 min total)
python scripts/run_experiment.py --config configs/s1_it.yaml    --seeds 0,1,...,29
python scripts/run_experiment.py --config configs/s2_aviation.yaml --seeds 0,1,...,29
python scripts/run_experiment.py --config configs/s3_consumer.yaml --seeds 0,1,...,29

# Canonical Table IV results (mean ± std + Wilcoxon)
python scripts/run_summary.py --seeds 0-29

# Extractor calibration + Table V bracket
python scripts/run_calibration.py --seeds 0-29

# Table VI kappa sensitivity
python scripts/run_kappa_sweep.py --seeds 0-29

# Table IV/ablation CSVs
python scripts/make_tables.py

# Capacity figure
python scripts/make_figures.py
```

## Structure

```
configs/               scenario YAML configs (s1_it, s2_aviation, s3_consumer)
src/s2s/               core framework
  beta_model.py          Beta prior update (S2S update equation)
  inspection_policy.py   3-tier adaptive inspection
  decision_engine.py     greedy MC allocator + all disposition baselines
  pipeline.py            single-seed end-to-end run
  metrics.py             TRV / DCR / ICS / TPR
  extractors/
    base.py              AbstractExtractor interface (accepts optional asset= arg)
    keyword.py           deterministic keyword classifier (lower bound)
    strong.py            oracle-text reader (upper bound reference)
  baselines/
    runner.py            named baselines: random, rule_based, xgboost, opt_only,
                         semantic_only, ours
src/data_generators/
  noise.py               decoupled note-generation noise (shared by S1/S2/S3)
  s1_generator.py        IT Infrastructure synthetic notes (500 assets)
  s2_loader.py           Aviation MRO synthetic SDR records (500 assets)
  s3_loader.py           Consumer Electronics synthetic reviews (1000 assets)
scripts/
  run_experiment.py      per-scenario CSV runner
  run_summary.py         Table IV: mean/std + Wilcoxon (30 seeds)
  run_calibration.py     extractor calibration r + Table V bracket
  run_kappa_sweep.py     Table VI: kappa sensitivity
  make_tables.py         Table IV / ablation CSVs
  make_figures.py        capacity figure
tests/
  test_smoke.py          unit + integration smoke tests (14 tests)
outputs/                 generated CSVs (gitignored)
paper/                   manuscript (LaTeX + PDF)
calibration/             parameter calibration sources (CALIBRATION_SOURCES.md)
```
