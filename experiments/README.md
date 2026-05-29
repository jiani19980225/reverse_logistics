# S2S Experiments — Reproducible Simulation

Companion code for: *Semantic Signal–Assisted Decision Support for Reverse
Logistics Resource Recovery Under Uncertainty* (IEEE IEEM 2026).

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

### 3. What controls ±std in Table 5?

Each seed controls three things via a single `np.random.default_rng(seed)`:
- **Batch composition**: which assets are generated (asset types, ages, conditions)
- **Per-asset yield draws**: realized Beta samples at decision time
- **Demand realization**: downstream demand fulfilled

Set once at the top of `pipeline.py`. All downstream RNG is derived from
child generators split from this master RNG. Documented here and enforced
in `test_smoke.py`.

## Quick Start

```bash
pip install -r requirements.txt

# Smoke test (10 assets, 2 seeds, <5 seconds)
python -m pytest tests/ -v

# Full experiment (500/1000 assets, 5 seeds, ~2 min)
python scripts/run_experiment.py --config configs/s1_it.yaml --seeds 0,1,2,3,4
python scripts/run_experiment.py --config configs/s2_aviation.yaml --seeds 0,1,2,3,4
python scripts/run_experiment.py --config configs/s3_consumer.yaml --seeds 0,1,2,3,4

# Generate tables and figures
python scripts/make_tables.py
python scripts/make_figures.py
```

## Structure

```
configs/           scenario + ablation YAML configs
src/s2s/           core framework (beta_model, inspection_policy, decision_engine, pipeline)
src/s2s/extractors/   extractor implementations (keyword, null, llm stub)
src/s2s/baselines/    baseline allocators (random, fifo, xgboost, opt_only, llm_cot)
src/data_generators/  asset generation per scenario
tests/             smoke + unit tests
scripts/           CLI entry points
outputs/           generated CSVs (gitignored except column schema)
data/raw/          downloaded public data (gitignored)
data/synthetic/    generated S1 narratives (committed)
data/processed/    normalized asset JSONs
```
