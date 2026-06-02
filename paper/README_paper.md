# Paper: Semantic Signal-Assisted Decision Support for Reverse Logistics

## Files

- `main.tex` — full IEEE Access manuscript (IEEEtran format)
- `refs.bib` — BibTeX bibliography

## To compile

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex   # run twice for cross-references
```

Requires a TeX distribution (e.g., TeX Live, MiKTeX, MacTeX).

## Key changes from prior draft (v7B)

All numbers are computed by the released RLDB code (`experiments/`) and are
fully reproducible. The following were corrected:

| Item | Old (v7B) | Corrected |
|---|---|---|
| S1 keyword extractor r | 0.87 | **0.47** (decoupled generation) |
| S2 keyword extractor r | 0.45 | **0.32** (decoupled generation) |
| S3 keyword extractor r | 0.40 | **0.36** (decoupled generation) |
| Table IV TRV (all rows) | stale | **run_summary.py output** |
| Table V | r=0.88/0.88 LLM on real FAA/Amazon | **extractor bracket: keyword vs oracle-text** |
| Table VI kappa lift | 3.8–3.9% | **13.70–14.56% (spread 0.86 pp)** |
| LLM blind study (§IV-F) | fabricated claim | **removed; replaced with synthetic bracket** |
| S1 note-generation | coupled (circular) | **decoupled via noise.py** |
| XGBoost extractor | constant φ=0.75 | **trained age→yield mapping** |
| Seeds in make_tables.py | 5 | **30** |
| llm_only baseline | crashed (undefined) | **removed; replaced by semantic_only** |

## Reproducing every table

```bash
cd experiments/
pip install -r requirements.txt

# Table IV
python scripts/run_summary.py --seeds 0-29

# Table V (extractor bracket)
python scripts/run_calibration.py --seeds 0-29

# Table VI
python scripts/run_kappa_sweep.py --seeds 0-29
```
