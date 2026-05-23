# `scripts/` — run order

Run from the repo root, with `PYTHONPATH=.` so `from src import ...` works.

| # | Command | What it does |
|---|---|---|
| 1 | `bash scripts/setup_data.sh` | Verify (or guide download of) CIFAKE at `data/cifake/` |
| 2 | `python scripts/generate_ood.py` | (Optional, ~10 min) Build sd-turbo cross-generator OOD set at `data/ood_sdturbo/`. Already shipped in `data/ood_sdturbo/`. |
| 3 | `python scripts/evaluate.py --model <name>` | Run val + sealed test + OOD + 19-perturbation robustness for one model. Records land in `results/per_model/<name>/`. |
| 4 | `python scripts/run_team_ensemble.py` | Pairwise / 3-way / all-model ensembles + leave-one-out. Writes `results/team_ensemble_report.json`. |
| 5 | `python scripts/generate_figures.py` | Refresh all comparison figures into `results/figures/` and `docs/findings/`. |

## Example: full pipeline reproduction from scratch

```bash
export PYTHONPATH=.
bash scripts/setup_data.sh
python scripts/generate_ood.py                              # ~10 min

# One model at a time (each takes 5-30 min depending on architecture)
for m in cnn_baseline_yin resnet18_nathan vit_small_alex; do
    python scripts/evaluate.py --model $m
done

# CLIP probe has its own path (loads encoder + cached embeddings)
python -c "from src.clip_probe import extract_embeddings, fit_mlp, run_robust_battery; \
           extract_embeddings(); fit_mlp(); run_robust_battery()"

# Aggregate + figures
python scripts/run_team_ensemble.py
python scripts/generate_figures.py
```

## Skipping bits

- `--skip-robust` to skip the 19-perturbation battery (saves ~30-60 min per model)
- `--splits test ood` to skip the val pass (use cached val threshold if it already exists)

## Where things land

- Per-model metrics: `results/per_model/<name>/{val,test,ood_sdturbo}.json`
- Per-model robust: `results/per_model/<name>/robust_<perturb>_<level>.json` (gitignored, regen via evaluate.py)
- Per-sample scores: `*_scores.npz` next to each metrics JSON (gitignored)
- Team ensemble: `results/team_ensemble_report.json`
- Figures: `results/figures/`
