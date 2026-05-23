# `results/`

What lives here — and what doesn't.

## What's checked in

```
results/
  per_model/
    cnn_baseline_yin/   README.md  (pending checkpoint)
    resnet18_nathan/    test.json  val.json  ood_sdturbo.json
    vit_small_alex/     test.json  val.json  ood_sdturbo.json
    freq_detector/      test.json  val.json  ood_sdturbo.json   (canonical = magnitude SpectrumCNN)
    clip_mlp_vit_b32_laion/  test.json  val.json  ood_sdturbo.json
  team_ensemble_report.json    pairwise + 3-way + all-ensemble + leave-one-out
  figures/                     12 curated PNGs + README captioning each
```

## What's gitignored (regenerable)

| Pattern | Why | How to regenerate |
|---|---|---|
| `per_model/*/robust_*.json` | 19 perturbation × N models = a lot of JSONs | `python scripts/evaluate.py --model <name>` (drops `--skip-robust`) |
| `per_model/*/*_scores.npz` | Per-sample probability arrays needed for ensembling but heavy | Same as above |
| any `*.npz`, `*.npy` | numpy caches | Re-run the relevant script |

## JSON schema

Every record under `per_model/<model>/` matches:

```json
{
  "model_name": "resnet18_nathan",
  "split": "test" | "val" | "ood:sdturbo" | "robust:<perturb>@<level>",
  "n": 20000,
  "positive_label": "FAKE(AI-generated)=1",
  "threshold": {"value": 0.528, "policy": "best_val_youden"},
  "metrics": {"accuracy", "auroc", "f1", "precision", "recall"},
  "confusion": {"tn", "fp", "fn", "tp"},
  "curves": {"roc": {"fpr","tpr"}, "pr": {"recall","precision"}},
  "provenance": {"timestamp", "seed", "config", "libs"}
}
```

The `team_ensemble_report.json` has its own schema (per-model AUROC,
pairwise/triple ensemble AUROCs, leave-one-out contributions); see
`src/ensemble.py`.

## Headline numbers (snapshot — see figures for the full story)

| Model | Test AUROC | OOD AUROC (sd-turbo) | OOD drop |
|---|---:|---:|---:|
| Yin CNN (reported, pending re-eval) | 0.9974 | n/a | — |
| Nathan ResNet-18 | 0.9977 | 0.9341 | −6.4 pp |
| Alex ViT-Small | 0.9994 | 0.9732 | **−2.6 pp** (smallest) |
| Leyi freq detector (mag CNN) | 0.9435 | 0.8150 | −12.8 pp |
| Leyi CLIP probe (LAION + MLP) | 0.9968 | 0.9485 | −4.9 pp |
| **Best pair: Alex ViT + CLIP probe** | 0.9988 | **0.9657** | — |
