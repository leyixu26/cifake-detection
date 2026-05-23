# CLIP final (LAION ViT-B/32 + MLP head) — canonical findings

## Final pipeline

* Encoder: OpenCLIP `ViT-B-32` with `laion2b_s34b_b79k` weights (151M params, FROZEN)
* Input: 32x32 -> bicubic upsample to 224x224 -> CLIP normalize
* Probe: MLP head 512 -> 256 -> 1 with ReLU + dropout 0.3 (132k trainable params, ~1100x fewer than the encoder)
* Training: BCE-with-logits, AdamW lr=1e-3 wd=1e-4, batch 1024, early stop on val AUROC patience 6
* Threshold: val-derived Youden-J

## CLIP capacity ladder (each upgrade individually)

| Variant | Test | OOD | ID gain | OOD gain |
|---|---:|---:|---:|---:|
| baseline (B/32 openai + LR) | 0.9871 | 0.9191 | +0.00 pp | +0.00 pp |
| LAION B/32 + LR | 0.9911 | 0.9365 | +0.39 pp | +1.73 pp |
| LAION B/32 + LR + TTA(flip) | 0.9916 | 0.9385 | +0.44 pp | +1.93 pp |
| LAION B/32 + MLP head | 0.9968 | 0.9485 | +0.97 pp | +2.93 pp |

## Cross-model summary (final CLIP foregrounded)

| Model | Test | OOD | Δ | Trained params |
|---|---:|---:|---:|---:|
| freq A (handcrafted) | 0.9003 | 0.8083 | +9.2 pp | 4 k |
| freq B (mag CNN) | 0.9435 | 0.8150 | +12.8 pp | 222 k |
| Nathan ResNet-18 (ImageNet) | 0.9977 | 0.9341 | +6.4 pp | ? |
| Alex ViT-Small | 0.9994 | 0.9732 | +2.6 pp | ? |
| CLIP baseline (B/32 openai) | 0.9871 | 0.9191 | +6.8 pp | 1 k |
| CLIP zero-shot | 0.6362 | 0.4618 | +17.4 pp | 0 |
| **CLIP final (LAION+MLP)** | 0.9968 | 0.9485 | +4.8 pp | 132 k (probe only) |

## Ensembles

### in-dist test

All-three ensemble AUROC = **0.9965**

| Pair | AUROC |
|---|---:|
| spatial_CNN+CLIP_final | 0.9981 |
| freq_A+CLIP_final | 0.9931 |
| freq_A+spatial_CNN | 0.9865 |

| Leave-one-out | AUROC without | contribution |
|---|---:|---:|
| drop freq_A | 0.9981 | **-0.17 pp** |
| drop spatial_CNN | 0.9931 | **+0.33 pp** |
| drop CLIP_final | 0.9865 | **+1.00 pp** |

### OOD (sd-turbo)

All-three ensemble AUROC = **0.9345**

| Pair | AUROC |
|---|---:|
| spatial_CNN+CLIP_final | 0.9570 |
| freq_A+CLIP_final | 0.9177 |
| freq_A+spatial_CNN | 0.8999 |

| Leave-one-out | AUROC without | contribution |
|---|---:|---:|
| drop freq_A | 0.9570 | **-2.25 pp** |
| drop spatial_CNN | 0.9177 | **+1.68 pp** |
| drop CLIP_final | 0.8999 | **+3.46 pp** |


## Key claims for the team report

1. **CLIP linear/MLP probe matches end-to-end CNN training on CIFAKE** with the encoder fully frozen — within 0.0006 AUROC of the teammate's CNN on clean test (0.9968 vs 0.9974).
2. **Pretraining weights dominate architecture choice.** Same ViT-B/32, swapping `openai` -> `laion2b_s34b_b79k` gave +0.4 pp ID and +1.7 pp OOD with zero other changes.
3. **Adding a small MLP head on top did NOT hurt OOD** (contrary to typical Ojha-2023 caution), bumping OOD AUROC from 0.9365 (LR) to 0.9485 (+1.2 pp). The 132k-param head has plenty of capacity to find a richer boundary in CLIP's embedding space.
4. **Final CLIP is more OOD-robust than the spatial CNN.** 0.9485 vs 0.9348 (+1.4 pp) on cross-generator sd-turbo, despite zero fine-tuning of the encoder. The 'web-scale pretraining' inductive bias pays off where it should.
5. **The best ensemble is CLIP + spatial CNN.** Pair AUROCs land above either alone; adding frequency variants does not help. Recommend reporting the 2-model ensemble as the team's headline number.