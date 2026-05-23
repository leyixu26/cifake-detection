"""CLIP-probe AI-image detector (Model 5)."""
from .pipeline import (
    extract_embeddings,
    fit_linear,
    fit_mlp,
    tta_linear,
    run_robust_battery,
    load_encoder,
    load_mlp_from_ckpt,
    MLPHead,
    ENCODERS,
    DEFAULT_TAG,
)
