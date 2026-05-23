"""Generate the cross-generator OOD set (sd-turbo) for evaluation.

Methodology — what we control:
  * Generator        : stabilityai/sd-turbo (different VAE + sampler + training data from SD-1.4)
  * Prompts          : the 10 CIFAR-10 class names (matched content distribution)
  * Resolution       : generate at 256x256, Lanczos resize to 32x32 (matches CIFAKE downsampling)
  * Real side        : sampled from CIFAKE's test/REAL (matched content + JPEG history)
  * JPEG quant tables: re-encoded through CIFAKE's exact tables (luma 1858, chroma 2780)
                       so the *only* difference between in-dist and OOD is the generator

Output layout (matches scripts/evaluate.py expectations):
    data/ood_sdturbo/
        REAL/    1000 CIFAR-10 originals
        FAKE/    1000 sd-turbo generated images
"""
from __future__ import annotations
import argparse
import glob
import os
import pathlib
import random
import shutil

import numpy as np
import torch
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OOD = REPO_ROOT / "data" / "ood_sdturbo"
DEFAULT_CIFAKE = REPO_ROOT / "data" / "cifake"

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
SEED = 42


def _device():
    if torch.cuda.is_available():        return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def copy_reals(cifake_root, ood_root, n=1000):
    src = pathlib.Path(cifake_root) / "test" / "REAL"
    dst = pathlib.Path(ood_root) / "REAL"
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(os.listdir(src))
    rng = random.Random(SEED); rng.shuffle(files)
    for k, fn in enumerate(files[:n]):
        shutil.copy(src / fn, dst / f"{k:04d}.jpg")
    print(f"copied {n} REAL images -> {dst}")


def gen_fakes(ood_root, n_per_class=100, steps=1):
    from diffusers import AutoPipelineForText2Image
    dev = _device()
    print(f"loading sd-turbo on {dev} (downloads ~2 GB on first call)")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float32  # fp32 for MPS reliability
    ).to(dev)
    pipe.set_progress_bar_config(disable=True)

    dst = pathlib.Path(ood_root) / "FAKE"
    dst.mkdir(parents=True, exist_ok=True)
    i = 0
    for cls in CIFAR10_CLASSES:
        prompt = f"a photograph of a {cls}"
        for k in range(n_per_class):
            g = torch.Generator(dev).manual_seed(i + 1000)
            img = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0,
                       height=256, width=256, generator=g).images[0]
            img = img.resize((32, 32), Image.LANCZOS)
            img.save(dst / f"{cls}_{k:03d}.jpg", "JPEG", quality=95)
            i += 1
            if i % 50 == 0:
                print(f"  generated {i}/{len(CIFAR10_CLASSES)*n_per_class}")
    print(f"done -> {dst}")


def reencode_with_cifake_qtables(ood_root, cifake_root):
    """Re-encode every OOD image through CIFAKE's exact JPEG quant tables so
    compression history matches in-dist exactly."""
    ref = Image.open(sorted(glob.glob(f"{cifake_root}/train/REAL/*.jpg"))[0])
    qtables = ref.quantization
    sum_str = ",".join(str(int(np.sum(v))) for v in qtables.values())
    print(f"CIFAKE qtables sums={sum_str} — re-encoding OOD to match")
    for cls in ("REAL", "FAKE"):
        for p in glob.glob(f"{ood_root}/{cls}/*.jpg"):
            im = Image.open(p).convert("RGB")
            im.save(p, "JPEG", qtables=qtables)
        n = len(glob.glob(f"{ood_root}/{cls}/*.jpg"))
        after = Image.open(glob.glob(f"{ood_root}/{cls}/*.jpg")[0]).quantization
        s = sum(int(np.sum(v)) for v in after.values())
        print(f"  {cls}: {n} files re-encoded, qt-sum = {s}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ood-root", default=str(DEFAULT_OOD),
                    help=f"Output directory for the OOD set (default: {DEFAULT_OOD}). "
                         f"Two subdirs REAL/ and FAKE/ are created here.")
    ap.add_argument("--cifake-root", default=str(DEFAULT_CIFAKE),
                    help=f"CIFAKE root with test/REAL/ used to sample the REAL side and "
                         f"to read the canonical JPEG quantization tables "
                         f"(default: {DEFAULT_CIFAKE}).")
    ap.add_argument("--n-per-class", type=int, default=100,
                    help="FAKE images per CIFAR-10 class (default 100 = 1000 total)")
    ap.add_argument("--skip-reals", action="store_true", help="don't (re)copy REAL side")
    ap.add_argument("--skip-fakes", action="store_true", help="don't (re)generate FAKE side")
    ap.add_argument("--skip-reencode", action="store_true",
                    help="don't re-encode to match CIFAKE qtables (NOT recommended)")
    args = ap.parse_args()

    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    os.makedirs(args.ood_root, exist_ok=True)

    if not args.skip_reals: copy_reals(args.cifake_root, args.ood_root)
    if not args.skip_fakes: gen_fakes(args.ood_root, n_per_class=args.n_per_class)
    if not args.skip_reencode:
        reencode_with_cifake_qtables(args.ood_root, args.cifake_root)
    print(f"\nOOD set ready at {args.ood_root}")


if __name__ == "__main__":
    main()
