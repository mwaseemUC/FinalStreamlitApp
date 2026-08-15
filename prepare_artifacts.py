"""Build the slim serving bundle from the research project.

The research project carries ~600 MB of artifacts (Chroma sqlite, 10k product
JPEGs, full-precision embedding matrices). None of that is needed to *serve*:

  * Chroma is unnecessary at this scale -- a 10k x 768 matmul is sub-millisecond
    in numpy, with no database process to deploy.
  * The product photos are already on Amazon's CDN; the app links to them.
  * float16 halves the matrices with no measurable retrieval difference
    (cosine agreement > 0.9999 -- asserted below rather than assumed).

Result: ~55 MB, deployable to Streamlit Cloud / Spaces / a container.

Run this once from the research machine, then commit `artifacts/` (or attach it
to a GitHub release and let `rag_core` download it).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts"
OUT.mkdir(exist_ok=True)

RESEARCH = Path(os.getenv("GENAI_DATA_DIR", Path.home() / "genai-final-data"))
CACHE = RESEARCH / "cache"
STORE = CACHE / "embeddings"

B32, L14 = "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"


def _load(model_id: str, kind: str):
    p = STORE / f"{model_id.replace('/', '--')}__{kind}.npz"
    if not p.exists():
        sys.exit(f"missing {p}\nRun the research notebook first, or set GENAI_DATA_DIR.")
    d = np.load(p, allow_pickle=True)
    return list(d["ids"]), d["mat"]


def to_fp16(mat: np.ndarray, label: str) -> np.ndarray:
    """Downcast, then prove the downcast is harmless for cosine retrieval."""
    half = mat.astype(np.float16)
    a = mat[:200].astype(np.float32)
    b = half[:200].astype(np.float32)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    assert cos.min() > 0.9999, f"{label}: fp16 changed the vectors ({cos.min():.6f})"
    print(f"  {label:26s} {mat.shape} fp32 {mat.nbytes/1e6:5.1f}MB "
          f"-> fp16 {half.nbytes/1e6:5.1f}MB  (min cosine {cos.min():.6f})")
    return half


def main() -> None:
    print(f"source: {RESEARCH}\n")

    text_ids_b, b32_text = _load(B32, "text_priced")
    text_ids_l, l14_text = _load(L14, "text_priced")
    img_ids_l, l14_img = _load(L14, "image")
    assert text_ids_b == text_ids_l, "text id ordering differs between encoders"

    print("embeddings:")
    np.save(OUT / "b32_text.npy", to_fp16(b32_text, "B/32 text (chat)"))
    np.save(OUT / "l14_text.npy", to_fp16(l14_text, "L/14 text (image path)"))
    np.save(OUT / "l14_image.npy", to_fp16(l14_img, "L/14 image (image path)"))

    # ---- product metadata, trimmed to what the UI actually renders ----------
    df = pd.read_parquet(CACHE / "corpus_clean_v2_price.parquet")
    keep = ["Uniq Id", "Product Name", "Category", "top_level_category",
            "Selling Price", "composite_text", "primary_image"]
    slim = (df[keep]
            .rename(columns={"Uniq Id": "uniq_id", "Product Name": "product_name",
                             "Category": "category", "Selling Price": "selling_price"})
            .set_index("uniq_id")
            .loc[text_ids_b]          # align rows to the embedding matrices
            .reset_index())
    slim.to_parquet(OUT / "products.parquet", index=False, compression="zstd")
    print(f"\nproducts.parquet           {len(slim):,} rows  "
          f"{(OUT / 'products.parquet').stat().st_size/1e6:.1f}MB")

    # ---- ids + calibrated gate ---------------------------------------------
    gate = json.loads((CACHE / "gate_model.json").read_text())["ViT-L/14"]
    (OUT / "config.json").write_text(json.dumps({
        "text_ids": text_ids_b,
        "image_ids": img_ids_l,
        "clip_text_model": B32,
        "clip_image_model": L14,
        "fusion_alpha": gate["alpha"],
        "gate": gate,
        "notes": {
            "fusion_alpha": "weight on image->image; (1-alpha) on image->text",
            "gate": "logistic over [top1_sim, margin, mean_top5, spread5], "
                    "calibrated to 80% precision on the answered set",
        },
    }, indent=1))
    print(f"config.json                {(OUT / 'config.json').stat().st_size/1e6:.1f}MB")

    total = sum(p.stat().st_size for p in OUT.iterdir()) / 1e6
    print(f"\nbundle total: {total:.1f} MB -> {OUT}")
    if total > 95:
        print("WARNING: individual files over 100MB need Git LFS or a release asset.")


if __name__ == "__main__":
    main()
