# Multimodal Product Assistant

A retrieval-augmented product-support chatbot over 10,001 Amazon products.
Answers questions from **text**, returns **product photographs** on request, and
identifies products from an **uploaded photo**, all grounded in the catalog,
with a calibrated confidence gate that declines rather than guessing.

Built for GEN AI Principles Course Project II. The research, evaluation, and
model selection live in a separate repository; this repo is the deployable
application extracted from it.

---

## What it does

| Interaction | Example | How it works |
|---|---|---|
| **Text question** | *"What's the price of the DB Longboards CoreFlex Crossbow?"* | MultiQuery expansion → CLIP text retrieval → grounded answer with cited sources |
| **Image request** | *"Can you show me a picture of the Apple AirPods Pro?"* | Text query → CLIP **image** index → product photographs |
| **Photo upload** | *(user uploads a product photo)* | CLIP ViT-L/14 → fusion retrieval → confidence gate → identification |

The assistant **declines** when the catalog doesn't cover a question, rather than
inventing an answer. That behaviour is the main thing the evaluation measures.

## Measured performance

| Path | Metric | Result |
|---|---|---|
| Text | Evaluation rubric (14 questions) | **13/14** with MultiQuery + Llama-3.3-70B |
| Text retrieval | Recall@5, 1,500 generated customer queries | 0.487 (0.395 without MultiQuery) |
| Image | Recall@1 / Recall@5, **held-out** photos | **0.592 / 0.751** |
| Image gate | Precision @ coverage | **0.804 @ 0.650** (AUC 0.903) |

"Held-out" means the query photo is a *different* picture of the product from the
one indexed, the honest analogue of a customer's own photo.

**Roughly 1 in 5 confident image identifications is still wrong.** The UI shows
three ranked candidates with photos instead of asserting one identity: the top
match alone is right 59% of the time, while the three shown contain the answer
71% of the time. Three is where the gain flattens (+3.8 points for the third
card, +2.1 for a fourth, +1.7 for a fifth), so it is the point past which extra
clutter stops paying for itself. This is an assistive system, not an autonomous
one.

---

## Quick start

```bash
git clone <this-repo> && cd genai-rag-app
python -m venv .venv && .venv\Scripts\activate       # Windows
# python3 -m venv .venv && source .venv/bin/activate # macOS/Linux

pip install -r requirements.txt
cp .env.example .env          # then add your keys

python smoke_test.py          # verify retrieval works (no API calls)
streamlit run app.py
```

A free Groq key ([console.groq.com/keys](https://console.groq.com/keys)) is
enough; it serves both open-weights Llama models. `OPENAI_API_KEY` is only
needed for the GPT-4o-mini option.

## Artifacts

`artifacts/` (47 MB) is the entire serving state:

| File | Size | Contents |
|---|---|---|
| `b32_text.npy` | 10 MB | CLIP ViT-B/32 text embeddings for chat retrieval |
| `l14_text.npy` | 15 MB | CLIP ViT-L/14 text embeddings for the image path |
| `l14_image.npy` | 15 MB | CLIP ViT-L/14 image embeddings for the image path |
| `products.parquet` | 5 MB | Product metadata and CDN image URLs |
| `config.json` | 1 MB | Ids, fusion α, calibrated gate coefficients |

Rebuild from the research project with:

```bash
GENAI_DATA_DIR=/path/to/genai-final-data python prepare_artifacts.py
```

**No vector database.** At 10k products a cosine search is a `(10001, 768)`
matmul, under a millisecond in numpy, and measured at **0.04 s** end-to-end per
uploaded image including CLIP inference. Chroma is used in the research notebook
for the coursework requirement; deploying it here would add a process and a
250 MB sqlite file to serve queries that numpy already answers instantly.
Embeddings are stored as float16, verified lossless for cosine retrieval
(min cosine 1.000000 against fp32).

Product photographs are linked from Amazon's CDN rather than bundled, which
keeps the repo small.

---

## Deployment

### Streamlit Community Cloud

Point it at `app.py`, then add secrets under **App settings → Secrets**:

```toml
GROQ_API_KEY = "gsk_..."
OPENAI_API_KEY = "sk-proj-..."
```

**Install the CPU build of PyTorch** or the default CUDA wheels will blow the
image-size limit. Add a `packages.txt`-style override by pinning in
`requirements.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.2
```

First run downloads the CLIP weights from HuggingFace (~0.6 GB for ViT-B/32,
~1.7 GB for ViT-L/14). They are cached afterwards, but the cold start is slow:
the text encoder loads eagerly, and the image encoder only on first upload, so a
text-only session never pays for ViT-L/14.

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

Pre-baking the HuggingFace cache into the image avoids the cold-start download.

---

## Architecture

```
                    ┌──────────────── Streamlit UI ────────────────┐
                    │  text box            file uploader           │
                    └───────┬──────────────────────┬───────────────┘
                            │                      │
              ┌─────────────▼──────────┐   ┌───────▼─────────────────┐
              │ wants_a_picture()?     │   │ CLIP ViT-L/14           │
              │  routes the request    │   │ image tower             │
              └──┬──────────────────┬──┘   └───────┬─────────────────┘
                 │ no               │ yes          │
    ┌────────────▼───────┐  ┌───────▼──────────┐   │  fusion
    │ follow-up rewrite  │  │ find_product_    │   │  α·(img→img)
    │ (resolves "the     │  │ images()         │   │ +(1-α)·(img→txt)
    │  blue one")        │  │ text→image index │   │
    └────────────┬───────┘  └───────┬──────────┘   │
                 │                  │      ┌───────▼─────────────────┐
    ┌────────────▼───────┐          │      │ logistic gate           │
    │ MultiQuery         │          │      │ 80% precision target    │
    │ expansion (×3)     │          │      └───────┬─────────────────┘
    └────────────┬───────┘          │              │
    ┌────────────▼───────┐          │      ┌───────▼─────────────────┐
    │ CLIP ViT-B/32 text │          │      │ confident? commit :     │
    │ → product index    │          │      │ show candidates         │
    └────────────┬───────┘          │      └───────┬─────────────────┘
                 │                  │              │
    ┌────────────▼──────────────────▼──────────────▼─────────────────┐
    │  LLM (Llama-3.3-70B / Llama-3.1-8B / GPT-4o-mini)              │
    │  few-shot grounding contract + SOURCES: citation line          │
    └────────────────────────────────────────────────────────────────┘
```

**Why the LLM never sees the photograph.** CLIP does the seeing; the LLM reasons
over retrieved catalog text. That division is what the brief specifies, and it is
why a text-only open-weights model is a correct choice here rather than a
compromise.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, routing, transcript rendering |
| `rag_core.py` | All retrieval, fusion, gating, prompts, and answer generation, with no Streamlit import, so it is testable and reusable |
| `prepare_artifacts.py` | Builds the serving bundle from the research project |
| `smoke_test.py` | Exercises every path the UI uses; `--llm` adds live generation |

## Known limitations

- **~1 in 5 confident image answers is wrong** at the calibrated gate. Assistive, not autonomous.
- **Held-out photos are still catalog photos**: professionally lit, plain background. A real phone snapshot is harder, so 0.592 is an upper bound on true upload performance.
- **Conversation memory is a query rewrite**, not full dialogue state. It resolves "the first one" and "the blue one" but does not track long-range context.
- **No reranking.** Recall@50 is 0.873 against Recall@1 0.592, so the correct product is usually retrieved but not ranked first, the clearest remaining headroom.
- **Groq free tier has a daily token cap** that a long demo session can hit. Switch to GPT-4o-mini as a fallback.
- **Catalog is toy-heavy** (~67% Toys & Games), so coverage outside that is thin and the assistant declines more often there.
