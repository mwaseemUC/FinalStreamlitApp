"""Importable core of the multimodal RAG system.

This module is the deployable extraction of the research notebook: no notebook
globals, no Chroma process, no local image corpus. Everything it needs is the
`artifacts/` bundle produced by `prepare_artifacts.py`.

Two retrieval paths, both measured in the research notebook:

  TEXT   query -> CLIP ViT-B/32 text tower -> product text index
         optional MultiQuery expansion (best measured text config)

  IMAGE  upload -> CLIP ViT-L/14 image tower -> fusion of
             alpha * (image -> catalogue images)
           + (1-alpha) * (image -> product text)
         then a logistic confidence gate calibrated to 80% precision

Measured on held-out photos the system has never seen: R@1 0.592, R@5 0.751.
The gate answers ~65% of uploads at ~80% precision, so roughly 1 in 5 confident
answers is still wrong -- which is why the UI presents candidates rather than
asserting a single identity.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARTIFACTS = Path(os.getenv("RAG_ARTIFACTS", HERE / "artifacts"))

# ─────────────────────────────────────────────────────────── prompts

GROUNDING_CORE = """You are a helpful product-support assistant for a large
online marketplace with a wide-ranging catalog -- electronics, toys, home goods,
hobbies, outdoor gear, and more.

Use only the product context provided to answer the customer's question. Do not
infer or add any information, specs, features, or prices that are not explicitly
stated in the context -- for example, do not assume a product is waterproof,
wireless, or has any other quality unless the context says so directly.

There are two different situations where you may not have a full answer:
1. No product in the context is actually relevant to the question. Say
   "I'm sorry, I don't have information on that in our current catalog."
   and cite no sources.
2. A specific, relevant product IS in the context, but it does not state the
   fact being asked about. Say that the specification is not listed for that
   product, and DO cite that product -- you consulted it, even though it didn't
   contain the answer."""

OUTPUT_CONTRACT = """
End your response with a final line in exactly this format
(no parentheses, no extra words):
SOURCES: <product_name> (<uniq_id>), <product_name> (<uniq_id>)
If no products were relevant, write exactly:
SOURCES: <None>"""

# Few-shot is the shipped style: the research grid found every output-contract
# failure was zero-shot, and two examples fixed it for every model tested.
FEW_SHOT = """
Example 1
Context includes: "Acme Wireless Earbuds | Electronics | Bluetooth 5.0, 12-hour
battery life, touch controls"
Question: "Are the Acme Wireless Earbuds waterproof?"
Answer: "The Acme Wireless Earbuds' listing doesn't specify a waterproof rating,
so I can't confirm that."
SOURCES: Acme Wireless Earbuds (example-id-123)

Example 2
Context includes: "Bluebird Wooden Puzzle | Toys & Games | 48 pieces, ages 3+"
Question: "Do you sell laptops?"
Answer: "I'm sorry, I don't have information on that in our current catalog."
SOURCES: <None>
"""

MULTI_QUERY_PROMPT = """You generate search queries for a retrieval system
containing information about products in an online marketplace catalog.

Create exactly {n} alternative search queries for the user's question.

The alternatives should include:
1. A close paraphrase using different wording.
2. A keyword-style query using product-description terminology.
3. A query isolating a distinct entity or attribute, if the question names more
   than one product or asks about a specific feature, price, or comparison.

Rules:
- Return one query per line.
- No numbering, bullets, commentary, or answers.

User question: {question}"""

IMAGE_QA_EXTRA = """

You are identifying a product from a photograph a customer uploaded. You cannot
see the photo yourself -- you are given the catalog entries that a vision model
matched to it, in rank order with similarity scores.

- Lead with the single best match: name it, and describe what it is and how it
  is used, using only its catalog text.
- If the match is marked UNCERTAIN, say plainly that you are not certain, and
  name the close alternatives rather than committing to one.
- Never describe visual details that are not in the catalog text."""


# ─────────────────────────────────────────────────────────── artifacts

@dataclass
class Catalog:
    """The serving bundle: product metadata plus three embedding matrices."""
    products: pd.DataFrame
    text_ids: list
    image_ids: list
    b32_text: np.ndarray
    l14_text: np.ndarray
    l14_image: np.ndarray
    config: dict
    _shared_ids: list = field(default_factory=list)
    _shared_text: np.ndarray | None = None
    _shared_image: np.ndarray | None = None

    @property
    def alpha(self) -> float:
        return float(self.config["fusion_alpha"])

    @property
    def gate(self) -> dict:
        return self.config["gate"]

    def name(self, uid: str) -> str:
        return self._by_id.get(uid, {}).get("product_name", "(unknown)")

    def row(self, uid: str) -> dict:
        return self._by_id.get(uid, {})


@lru_cache(maxsize=1)
def load_catalog(artifacts: str | None = None) -> Catalog:
    """Load the bundle once. Cheap enough to call from a Streamlit cache."""
    root = Path(artifacts) if artifacts else ARTIFACTS
    if not (root / "config.json").exists():
        raise FileNotFoundError(
            f"No artifacts at {root}. Run `python prepare_artifacts.py` on the "
            f"research machine, or set RAG_ARTIFACTS."
        )
    cfg = json.loads((root / "config.json").read_text())
    products = pd.read_parquet(root / "products.parquet")

    cat = Catalog(
        products=products,
        text_ids=list(cfg["text_ids"]),
        image_ids=list(cfg["image_ids"]),
        b32_text=np.load(root / "b32_text.npy").astype(np.float32),
        l14_text=np.load(root / "l14_text.npy").astype(np.float32),
        l14_image=np.load(root / "l14_image.npy").astype(np.float32),
        config=cfg,
    )
    cat._by_id = products.set_index("uniq_id").to_dict("index")

    # Fusion needs both modalities for the same product, aligned row-wise.
    tpos = {u: i for i, u in enumerate(cat.text_ids)}
    ipos = {u: i for i, u in enumerate(cat.image_ids)}
    shared = [u for u in cat.image_ids if u in tpos]
    cat._shared_ids = shared
    cat._shared_text = np.stack([cat.l14_text[tpos[u]] for u in shared])
    cat._shared_image = np.stack([cat.l14_image[ipos[u]] for u in shared])
    return cat


# ─────────────────────────────────────────────────────────── encoders

@lru_cache(maxsize=2)
def _clip(model_id: str):
    """Load a CLIP model lazily -- the image tower is only paid for on upload."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return model, CLIPProcessor.from_pretrained(model_id), device


def embed_text(text: str, model_id: str | None = None) -> np.ndarray:
    import torch
    cat = load_catalog()
    model, proc, device = _clip(model_id or cat.config["clip_text_model"])
    with torch.no_grad():
        x = proc(text=[text], return_tensors="pt", padding=True, truncation=True)
        f = model.get_text_features(**{k: v.to(device) for k, v in x.items()})
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        f = f / f.norm(dim=-1, keepdim=True)
    return f[0].float().cpu().numpy()


def embed_image(image, model_id: str | None = None) -> np.ndarray:
    """image: a PIL.Image or a path."""
    import torch
    from PIL import Image
    cat = load_catalog()
    model, proc, device = _clip(model_id or cat.config["clip_image_model"])
    if not hasattr(image, "convert"):
        image = Image.open(image)
    with torch.no_grad():
        x = proc(images=[image.convert("RGB")], return_tensors="pt")
        x = {k: (v.to(device).half() if v.dtype == torch.float32 and device == "cuda"
                 else v.to(device)) for k, v in x.items()}
        f = model.get_image_features(**x)
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        f = f / f.norm(dim=-1, keepdim=True)
    return f[0].float().cpu().numpy()


# ─────────────────────────────────────────────────────────── retrieval

def search_text(query: str, k: int = 4) -> list[dict]:
    """Plain dense retrieval over the product text index."""
    cat = load_catalog()
    sims = cat.b32_text @ embed_text(query)
    order = np.argsort(-sims)[:k]
    return [_hit(cat, cat.text_ids[i], float(sims[i])) for i in order]


def _hit(cat: Catalog, uid: str, score: float) -> dict:
    r = cat.row(uid)
    return {"uniq_id": uid, "score": score,
            "product_name": r.get("product_name", "(unknown)"),
            "text": r.get("composite_text", ""),
            "image_url": r.get("primary_image", ""),
            "price": r.get("selling_price", ""),
            "category": r.get("top_level_category", "")}


def generate_multi_queries(question: str, llm, n: int = 3) -> list[str]:
    """Query expansion. Measured best text config: MultiQuery + llama-3.3-70b."""
    raw = llm.invoke(MULTI_QUERY_PROMPT.format(question=question, n=n)).content
    out = [question.strip()]
    for line in raw.splitlines():
        q = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"\'')
        if q and q.lower() not in {x.lower() for x in out}:
            out.append(q)
        if len(out) == n + 1:
            break
    return out


def search_text_multiquery(question: str, llm, k_per: int = 4,
                           max_docs: int = 8) -> tuple[list[dict], list[str]]:
    """Round-robin union over expanded queries.

    MultiQuery is a *breadth* technique: corpus-wide it lowers Recall@1 slightly
    while raising Recall@5, and it is what rescues multi-entity questions
    ("which is cheaper, X or Y") that a single search answers with only one of
    the two products.
    """
    queries = generate_multi_queries(question, llm)
    per = {q: search_text(q, k=k_per) for q in queries}
    seen, docs = set(), []
    for rank in range(k_per):
        for q in queries:
            if rank < len(per[q]):
                d = per[q][rank]
                if d["uniq_id"] not in seen:
                    seen.add(d["uniq_id"])
                    docs.append(d)
                if len(docs) >= max_docs:
                    return docs, queries
    return docs, queries


def identify_from_image(image, n: int = 5) -> dict:
    """Fusion retrieval + calibrated confidence gate for an uploaded photo."""
    cat = load_catalog()
    v = embed_image(image).astype(np.float32)
    sims = (cat.alpha * (cat._shared_image @ v)
            + (1 - cat.alpha) * (cat._shared_text @ v))
    order = np.argsort(-sims)[:max(n, 5)]
    top5 = sims[order[:5]]

    g = cat.gate
    feats = np.array([top5[0], top5[0] - top5[1], top5.mean(), top5[0] - top5[4]])
    z = (((feats - np.array(g["scaler_mean"])) / np.array(g["scaler_scale"]))
         @ np.array([g["logistic_coef"][k] for k in
                     ("top1_sim", "margin", "mean_top5", "spread5")])
         + g["logistic_intercept"])
    p = float(1 / (1 + np.exp(-z)))

    return {"confident": p >= g["threshold"], "confidence": p,
            "threshold": g["threshold"],
            "candidates": [_hit(cat, cat._shared_ids[i], float(sims[i])) for i in order[:n]]}


def find_product_images(query: str, k: int = 3) -> list[dict]:
    """Text query -> product photographs.

    The third interaction type in the brief ("Can you show me a picture of X?").
    Searches the IMAGE index with a text embedding, which is what the shared
    CLIP space is for.
    """
    cat = load_catalog()
    q = embed_text(query, model_id=cat.config["clip_image_model"])
    tpos = {u: i for i, u in enumerate(cat.image_ids)}
    sims = cat.l14_image @ q
    order = np.argsort(-sims)[:k]
    return [_hit(cat, cat.image_ids[i], float(sims[i])) for i in order]


# ─────────────────────────────────────────────────────────── models

MODELS = {
    "llama-3.3-70b": {"provider": "groq", "id": "llama-3.3-70b-versatile",
                      "label": "Llama 3.3 70B (open weights, best measured)"},
    "llama-3.1-8b": {"provider": "groq", "id": "llama-3.1-8b-instant",
                     "label": "Llama 3.1 8B (open weights, fastest)"},
    "gpt-4o-mini": {"provider": "openai", "id": "gpt-4o-mini",
                    "label": "GPT-4o-mini (hosted, low latency)"},
}


@lru_cache(maxsize=4)
def get_chat_model(key: str, temperature: float = 0.0):
    spec = MODELS[key]
    if spec["provider"] == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=spec["id"], temperature=temperature)
    from langchain_groq import ChatGroq
    return ChatGroq(model=spec["id"], temperature=temperature)


# ─────────────────────────────────────────────────────────── answering

def _format_context(docs: Iterable[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {d['product_name']} ({d['uniq_id']})\n{d['text'][:900]}"
        for i, d in enumerate(docs, 1)
    )


def _history_block(history, limit: int = 6) -> str:
    if not history:
        return ""
    turns = history[-limit:]
    lines = "\n".join(f"{r.upper()}: {t}" for r, t in turns)
    return ("\nConversation so far (for pronoun/reference resolution only -- never "
            "use it as a source of product facts):\n" + lines + "\n")


def _resolve_followup(question: str, history, llm) -> str:
    """Rewrite a context-dependent follow-up into a standalone query.

    Without this, "what about the blue one?" embeds as a query about the colour
    blue and retrieves nothing useful. Retrieval has no memory; only the rewrite
    step does.
    """
    if not history:
        return question
    recent = "\n".join(f"{r.upper()}: {t}" for r, t in history[-4:])
    prompt = (
        "Rewrite the user's latest message as a standalone product-search query, "
        "resolving any pronouns or references using the conversation. Return only "
        "the rewritten query, nothing else. If it is already standalone, return it "
        f"unchanged.\n\nConversation:\n{recent}\n\nLatest message: {question}"
    )
    try:
        return llm.invoke(prompt).content.strip().strip('"') or question
    except Exception:
        return question


def answer_text_question(question: str, model_key: str = "llama-3.3-70b",
                         use_multiquery: bool = True, history=None) -> dict:
    """Text chat path. Returns answer, sources, retrieved docs, and diagnostics."""
    from langchain_core.prompts import ChatPromptTemplate

    # Catalog-wide questions are answered by computation, not retrieval. Doing
    # this before the LLM sees anything prevents a confident wrong answer over
    # an arbitrary top-k slice.
    if is_aggregate_query(question):
        agg = answer_aggregate(question)                  # fast keyword path
        kind = None
        if agg is None:
            # Keywords could not resolve it. Before refusing, let the model
            # classify the intent -- this is what makes typos and paraphrases
            # work ("exepensive", "what costs the most money"), and lets an
            # over-inclusive pre-filter hand ordinary questions back to search.
            kind = classify_aggregate(question, get_chat_model(model_key))
            if kind != "not_aggregate":
                agg = answer_aggregate(question, kind=kind)
        if agg is not None:
            return {"answer": agg["answer"], "sources": "", "docs": agg["docs"],
                    "queries": [question], "rewritten_query": question,
                    "aggregate": agg["kind"]}
        # Only refuse when the question really is a catalog-wide one we cannot
        # compute. If the classifier says it is an ordinary product question,
        # fall through to normal retrieval below.
        if kind != "not_aggregate":
            return {
                "answer": ("That's a question about the catalog as a whole, and "
                           "the catalog doesn't carry the data to answer it — I "
                           "have product names, categories, prices, descriptions "
                           "and images, but not ratings, reviews, sales or "
                           "stock.\n\nI can answer exactly: the most or least "
                           "expensive product, the average price, and how many "
                           "products there are. Or ask me about any specific "
                           "product."),
                "sources": "", "docs": [], "queries": [question],
                "rewritten_query": question, "aggregate": "unsupported",
            }

    llm = get_chat_model(model_key)
    search_query = _resolve_followup(question, history, llm) if history else question

    if use_multiquery:
        docs, queries = search_text_multiquery(search_query, llm)
    else:
        docs, queries = search_text(search_query, k=4), [search_query]

    system = GROUNDING_CORE + "\n" + FEW_SHOT + OUTPUT_CONTRACT
    messages = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{history}Product context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ]).format_messages(history=_history_block(history),
                       context=_format_context(docs), question=question)

    raw = llm.invoke(messages).content.strip()
    answer, _, sources = raw.partition("SOURCES:")
    return {"answer": answer.strip(), "sources": sources.strip(), "docs": docs,
            "queries": queries, "rewritten_query": search_query}


def answer_image_question(image, question: str | None = None,
                          model_key: str = "llama-3.3-70b", history=None) -> dict:
    """Image upload path: fusion retrieval, confidence gate, grounded answer."""
    from langchain_core.prompts import ChatPromptTemplate

    question = question or ("Can you identify the product in this image and "
                            "describe its usage?")
    result = identify_from_image(image, n=5)
    docs = result["candidates"]

    context = "\n\n".join(
        f"[Rank {i}, similarity {d['score']:.3f}] {d['product_name']} ({d['uniq_id']})\n"
        f"{d['text'][:900]}" for i, d in enumerate(docs, 1)
    )
    if not result["confident"]:
        context = ("MATCH QUALITY: UNCERTAIN -- the vision model is not confident in a "
                   "single identification.\n\n" + context)

    llm = get_chat_model(model_key)
    messages = ChatPromptTemplate.from_messages([
        ("system", GROUNDING_CORE + IMAGE_QA_EXTRA + "\n" + FEW_SHOT + OUTPUT_CONTRACT),
        ("human", "{history}Matched catalog entries:\n{context}\n\n"
                  "Question: {question}\n\nAnswer:"),
    ]).format_messages(history=_history_block(history), context=context, question=question)

    raw = llm.invoke(messages).content.strip()
    answer, _, sources = raw.partition("SOURCES:")
    return {**result, "answer": answer.strip(), "sources": sources.strip(), "docs": docs}


# ─────────────────────────────────────────────────────────── routing

_IMAGE_REQUEST = re.compile(
    r"\b(show|see|picture|photo|image|what does .* look like|looks? like)\b", re.I)

# ─────────────────────────────────────────────── aggregate queries

_SUPERLATIVE = re.compile(
    r"\b(most|least|cheapest|priciest|dearest|highest|lowest|max(?:imum)?|"
    r"min(?:imum)?|biggest|smallest|top|average|mean|median|typical|usual|"
    r"total|overall|altogether|how many|how much.*(?:in total|overall)|count)\b",
    re.I)
_PRICE_WORD = re.compile(
    r"\b(expensive|exp?ensive\w*|pric\w*|cost\w*|cheap\w*|dear)\b", re.I)


def is_aggregate_query(question: str) -> bool:
    """Cheap pre-filter for questions that depend on the WHOLE catalog.

    "What's the most expensive product?" cannot be answered by semantic
    retrieval: the top-k documents most *similar* to that sentence have nothing
    to do with the maximum price. Left unhandled, the model dutifully answers
    "most expensive" over whatever 8 documents it was handed and produces a
    confidently wrong answer citing a real product -- the worst failure shape,
    because it looks correct.

    This only decides whether to spend one classification call, so it is
    deliberately over-inclusive: a false positive costs a cheap LLM call that
    returns "unsupported" and falls through to normal retrieval, while a false
    negative costs a wrong answer.
    """
    return bool(_SUPERLATIVE.search(question) or
                (_PRICE_WORD.search(question) and
                 re.search(r"\b(any|all|your|the) (product|item|thing|stuff)s?\b",
                           question, re.I)))


AGGREGATE_KINDS = ("most_expensive", "least_expensive", "count",
                   "average_price", "unsupported", "not_aggregate")

_CLASSIFY_PROMPT = """Classify the user's question about a product catalog into
exactly one label. Reply with the label only, nothing else.

most_expensive  - asks which product costs the most (highest price, priciest)
least_expensive - asks which product costs the least (cheapest, lowest price)
count           - asks how many products the catalog contains
average_price   - asks for the average, typical or median price
unsupported     - a question about the catalog AS A WHOLE that needs data we do
                  not have (popularity, ratings, reviews, sales, stock, colours)
not_aggregate   - an ordinary question about a specific product or category,
                  answerable by searching for relevant products

The catalog has only product names, categories, prices, descriptions and images.
Tolerate typos and paraphrasing.

Examples:
"whats the most exepensive product" -> most_expensive
"which item costs the most money" -> most_expensive
"what is the most popular product" -> unsupported
"do you have any cheap plush toys" -> not_aggregate
"what's the price of the Qwinto" -> not_aggregate

Question: {q}
Label:"""


def classify_aggregate(question: str, llm) -> str:
    """Fallback intent classifier for aggregate questions.

    Keyword matching is brittle in exactly the way that matters here: a typo
    ("exepensive") or a paraphrase ("what costs the most money") routes a
    question we CAN answer exactly into a refusal. One cheap LLM call on the
    aggregate path is worth more than an ever-growing regex.
    """
    try:
        label = llm.invoke(_CLASSIFY_PROMPT.format(q=question)).content.strip().lower()
    except Exception:
        return "unsupported"
    label = re.sub(r"[^a-z_]", "", label)
    return label if label in AGGREGATE_KINDS else "unsupported"


def answer_aggregate(question: str, kind: str | None = None) -> dict | None:
    """Answer catalog-wide questions with a structured lookup instead of RAG.

    `kind` forces a branch when the caller has already classified the question
    (see classify_aggregate). Returns None when the question is an aggregate we
    cannot compute, so the caller can decline honestly rather than guess.
    """
    cat = load_catalog()
    q = question.lower()
    df = cat.products

    if "_price_value" not in df.columns:
        df["_price_value"] = df["selling_price"].map(_parse_price)
    priced = df[df["_price_value"].notna()]

    if kind == "unsupported":
        return None

    # An aggregate can be scoped ("most expensive funko pop"). Computing the
    # global maximum for a scoped question returns a real product that answers
    # a question nobody asked.
    subject = _aggregate_subject(question)

    # ---- count -----------------------------------------------------------
    if kind == "count" or (kind is None and re.search(r"\bhow many\b", q)
                           and not _PRICE_WORD.search(q)):
        if subject:
            pool, how = _scope_to_subject(df, subject)
            if not len(pool):
                return {"answer": f"I couldn't find any products matching "
                                  f"“{subject}” in the catalog.",
                        "docs": [], "kind": "count"}
            return {"answer": f"The catalog contains **{len(pool):,}** products "
                              f"matching “{subject}”.",
                    "docs": [_hit(cat, r["uniq_id"], 0.0)
                             for _, r in pool.head(3).iterrows()],
                    "kind": "count"}
        return {"answer": f"The catalog contains {len(df):,} products.",
                "docs": [], "kind": "count"}

    # ---- price superlatives / averages -----------------------------------
    # Must be about PRICE, not merely contain a superlative: "the most popular
    # product" also matches \bmost\b but is not a price question, and answering
    # it with the most expensive product would be a confident non-sequitur.
    is_price_super = kind in ("most_expensive", "least_expensive", "average_price") or (
        kind is None and (
            re.search(r"\b(cheapest|priciest|dearest)\b", q)
            or (re.search(r"\b(most|least)\b", q) and re.search(r"\bexpensive\b", q))
            or (_PRICE_WORD.search(q)
                and re.search(r"\b(most|least|highest|lowest|max|min|average|mean)\b", q))
        )
    )
    if is_price_super:
        pool, how = _scope_to_subject(priced, subject)
        if subject and not len(pool):
            return {"answer": (f"I couldn't find any products matching “{subject}” "
                               f"in the catalog, so I can't work that out. "
                               f"Try a different wording, or ask about the catalog "
                               f"as a whole."),
                    "docs": [], "kind": "unsupported_scope"}
        scope = f" matching “{subject}”" if subject else " in the catalog"
        basis = (f"Computed across all {len(pool):,} products{scope} with a listed "
                 f"price" + (f" ({how})" if subject else "") + ", not from "
                 "retrieved documents.")

        if kind == "average_price" or (kind is None and re.search(r"\b(average|mean)\b", q)):
            return {"answer": f"The average price across the {len(pool):,} "
                              f"products{scope} with a listed price is "
                              f"**${pool['_price_value'].mean():,.2f}**.",
                    "docs": [], "kind": "average_price"}

        cheapest = (kind == "least_expensive" if kind
                    else bool(re.search(r"\b(cheapest|least expensive|lowest|min)", q)))
        top = (pool.nsmallest(5, "_price_value") if cheapest
               else pool.nlargest(5, "_price_value"))
        row = top.iloc[0]
        label = "least" if cheapest else "most"
        return {
            "answer": (f"The {label} expensive product{scope} is "
                       f"**{row['product_name']}** at {row['selling_price']}.\n\n"
                       f"({basis})"),
            "docs": [_hit(cat, r["uniq_id"], float(r["_price_value"]))
                     for _, r in top.iterrows()],
            "kind": f"{label}_expensive",
        }

    return None


# Words that carry no product meaning in an aggregate question. Everything left
# after removing these is treated as the SUBJECT the aggregate is scoped to:
# "most expensive funko pop" -> "funko pop".
_AGG_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "of", "in", "on", "for", "to", "do",
    "does", "you", "your", "yours", "i", "me", "my", "we", "us", "have", "has",
    "got", "sell", "sells", "selling", "carry", "there", "here", "it", "its",
    "what", "whats", "which", "who", "how", "many", "much", "tell", "show",
    "find", "give", "can", "could", "please", "and", "or", "any", "some", "all",
    "most", "least", "cheapest", "priciest", "dearest", "expensive", "cheap",
    "cheaper", "price", "priced", "prices", "pricing", "cost", "costs", "costly",
    "highest", "lowest", "max", "maximum", "min", "minimum", "top", "best",
    "average", "mean", "median", "typical", "usual", "total", "overall",
    "product", "products", "item", "items", "thing", "things", "stuff",
    "catalog", "catalogue", "store", "shop", "inventory", "money", "dollar",
    "dollars", "worth", "one", "ones", "that", "this", "these", "those",
}


def _aggregate_subject(question: str) -> str:
    """What the aggregate is scoped to, if anything.

    "what's the most expensive product" -> ""            (whole catalog)
    "most expensive funko pop"           -> "funko pop"   (scoped)
    """
    toks = [t for t in re.findall(r"[a-z0-9'&-]+", question.lower())
            if t not in _AGG_STOPWORDS and len(t) > 1]
    return " ".join(toks).strip()


def _scope_to_subject(df, subject: str):
    """Narrow the candidate pool to products matching `subject`.

    Literal token matching first -- for a brand like "funko pop" that is both
    precise and complete, which semantic search is not. Falls back to retrieval
    when the words do not appear literally ("gifts for toddlers").
    Returns (pool, how) where how describes the match for the user.
    """
    if not subject:
        return df, ""

    hay = (df["product_name"].fillna("") + " " + df["category"].fillna("")).str.lower()
    mask = None
    for tok in subject.split():
        # Crude singular stem so "funko pops" matches "Funko Pop!". Substring
        # rather than word-boundary matching, so plurals and punctuation
        # ("Pop!", "Pops") fall out for free.
        stem = tok[:-1] if len(tok) > 3 and tok.endswith("s") else tok
        m = hay.str.contains(re.escape(stem), regex=True, na=False)
        mask = m if mask is None else (mask & m)
    if mask is not None and mask.any():
        return df[mask], "name or category"

    # NO semantic fallback here, deliberately. Retrieval always returns its
    # k nearest products, however unrelated -- so taking a maximum over them
    # reintroduces exactly the bug this function exists to prevent: a confident
    # answer about "the most expensive flying unicorn submarine" that is really
    # just the most expensive item in an arbitrary pool. If the words do not
    # appear in the catalog, say so.
    return df.iloc[0:0], "none"


def _parse_price(s) -> float | None:
    if not isinstance(s, str) or not s:
        return None
    m = re.findall(r"\$\s*([\d,]+\.?\d*)", s)
    return float(m[0].replace(",", "")) if m else None


def cited_ids(sources: str) -> list[str]:
    """Extract the uniq_ids the model actually cited from its SOURCES: line.

    The UI should show what the ANSWER used, not everything retrieval returned --
    otherwise an answer about one product is illustrated with the other seven
    candidates, which reads as if the assistant got it wrong.
    """
    return re.findall(r"\(([0-9a-f]{32})\)", sources or "")


def wants_a_picture(question: str) -> bool:
    """Detect the brief's third interaction type: 'show me a picture of X'."""
    q = question.lower()
    return bool(_IMAGE_REQUEST.search(q)) and not q.startswith("what is")


def strip_image_request(question: str) -> str:
    """Reduce 'can you show me a picture of the X?' to 'the X' for retrieval."""
    q = re.sub(r"^(can you |could you |please )?(show|find|get) (me )?(a |an |the )?"
               r"(picture|photo|image|pic)s? of ", "", question, flags=re.I)
    return re.sub(r"[?.!]+$", "", q).strip() or question
