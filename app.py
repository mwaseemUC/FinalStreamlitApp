"""Multimodal product-support assistant — Streamlit UI.

Three interaction types, matching the project brief:
  1. text question              -> grounded answer with cited sources
  2. "show me a picture of X"   -> product photographs
  3. image upload               -> identification, confidence-gated

Design note: the image path deliberately presents *candidates* rather than
asserting one identity. Measured on held-out photos, top-1 is right 59% of the
time and top-5 contains the answer 75% of the time, so a ranked shortlist is an
honest surface for the accuracy the system actually has.
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

import rag_core as core

load_dotenv()

st.set_page_config(page_title="Multimodal Product Assistant",
                   page_icon="🛍️", layout="wide")


# ───────────────────────────────────────────────────────── resources

@st.cache_resource(show_spinner="Loading catalog…")
def _catalog():
    return core.load_catalog()


@st.cache_resource(show_spinner="Loading CLIP text encoder…")
def _warm_text_encoder():
    core.embed_text("warmup")
    return True


def _missing_keys(model_key: str) -> str | None:
    provider = core.MODELS[model_key]["provider"]
    var = "GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY"
    return None if os.getenv(var) else var


# ───────────────────────────────────────────────────────── sidebar

st.sidebar.title("🛍️ Product Assistant")
st.sidebar.caption("CLIP retrieval + grounded LLM answers over 10,001 Amazon products")

model_key = st.sidebar.selectbox(
    "Answer model",
    list(core.MODELS),
    format_func=lambda k: core.MODELS[k]["label"],
    index=0,
)
use_mq = st.sidebar.toggle(
    "MultiQuery retrieval", value=True,
    help="Expands your question into several search queries and unions the "
         "results. Measured: 13/14 vs 10/14 on the evaluation set. Slower.",
)
show_debug = st.sidebar.toggle("Show retrieval details", value=False)

st.sidebar.divider()
st.sidebar.markdown(
    """**Measured performance**

*Text* — 13/14 on the evaluation rubric (MultiQuery + Llama-3.3-70B).

*Image* — R@1 0.59, R@5 0.75 on photographs the system has never seen.
The confidence gate answers ~65% of uploads at ~80% precision.

**~1 in 5 confident image answers is still wrong**, which is why candidates are
shown rather than a single assertion."""
)

missing = _missing_keys(model_key)
if missing:
    st.sidebar.error(f"{missing} not set — add it to `.env`")

if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()


# ───────────────────────────────────────────────────────── state

if "messages" not in st.session_state:
    st.session_state.messages = []          # list[(role, text)] for the LLM
if "rendered" not in st.session_state:
    st.session_state.rendered = []          # list[dict] for the transcript


def _history():
    return st.session_state.messages


def _render_products(items, caption_score: str = "match"):
    cols = st.columns(min(len(items), 3))
    for col, d in zip(cols, items[:3]):
        with col:
            if d.get("image_url"):
                st.image(d["image_url"], use_container_width=True)
            st.markdown(f"**{d['product_name'][:70]}**")
            meta = [x for x in (d.get("price"), d.get("category")) if x]
            if meta:
                st.caption(" · ".join(meta))
            st.caption(f"{caption_score}: {d['score']:.3f}")


def _render_turn(turn: dict):
    with st.chat_message(turn["role"]):
        if turn.get("image") is not None:
            st.image(turn["image"], width=220)
        st.markdown(turn["content"])
        if turn.get("gate"):
            g = turn["gate"]
            if g["confident"]:
                st.success(f"Confident identification (p={g['confidence']:.2f})")
            else:
                st.warning(
                    f"Not confident (p={g['confidence']:.2f} < {g['threshold']:.2f}) — "
                    "showing the closest matches instead of committing to one."
                )
        if turn.get("products"):
            _render_products(turn["products"], turn.get("score_label", "match"))
        if turn.get("sources"):
            st.caption(f"**Sources:** {turn['sources']}")
        if turn.get("debug") and show_debug:
            with st.expander("Retrieval details"):
                st.json(turn["debug"])


# ───────────────────────────────────────────────────────── header

st.title("Multimodal Product Assistant")
st.caption(
    "Ask about products, ask to see one, or upload a photo to identify it. "
    "Answers are grounded in the catalog — the assistant declines rather than "
    "guessing when the catalog doesn't cover your question."
)

_catalog()

for turn in st.session_state.rendered:
    _render_turn(turn)


# ───────────────────────────────────────────────────────── input

uploaded = st.file_uploader(
    "Upload a product photo (optional)", type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)
prompt = st.chat_input("Ask about a product, or describe what you're looking for…")

if prompt or uploaded:
    if missing:
        st.error(f"Set {missing} in `.env` before asking.")
        st.stop()

    question = prompt or "Can you identify the product in this image and describe its usage?"
    image = Image.open(uploaded) if uploaded else None

    user_turn = {"role": "user", "content": question, "image": image}
    st.session_state.rendered.append(user_turn)
    _render_turn(user_turn)

    with st.chat_message("assistant"):
        # ── image upload ────────────────────────────────────────────────
        if image is not None:
            with st.spinner("Matching against the catalog…"):
                r = core.answer_image_question(image, question, model_key,
                                               history=_history())
            st.markdown(r["answer"])
            if r["confident"]:
                st.success(f"Confident identification (p={r['confidence']:.2f})")
            else:
                st.warning(
                    f"Not confident (p={r['confidence']:.2f} < {r['threshold']:.2f}) — "
                    "showing the closest matches instead of committing to one."
                )
            _render_products(r["docs"], "similarity")
            if r["sources"]:
                st.caption(f"**Sources:** {r['sources']}")
            turn = {"role": "assistant", "content": r["answer"], "sources": r["sources"],
                    "products": r["docs"], "score_label": "similarity",
                    "gate": {"confident": r["confident"], "confidence": r["confidence"],
                             "threshold": r["threshold"]},
                    "debug": {"confidence": r["confidence"],
                              "candidates": [d["product_name"] for d in r["docs"]]}}

        # ── "show me a picture of X" ────────────────────────────────────
        elif core.wants_a_picture(question):
            target = core.strip_image_request(question)
            with st.spinner("Finding product images…"):
                items = core.find_product_images(target, k=3)
            msg = (f"Here {'is' if len(items) == 1 else 'are'} the closest "
                   f"{'match' if len(items) == 1 else 'matches'} for **{target}**:")
            st.markdown(msg)
            _render_products(items, "similarity")
            st.caption("Shown by visual match against the catalog. "
                       "If none of these look right, the catalog may not carry it.")
            turn = {"role": "assistant", "content": msg, "products": items,
                    "score_label": "similarity",
                    "debug": {"image_search_query": target}}

        # ── text question ───────────────────────────────────────────────
        else:
            with st.spinner("Retrieving and answering…"):
                r = core.answer_text_question(question, model_key, use_mq,
                                              history=_history())
            st.markdown(r["answer"])
            if r["sources"] and "<None>" not in r["sources"]:
                st.caption(f"**Sources:** {r['sources']}")
                _render_products(r["docs"], "match")
            turn = {"role": "assistant", "content": r["answer"], "sources": r["sources"],
                    "products": r["docs"] if "<None>" not in r["sources"] else [],
                    "debug": {"queries": r["queries"],
                              "rewritten_query": r["rewritten_query"],
                              "retrieved": [d["product_name"] for d in r["docs"]]}}

        if turn.get("debug") and show_debug:
            with st.expander("Retrieval details"):
                st.json(turn["debug"])

    st.session_state.rendered.append(turn)
    st.session_state.messages.append(("user", question))
    st.session_state.messages.append(("assistant", turn["content"]))
