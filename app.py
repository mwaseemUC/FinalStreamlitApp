"""Multimodal product-support assistant — Streamlit UI.

Three interaction types, matching the project brief:
  1. text question              -> grounded answer with cited sources
  2. "show me a picture of X"   -> product photographs
  3. image upload               -> identification, confidence-gated

Design note: the image path deliberately presents *candidates* rather than
asserting one identity. Measured on held-out photos, the top result is right
59% of the time while the top three contain the answer 71% of the time, so a
short ranked list is an honest surface for the accuracy the system actually has.
See CARDS_SHOWN for why three.
"""
from __future__ import annotations

import os
import re

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
    """Load the text tower at boot rather than on the first question.

    Streamlit Cloud boots the app once and keeps it warm, so paying ~0.6GB of
    model download during startup is far better than making the first user wait
    for it mid-question. The ViT-L/14 image tower is deliberately NOT warmed --
    it is 1.7GB and a text-only session never needs it.
    """
    core.embed_text("warmup")
    return True


def _missing_keys(model_key: str) -> str | None:
    provider = core.MODELS[model_key]["provider"]
    var = "GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY"
    return None if os.getenv(var) else var


# ───────────────────────────────────────────────────────── sidebar

st.sidebar.title("🛍️ Product Assistant")
st.sidebar.caption("CLIP retrieval + grounded LLM answers over 10,001 Amazon products")

# Default to the open-weights model: it scores highest on our evaluation rubric
# (13/14 vs 12/14) and satisfies the project's open-source LLM requirement.
# It is slower (~8-10s vs ~1.4s), so gpt-4o-mini is offered as a fast fallback.
_MODEL_ORDER = ["llama-3.3-70b", "gpt-4o-mini", "llama-3.1-8b"]
model_key = st.sidebar.selectbox(
    "Answer model",
    [k for k in _MODEL_ORDER if k in core.MODELS],
    format_func=lambda k: core.MODELS[k]["label"],
    index=0,
    help="Llama 3.3 70B (default) scores highest on our evaluation, 13/14, and "
         "is open-weights. GPT-4o-mini scores 12/14 and answers ~6x faster if "
         "you need lower latency.",
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

*Image* — on photographs the system has never seen, the top match is right
**59%** of the time; the **three shown** contain it **71%** of the time.
The confidence gate answers ~65% of uploads at ~80% precision.

**~1 in 5 confident image answers is still wrong**, which is why three candidates
are shown rather than a single assertion."""
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


# How many product cards to show, and why.
#
# Measured on 7,172 held-out photos (ViT-L/14 + fusion), the chance the correct
# product is among the cards shown:
#     1 card  0.592     3 cards 0.714     5 cards 0.751
#     2 cards 0.676     4 cards 0.735    10 cards 0.799
# The gain flattens after the third card (+3.8 pts for the 3rd, +2.1 for the
# 4th, +1.7 for the 5th), so 3 is where extra clutter stops paying for itself.
#
# The LLM still receives 5 candidates as context -- more context helps it
# recover a match ranked 2-5 -- so the model reads more than the user sees.
# That split is deliberate, not an oversight.
CARDS_SHOWN = 3
LLM_CANDIDATES = 5

# Streamlit re-runs the whole script on every interaction and re-renders the
# entire transcript, so page weight is paid again on each keystroke-submit.
# Product photos dominate that weight, so they are requested as CDN thumbnails
# and old turns stop rendering images entirely.
FULL_RENDER_TURNS = 6          # recent turns keep their images


def _thumb(url: str, px: int = 200) -> str:
    """Ask Amazon's CDN for a resized image instead of the full-resolution one.

    Inserting ._SX<px>_ before the extension is an Amazon CDN convention and
    cuts roughly 80% of the bytes (43KB -> 8KB measured), which is the single
    biggest lever on how heavy the page feels.
    """
    if not url or "images-na.ssl-images-amazon.com" not in url and "media-amazon.com" not in url:
        return url
    return re.sub(r"\.(jpg|jpeg|png)$", rf"._SX{px}_.\1", url, flags=re.I)


def _render_products(items, caption_score: str = "match", limit: int = CARDS_SHOWN):
    """Product cards at a fixed width.

    Catalog images have wildly varying aspect ratios (banner art next to a boxed
    game), so filling the column makes one card tower over its neighbours. A
    fixed width keeps the row scannable.
    """
    items = items[:limit]
    if not items:
        return
    cols = st.columns(len(items))
    for col, d in zip(cols, items):
        with col:
            if d.get("image_url"):
                st.image(_thumb(d["image_url"], 260), width=170)
            st.markdown(f"**{d['product_name'][:60]}**")
            meta = [x for x in (d.get("price"), d.get("category")) if x]
            if meta:
                st.caption(" · ".join(meta))
            if d.get("score") is not None:
                st.caption(f"{caption_score}: {d['score']:.3f}")


def _render_ranked(items, top_caption: str, alt_heading: str,
                   score_label: str | None = "similarity"):
    """One headline result, the rest visibly secondary.

    Equal-sized cards read as equally likely answers. That is wrong for both
    paths that use this: an identification's top match is right 59% of the time
    while the next two add 12 points between them, and a "most expensive" answer
    has exactly one correct product with the others as context.

    score_label=None hides the numeric score, for cases where the ranking
    quantity (price) is already shown in the metadata line.
    """
    if not items:
        return
    top, others = items[0], items[1:CARDS_SHOWN]

    left, right = st.columns([1, 2], vertical_alignment="center")
    with left:
        if top.get("image_url"):
            st.image(_thumb(top["image_url"], 300), width=200)
    with right:
        st.markdown(f"**{top['product_name'][:90]}**")
        meta = [x for x in (top.get("price"), top.get("category")) if x]
        if meta:
            st.caption(" · ".join(meta))
        caption = top_caption
        if score_label and top.get("score") is not None:
            caption += f" · {score_label} {top['score']:.3f}"
        st.caption(caption)

    if others:
        st.caption(f"**{alt_heading}**")
        cols = st.columns(len(others) * 2)          # keep them visually smaller
        for col, d in zip(cols, others):
            with col:
                if d.get("image_url"):
                    st.image(_thumb(d["image_url"], 150), width=95)
                st.caption(d["product_name"][:38])
                bits = [d.get("price") or ""]
                if score_label and d.get("score") is not None:
                    bits.append(f"{d['score']:.3f}")
                st.caption(" · ".join(b for b in bits if b))


def _render_identification(cands, confident: bool):
    _render_ranked(
        cands,
        top_caption="Top match" if confident else "Closest match",
        alt_heading="Other close matches" if confident else "Could also be",
        score_label="similarity",
    )


def _render_turn(turn: dict, with_images: bool = True):
    """Render one transcript turn.

    with_images=False keeps the text but drops every photo. Older turns use it:
    Streamlit replays the whole transcript on each interaction, so a long
    conversation otherwise accumulates dozens of remote images that the browser
    re-lays-out every time, which is what makes typing feel laggy.
    """
    with st.chat_message(turn["role"]):
        if turn.get("image") is not None and with_images:
            st.image(turn["image"], width=220)
        st.markdown(turn["content"])
        if not with_images and (turn.get("products") or turn.get("image") is not None):
            st.caption("_(images hidden — scroll-back is trimmed to keep typing responsive)_")
            if turn.get("sources"):
                st.caption(f"**Sources:** {turn['sources']}")
            return
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
            layout = turn.get("layout")
            if layout == "identification":
                _render_identification(turn["products"],
                                       turn.get("gate", {}).get("confident", False))
            elif layout == "ranked":
                most = turn.get("aggregate") == "most_expensive"
                _render_ranked(
                    turn["products"],
                    top_caption="Most expensive" if most else "Least expensive",
                    alt_heading="Next most expensive" if most else "Next least expensive",
                    score_label=None,
                )
            else:
                _render_products(turn["products"], turn.get("score_label", "match"))
        if turn.get("sources"):
            st.caption(f"**Sources:** {turn['sources']}")
        if turn.get("debug") and show_debug:
            with st.expander("Retrieval details"):
                st.json(turn["debug"])


# ───────────────────────────────────────────────────────── header

st.title("Multimodal Product Assistant")
st.caption(
    "Ask about products, ask to see one, or attach a photo — use the 📎 in the "
    "message box, and type alongside it to ask something specific about the "
    "picture. Answers are grounded in the catalog: the assistant declines rather "
    "than guessing when the catalog doesn't cover your question."
)

if not st.session_state.get("rendered"):
    with st.expander("What can I ask?", expanded=True):
        st.markdown(
            "- **Text** — *“What's the price of the DB Longboards CoreFlex Crossbow?”*\n"
            "- **See a product** — *“Can you show me a picture of a remote control car?”*\n"
            "- **Attach a photo** (📎) — identifies the product from the catalog\n"
            "- **Photo + your own question** — *“is this suitable for a toddler?”* "
            "attached to an image, instead of the default identification\n"
            "- **Follow-ups work** — *“what about the price of the first one?”*"
        )

_catalog()
_warm_text_encoder()

_transcript = st.session_state.rendered
_cutoff = max(0, len(_transcript) - FULL_RENDER_TURNS)
if _cutoff:
    st.caption(f"_{_cutoff} earlier message(s) shown without images._")
for _i, turn in enumerate(_transcript):
    _render_turn(turn, with_images=_i >= _cutoff)


# ───────────────────────────────────────────────────────── input

# One chat-level control for everything: type a question, attach a photo, or
# both. Attaching without typing falls back to a default question, but the user
# can always override it by typing alongside the attachment.
submission = st.chat_input(
    "Ask about a product, attach a photo, or both…",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"],
)

if submission:
    if missing:
        st.error(f"Set {missing} in `.env` before asking.")
        st.stop()

    # ChatInputValue exposes .text and .files; stay tolerant of a plain string
    # so the app still works if the input widget is ever swapped back.
    text = (getattr(submission, "text", None) or "").strip() if not isinstance(
        submission, str) else submission.strip()
    files = list(getattr(submission, "files", []) or [])

    image = Image.open(files[0]) if files else None
    DEFAULT_IMAGE_Q = "Can you identify the product in this image and describe its usage?"
    question = text or (DEFAULT_IMAGE_Q if image is not None else "")
    if not question:
        st.stop()

    user_turn = {"role": "user", "content": question, "image": image}
    st.session_state.rendered.append(user_turn)
    _render_turn(user_turn)

    with st.chat_message("assistant"):
        # ── image attached (with or without an accompanying question) ────
        # A typed question is passed through verbatim, so "what colour is this?"
        # or "is this waterproof?" work on an uploaded photo, not just the
        # default "identify this product".
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
            _render_identification(r["docs"], r["confident"])
            if r["sources"]:
                st.caption(f"**Sources:** {r['sources']}")
            turn = {"role": "assistant", "content": r["answer"], "sources": r["sources"],
                    "products": r["docs"], "score_label": "similarity",
                    "layout": "identification",
                    "gate": {"confident": r["confident"], "confidence": r["confidence"],
                             "threshold": r["threshold"]},
                    "debug": {"confidence": r["confidence"],
                              "candidates": [d["product_name"] for d in r["docs"]]}}

        # ── "show me a picture of X" ────────────────────────────────────
        elif core.wants_a_picture(question):
            target = core.strip_image_request(question)
            with st.spinner("Finding product images…"):
                items = core.find_product_images(target, k=CARDS_SHOWN)
            msg = (f"Here {'is' if len(items) == 1 else 'are'} the closest "
                   f"{'match' if len(items) == 1 else 'matches'} for **{target}**:")
            st.markdown(msg)
            _render_products(items, "similarity", limit=CARDS_SHOWN)
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

            # Show the products the ANSWER cited, not everything retrieval
            # returned. Showing all 8 candidates next to an answer about one of
            # them reads as though the assistant picked the wrong product.
            cited = core.cited_ids(r["sources"])
            if cited:
                by_id = {d["uniq_id"]: d for d in r["docs"]}
                shown = [by_id[c] for c in cited if c in by_id]
            elif r.get("aggregate"):
                shown = r["docs"]        # computed answers show their own ranking
            else:
                shown = []

            if r["sources"] and "<None>" not in r["sources"]:
                st.caption(f"**Sources:** {r['sources']}")

            agg_kind = str(r.get("aggregate") or "")
            is_price_rank = agg_kind.endswith("expensive")
            layout = "ranked" if is_price_rank else "row"
            if shown:
                if is_price_rank:
                    # One product IS the answer; the rest are context. Price is
                    # already in the metadata line, so no numeric score.
                    _render_ranked(
                        shown,
                        top_caption=("Most expensive" if agg_kind == "most_expensive"
                                     else "Least expensive"),
                        alt_heading=("Next most expensive" if agg_kind == "most_expensive"
                                     else "Next least expensive"),
                        score_label=None,
                    )
                else:
                    _render_products(shown, "match", limit=CARDS_SHOWN)

            turn = {"role": "assistant", "content": r["answer"], "sources": r["sources"],
                    "products": shown, "layout": layout, "aggregate": agg_kind,
                    "score_label": "match",
                    "debug": {"queries": r["queries"],
                              "rewritten_query": r["rewritten_query"],
                              "aggregate": r.get("aggregate"),
                              "cited": cited,
                              "retrieved": [d["product_name"] for d in r["docs"]]}}

        if turn.get("debug") and show_debug:
            with st.expander("Retrieval details"):
                st.json(turn["debug"])

    st.session_state.rendered.append(turn)
    st.session_state.messages.append(("user", question))
    st.session_state.messages.append(("assistant", turn["content"]))
