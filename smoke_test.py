"""Smoke test: exercises every path the Streamlit app uses, without Streamlit.

Run after `prepare_artifacts.py` and before deploying:
    python smoke_test.py           # retrieval only, no API calls
    python smoke_test.py --llm     # also exercises the answer generation
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import rag_core as core  # noqa: E402

WITH_LLM = "--llm" in sys.argv
ok = True


def check(label: str, condition: bool, detail: str = "") -> None:
    global ok
    ok = ok and condition
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


print("=" * 72)
print("1. catalog")
print("=" * 72)
t0 = time.time()
cat = core.load_catalog()
load_s = time.time() - t0
print(f"  loaded in {load_s:.1f}s, {len(cat.text_ids):,} products, "
      f"alpha={cat.alpha}, gate threshold={cat.gate['threshold']:.2f}")
check("catalog size", len(cat.text_ids) == 10001)
check("fusion matrices aligned",
      cat._shared_text.shape[0] == cat._shared_image.shape[0] == len(cat._shared_ids),
      f"{len(cat._shared_ids):,} products with both modalities")
check("load under 10s", load_s < 10, f"{load_s:.1f}s")

print("\n" + "=" * 72)
print("2. text retrieval")
print("=" * 72)
hits = core.search_text("waterproof bluetooth speaker", k=3)
for d in hits:
    print(f"    {d['score']:.3f}  {d['product_name'][:58]}")
check("returns k results", len(hits) == 3)
check("hits carry image urls", all(h["image_url"].startswith("http") for h in hits))

print("\n" + "=" * 72)
print("3. 'show me a picture' routing  (brief interaction type 3)")
print("=" * 72)
cases = [
    ("Can you show me a picture of the Apple AirPods Pro?", True),
    ("show me a remote control car", True),
    ("What plush or stuffed animal toys do you have?", False),
    ("What is the price of the DB Longboards CoreFlex Crossbow?", False),
]
for q, expected in cases:
    got = core.wants_a_picture(q)
    check(f"route {'IMAGE' if expected else 'TEXT '}: {q[:46]}", got == expected)
print(f"  stripped: '{core.strip_image_request(cases[0][0])}'")
imgs = core.find_product_images("a remote control car", k=3)
for d in imgs:
    print(f"    {d['score']:.3f}  {d['product_name'][:58]}")
check("image search returns results", len(imgs) == 3)

print("\n" + "=" * 72)
print("4. image identification + confidence gate")
print("=" * 72)
qdir = Path(r"C:\Users\wasee\genai-final-data\cache\image_cache_query")
samples = sorted(qdir.glob("*.jpg"))[:12] if qdir.exists() else []
if not samples:
    print("  (held-out photos unavailable on this machine, skipping)")
else:
    n_correct = n_conf = 0
    t0 = time.time()
    for p in samples:
        r = core.identify_from_image(p)
        correct = r["candidates"][0]["uniq_id"] == p.stem
        n_correct += correct
        n_conf += r["confident"]
    dt = (time.time() - t0) / len(samples)
    print(f"  {len(samples)} held-out photos: top-1 correct {n_correct}/{len(samples)}, "
          f"gate answered {n_conf}/{len(samples)}, {dt:.2f}s per image")
    r = core.identify_from_image(samples[3])
    print(f"  example, true: {cat.name(samples[3].stem)[:52]}")
    print(f"    gate: {'ANSWER' if r['confident'] else 'DECLINE'} p={r['confidence']:.3f}")
    for i, d in enumerate(r["candidates"][:3], 1):
        mark = "<-- correct" if d["uniq_id"] == samples[3].stem else ""
        print(f"    {i}. {d['score']:.3f}  {d['product_name'][:48]} {mark}")
    check("per-image latency under 3s", dt < 3.0, f"{dt:.2f}s")
    check("gate produces a probability", 0.0 <= r["confidence"] <= 1.0)

if WITH_LLM:
    print("\n" + "=" * 72)
    print("5. answer generation (live API calls)")
    print("=" * 72)
    for model in ("gpt-4o-mini",):
        t0 = time.time()
        r = core.answer_text_question("What plush or stuffed animal toys do you have?",
                                      model_key=model, use_multiquery=False)
        print(f"  [{model}] {time.time()-t0:.1f}s")
        print(f"    {r['answer'][:180]}")
        print(f"    SOURCES: {r['sources'][:110]}")
        check("answered", len(r["answer"]) > 20)
        check("cited a source", "<None>" not in r["sources"])

        t0 = time.time()
        r2 = core.answer_text_question("Do you sell cars or auto parts?",
                                       model_key=model, use_multiquery=False)
        print(f"  refusal case ({time.time()-t0:.1f}s): {r2['answer'][:110]}")
        check("declined out-of-catalog", "<None>" in r2["sources"] or
              "don't have information" in r2["answer"].lower())

        hist = [("user", "What plush or stuffed animal toys do you have?"),
                ("assistant", r["answer"][:400])]
        rq = core._resolve_followup("what about the price of the first one?", hist,
                                    core.get_chat_model(model))
        print(f"  follow-up rewrite: 'what about the price of the first one?' -> '{rq[:80]}'")
        check("follow-up rewritten to standalone", rq.lower() != "what about the price of the first one?")
else:
    print("\n(skipping LLM generation; pass --llm to include it)")

print("\n" + "=" * 72)
print("ALL PASS" if ok else "FAILURES ABOVE")
print("=" * 72)
sys.exit(0 if ok else 1)
