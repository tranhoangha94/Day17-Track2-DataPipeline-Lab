"""Compare decontamination with two similarity backends (extension demo).

    python compare_embed_models.py

Defaults (zero-key, no pip extra):
  Model A = exact   — lab decontaminate (normalize + exact match)
  Model B = fuzzy   — SequenceMatcher (catches paraphrases)

Other options via env:
  EMBED_MODEL_A=hash
  EMBED_MODEL_B=tfidf
  EMBED_MODEL_B=sentence-transformers/all-MiniLM-L6-v2
  EMBED_MODEL_B=gemini-3.1-flash-lite   # needs GEMINI_API_KEY in .env
  EMBED_SIM_THRESHOLD=0.55
"""
from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import numpy as np

from pipeline.dataset import build_eval_set, build_preference_pairs, decontaminate
from pipeline.embed import embed_text as hash_embed

ROOT = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if val and key not in os.environ:
            os.environ[key] = val


_load_dotenv()

MODEL_A = os.getenv("EMBED_MODEL_A", "exact")
MODEL_B = os.getenv("EMBED_MODEL_B", "gemini-3.1-flash-lite")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
SIM_THRESHOLD = float(os.getenv("EMBED_SIM_THRESHOLD", "0.55"))


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


class TfidfBackend:
    def __init__(self, corpus: list[str]) -> None:
        self._vocab: dict[str, int] = {}
        docs = [re.findall(r"[a-z0-9]+", _norm(t)) for t in corpus]
        for doc in docs:
            for tok in doc:
                self._vocab.setdefault(tok, len(self._vocab))
        n = len(corpus)
        df = np.zeros(len(self._vocab))
        for doc in docs:
            for idx in {self._vocab[t] for t in doc}:
                df[idx] += 1
        self._idf = np.log((1 + n) / (1 + df)) + 1

    def vec(self, text: str) -> np.ndarray:
        toks = re.findall(r"[a-z0-9]+", _norm(text))
        out = np.zeros(len(self._vocab))
        if not toks:
            return out
        tf: dict[int, int] = {}
        for t in toks:
            tf[self._vocab[t]] = tf.get(self._vocab[t], 0) + 1
        for idx, cnt in tf.items():
            out[idx] = (cnt / len(toks)) * self._idf[idx]
        return out


class SentenceTransformerBackend:
    def __init__(self, name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = name
        self._model = SentenceTransformer(name)

    def sim(self, a: str, b: str) -> float:
        va, vb = self._model.encode([a, b], normalize_embeddings=True)
        return float(np.dot(va, vb))


class GeminiSemanticBackend:
    """Use Gemini 3.1 Flash Lite as a semantic-similarity judge (0–1)."""

    _SCHEMA = {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "description": "Semantic similarity from 0.0 (unrelated) to 1.0 (same intent).",
            }
        },
        "required": ["score"],
    }

    def __init__(self, model: str) -> None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")
        self.model = model
        self._client = genai.Client(api_key=api_key)
        self._cache: dict[tuple[str, str], float] = {}

    def sim(self, a: str, b: str) -> float:
        ka, kb = _norm(a), _norm(b)
        if ka == kb:
            return 1.0
        key = (ka, kb)
        if key in self._cache:
            return self._cache[key]
        rev = (kb, ka)
        if rev in self._cache:
            return self._cache[rev]

        prompt = (
            "You score whether two user prompts ask the same question for ML dataset "
            "decontamination. Return JSON only.\n"
            f"Prompt A: {a}\n"
            f"Prompt B: {b}\n"
            "Score 1.0 if same intent (paraphrases count), 0.0 if unrelated."
        )
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": self._SCHEMA,
                "temperature": 0,
            },
        )
        raw = json.loads(resp.text or "{}")
        score = float(max(0.0, min(1.0, raw["score"])))
        self._cache[key] = score
        return score


def make_sim_fn(name: str, corpus: list[str]) -> tuple[str, Callable[[str, str], float]]:
    key = name.strip().lower()

    if key == "exact":
        return "exact (lab default)", lambda a, b: 1.0 if _norm(a) == _norm(b) else 0.0

    if key == "hash":
        return "hash (token hash embed)", lambda a, b: _cos(
            np.asarray(hash_embed(a)), np.asarray(hash_embed(b))
        )

    if key == "fuzzy":
        return "fuzzy (SequenceMatcher)", lambda a, b: SequenceMatcher(
            None, _norm(a), _norm(b)
        ).ratio()

    if key == "tfidf":
        tf = TfidfBackend(corpus)
        return "tfidf (pure-python)", lambda a, b: _cos(tf.vec(a), tf.vec(b))

    if key.startswith("gemini") or name.startswith("gemini-"):
        model = GEMINI_MODEL if key == "gemini" else name
        gb = GeminiSemanticBackend(model)
        return f"gemini ({model})", gb.sim

    if key.startswith("sentence-transformers/"):
        st = SentenceTransformerBackend(name)
        return st.name, st.sim

    raise ValueError(
        f"Unknown model: {name!r}. Use exact, hash, fuzzy, tfidf, gemini-3.1-flash-lite, "
        "or sentence-transformers/..."
    )


def decontaminate_by_sim(
    pairs: list[dict],
    eval_set: list[dict],
    sim_fn: Callable[[str, str], float],
    threshold: float,
) -> tuple[list[dict], list[tuple[str, str, float]]]:
    eval_inputs = [e["input"] for e in eval_set]
    kept, dropped = [], []
    for pair in pairs:
        best_eval, best_sim = "", 0.0
        for inp in eval_inputs:
            sim = sim_fn(pair["prompt"], inp)
            if sim > best_sim:
                best_sim, best_eval = sim, inp
        if best_sim >= threshold:
            dropped.append((pair["prompt"], best_eval, best_sim))
        else:
            kept.append(pair)
    return kept, dropped


def _print_results(label: str, kept: list[dict], dropped: list[tuple[str, str, float]]) -> None:
    print(f"\n[{label}]")
    print(f"  kept={len(kept)}, dropped={len(dropped)}")
    for prompt, ev, sim in dropped:
        print(f"    drop sim={sim:.3f}")
        print(f"      pair : {prompt[:70]!r}")
        print(f"      eval : {ev[:70]!r}")


def main() -> None:
    import duckdb

    from pipeline.traces import load_traces, traces_to_bronze

    con = duckdb.connect(":memory:")
    traces_to_bronze(con, load_traces())
    eval_set = build_eval_set(con)
    pairs = build_preference_pairs(con)
    con.close()

    paraphrase_pair = {
        "prompt": "is a widget bought ten days ago eligible for return?",
        "chosen": "Yes, within 30 days.",
        "rejected": "No returns ever.",
    }
    test_pairs = pairs + [paraphrase_pair]
    corpus = [e["input"] for e in eval_set] + [p["prompt"] for p in test_pairs]
    exact_clean = decontaminate(pairs, eval_set)

    print("=== Compare 2 models for decontamination ===\n")
    print(f"Eval rows           : {len(eval_set)}")
    print(f"Preference pairs    : {len(pairs)} (raw)")
    print(f"Exact-match (lab)   : {len(exact_clean)} kept / {len(pairs) - len(exact_clean)} dropped")
    print(f"Sim threshold       : {SIM_THRESHOLD}")
    print(f"Model A             : {MODEL_A}")
    print(f"Model B             : {MODEL_B}")

    try:
        label_a, sim_a = make_sim_fn(MODEL_A, corpus)
        label_b, sim_b = make_sim_fn(MODEL_B, corpus)
    except (ImportError, ValueError) as exc:
        print(f"\nError: {exc}")
        print("Gemini: pip install google-genai  +  set GEMINI_API_KEY in .env")
        print("ST:     pip install sentence-transformers")
        return

    kept_a, drop_a = decontaminate_by_sim(test_pairs, eval_set, sim_a, SIM_THRESHOLD)
    kept_b, drop_b = decontaminate_by_sim(test_pairs, eval_set, sim_b, SIM_THRESHOLD)
    _print_results(f"Model A — {label_a}", kept_a, drop_a)
    _print_results(f"Model B — {label_b}", kept_b, drop_b)

    eval0, p = eval_set[0]["input"], paraphrase_pair["prompt"]
    print("\n--- Paraphrase probe ---")
    print(f"  eval : {eval0!r}")
    print(f"  pair : {p!r}")
    print(f"  Model A sim : {sim_a(p, eval0):.3f}")
    print(f"  Model B sim : {sim_b(p, eval0):.3f}")

    only_b = {d[0] for d in drop_b} - {d[0] for d in drop_a}
    only_a = {d[0] for d in drop_a} - {d[0] for d in drop_b}
    if only_b:
        print(f"\n  Model B caught {len(only_b)} extra leak(s) vs A (incl. paraphrase).")
    if only_a:
        print(f"  Model A caught {len(only_a)} extra leak(s) vs B.")
    if not only_a and not only_b:
        print("\n  Same drop set at this threshold.")


if __name__ == "__main__":
    main()
