from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable


_token_re = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _token_re.findall(text.lower())


@dataclass(frozen=True)
class TfidfModel:
    idf: dict[str, float]
    doc_vectors: dict[int, dict[str, float]]  # doc_id -> sparse vector
    doc_norms: dict[int, float]


def _l2_norm(vec: dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in vec.values()))


def _dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def cosine_sim(a: dict[str, float], a_norm: float, b: dict[str, float], b_norm: float) -> float:
    if a_norm <= 0.0 or b_norm <= 0.0:
        return 0.0
    return _dot(a, b) / (a_norm * b_norm)


def fit_tfidf(docs: dict[int, str]) -> TfidfModel:
    """
    Simple TF-IDF: tf = 1 + log(count), idf = log((N+1)/(df+1)) + 1.
    """
    n = len(docs)
    df: dict[str, int] = {}
    tokenized: dict[int, list[str]] = {}
    for doc_id, text in docs.items():
        toks = tokenize(text)
        tokenized[doc_id] = toks
        seen = set(toks)
        for t in seen:
            df[t] = df.get(t, 0) + 1

    idf = {t: (math.log((n + 1) / (c + 1)) + 1.0) for t, c in df.items()}

    doc_vectors: dict[int, dict[str, float]] = {}
    doc_norms: dict[int, float] = {}
    for doc_id, toks in tokenized.items():
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        vec: dict[str, float] = {}
        for t, c in tf.items():
            w_tf = 1.0 + math.log(c)
            vec[t] = w_tf * idf.get(t, 0.0)
        norm = _l2_norm(vec)
        doc_vectors[doc_id] = vec
        doc_norms[doc_id] = norm

    return TfidfModel(idf=idf, doc_vectors=doc_vectors, doc_norms=doc_norms)


def build_profile(
    model: TfidfModel, liked_docs: Iterable[tuple[int, float]]
) -> tuple[dict[str, float], float]:
    """
    Builds a weighted user profile vector from (doc_id, weight) liked items.
    """
    profile: dict[str, float] = {}
    total_w = 0.0
    for doc_id, w in liked_docs:
        vec = model.doc_vectors.get(doc_id)
        if not vec:
            continue
        total_w += w
        for t, v in vec.items():
            profile[t] = profile.get(t, 0.0) + w * v
    if total_w > 0.0:
        inv = 1.0 / total_w
        for t in list(profile.keys()):
            profile[t] *= inv
    return profile, _l2_norm(profile)

