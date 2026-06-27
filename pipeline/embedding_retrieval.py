"""
Embedding retrieval stage - transformer-based semantic search.

Converts each knowledge entry into a dense sentence embedding using the
``sentence-transformers/all-MiniLM-L6-v2`` model, then matches a researcher's
free-text observation against that embedding space to find the most relevant
entries.

The public contract ``retrieve(observation) -> list[dict]`` is unchanged, so
all downstream stages (quality_ranking, graph_traversal, response_synthesis)
continue to work without modification.

Corpus embeddings are cached to disk as a ``.npz`` file and are regenerated
only when the dataset changes (detected via an MD5 fingerprint of entry IDs).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATA_DIR: Path = (
    Path(__file__).parent.parent / "experimental data" / "experimental data"
)
KNOWLEDGE_FILE: Path = DATA_DIR / "experiential_knowledge_41.json"
CACHE_FILE: Path = Path(__file__).parent / ".cache_corpus_embeddings.npz"
MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Knowledge base - loaded once at module import
# ---------------------------------------------------------------------------

KNOWLEDGE: List[dict] = json.loads(
    KNOWLEDGE_FILE.read_text(encoding="utf-8")
)["knowledge"]
KNOWLEDGE_BY_ID: dict = {entry["id"]: entry for entry in KNOWLEDGE}

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance, loading it on first call.

    The model is initialised lazily so that a warm-cache startup incurs
    zero model-load overhead.
    """
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Model loaded successfully.")
    return _model


# ---------------------------------------------------------------------------
# Corpus document construction
# ---------------------------------------------------------------------------


def _build_document(entry: dict) -> str:
    """Build the single combined text document used to embed a knowledge entry.

    Concatenates the following fields (when present):
    - title
    - category
    - trigger_condition  (list items joined with a space, or raw string)
    - knowledge
    - abstracted_pattern.pattern  (the IF/THEN pattern, if present)

    Parameters
    ----------
    entry : dict
        A single knowledge entry from the corpus.

    Returns
    -------
    str
        A single space-joined string ready for encoding.
    """
    parts: List[str] = []

    title = entry.get("title", "")
    if title:
        parts.append(title)

    category = entry.get("category", "")
    if category:
        parts.append(category)

    trigger_conditions = entry.get("trigger_condition", [])
    if isinstance(trigger_conditions, list):
        parts.append(" ".join(trigger_conditions))
    elif isinstance(trigger_conditions, str) and trigger_conditions:
        parts.append(trigger_conditions)

    knowledge = entry.get("knowledge", "")
    if knowledge:
        parts.append(knowledge)

    abstracted = entry.get("abstracted_pattern", {})
    if isinstance(abstracted, dict):
        pattern = abstracted.get("pattern", "")
        if pattern:
            parts.append(pattern)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _corpus_fingerprint() -> str:
    """Return an MD5 hex digest that uniquely identifies the current corpus.

    The digest is computed over the ordered list of entry IDs so that any
    addition, removal, or reordering of entries invalidates the cache.

    Returns
    -------
    str
        32-character lowercase hex string.
    """
    ids = [entry["id"] for entry in KNOWLEDGE]
    return hashlib.md5(json.dumps(ids).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public reusable functions
# ---------------------------------------------------------------------------


def build_corpus_embeddings() -> np.ndarray:
    """Build and return dense embeddings for every entry in the knowledge corpus.

    The result is cached to ``CACHE_FILE`` as a compressed ``.npz`` archive.
    The cache is reloaded automatically when the corpus fingerprint matches.
    The cache is rebuilt when the fingerprint does not match (dataset changed)
    or when the cache file is missing or corrupt.

    Returns
    -------
    np.ndarray
        Shape ``(N, D)`` float32 array.  N = number of knowledge entries,
        D = model embedding dimension (384 for all-MiniLM-L6-v2).
        Every row is L2-normalised so that dot-product equals cosine similarity.
    """
    fingerprint = _corpus_fingerprint()

    # --- attempt cache load ---
    if CACHE_FILE.exists():
        try:
            cached = np.load(str(CACHE_FILE), allow_pickle=False)
            cached_fp = str(cached["fingerprint"])
            if cached_fp == fingerprint:
                logger.info(
                    "Cache hit - loading corpus embeddings from %s", CACHE_FILE
                )
                return cached["embeddings"].astype(np.float32)
            logger.info(
                "Cache fingerprint mismatch - rebuilding corpus embeddings."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read cache (%s); rebuilding.", exc)

    # --- generate embeddings ---
    logger.info(
        "Generating embeddings for %d knowledge entries with model '%s'.",
        len(KNOWLEDGE),
        MODEL_NAME,
    )
    t0 = time.perf_counter()
    documents = [_build_document(entry) for entry in KNOWLEDGE]
    model = _get_model()
    embeddings: np.ndarray = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalise: dot-product == cosine similarity
        show_progress_bar=False,
    ).astype(np.float32)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Embedding generation complete: %d vectors in %.2f s.",
        len(embeddings),
        elapsed,
    )

    # --- write cache ---
    np.savez_compressed(
        str(CACHE_FILE),
        embeddings=embeddings,
        fingerprint=np.array(fingerprint),
        ids=np.array([entry["id"] for entry in KNOWLEDGE]),
    )
    logger.info("Corpus embeddings cached to %s", CACHE_FILE)

    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Encode a single researcher query into the same embedding space as the corpus.

    Parameters
    ----------
    query : str
        Free-text observation string from the researcher.

    Returns
    -------
    np.ndarray
        Shape ``(D,)`` L2-normalised float32 embedding vector.
    """
    logger.info("Embedding query: %.80s", query)
    model = _get_model()
    vec: np.ndarray = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return vec


def top_k_similar(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    k: int = 5,
) -> List[Tuple[str, float]]:
    """Return the top-k corpus entries most similar to the query by cosine similarity.

    Because both the query vector and every corpus row are L2-normalised,
    cosine similarity is computed as a plain matrix-vector dot product.

    Parameters
    ----------
    query_embedding : np.ndarray
        Shape ``(D,)`` query vector (L2-normalised).
    corpus_embeddings : np.ndarray
        Shape ``(N, D)`` corpus matrix (each row L2-normalised).
    k : int
        Number of results to return (default 5).

    Returns
    -------
    List[Tuple[str, float]]
        List of ``(entry_id, similarity_score)`` tuples sorted in descending
        order of similarity.
    """
    sims: np.ndarray = corpus_embeddings @ query_embedding  # shape (N,)
    top_indices = np.argsort(-sims)[:k]
    ids = [entry["id"] for entry in KNOWLEDGE]
    return [(ids[i], float(sims[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve(observation: str, k: int = 3) -> List[dict]:
    """Return the top-k candidate knowledge entries for the given observation.

    This is the primary entry point consumed by ``quality_ranking.rank()``.
    Each returned dict is a copy of the original knowledge-entry dict with an
    additional ``"similarity"`` key (float, rounded to 4 decimal places).

    Parameters
    ----------
    observation : str
        Free-text security observation from the researcher.
    k : int
        Number of candidates to return.  Defaults to 3 to match the contract
        expected by downstream callers.

    Returns
    -------
    List[dict]
        Candidate knowledge entries each augmented with ``"similarity"``.
    """
    t0 = time.perf_counter()

    corpus_embeddings = build_corpus_embeddings()
    query_embedding = embed_query(observation)
    matches = top_k_similar(query_embedding, corpus_embeddings, k=k)

    elapsed = time.perf_counter() - t0
    if matches:
        logger.info(
            "Retrieval complete in %.3f s - top match: %s (score %.4f)",
            elapsed,
            matches[0][0],
            matches[0][1],
        )

    return [
        {**KNOWLEDGE_BY_ID[entry_id], "similarity": round(score, 4)}
        for entry_id, score in matches
    ]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Run the prescribed self-test query and print Top-5 results with scores.

    Exercises the complete pipeline path:
      corpus embedding (or cache load) -> query encoding ->
      cosine-similarity ranking -> result formatting.
    """
    query = (
        "The application behaves differently when whitespace is added to HTTP headers."
    )
    separator = "=" * 72
    print("\n" + separator)
    print("Self-test query:")
    print("  " + repr(query))
    print(separator)

    results = retrieve(query, k=5)

    header = "{:<6} {:<12} {:<8}  {}".format("Rank", "ID", "Score", "Title")
    print("\n" + header)
    print("-" * 72)
    for rank_pos, entry in enumerate(results, start=1):
        title = str(entry.get("title", ""))[:52]
        score = entry["similarity"]
        eid = entry["id"]
        print("{:<6} {:<12} {:<8.4f}  {}".format(rank_pos, eid, score, title))
    print(separator + "\n")


if __name__ == "__main__":
    _self_test()
