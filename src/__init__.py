"""LitScope-Lite: NLP system for scientific claim verification using SciFact."""

from .segmentation import segment_claims
from .retrieval import SciFactRetriever
from .utils import (
    get_data_path,
    load_jsonl,
    save_jsonl,
    compute_recall_at_k,
    min_max_scale,
)

__all__ = [
    "segment_claims",
    "SciFactRetriever",
    "get_data_path",
    "load_jsonl",
    "save_jsonl",
    "compute_recall_at_k",
    "min_max_scale",
]

