"""Utility functions for LitScope-Lite."""

import json
import os
from pathlib import Path
from typing import Any, Iterable, Union
import numpy as np


def get_project_root() -> Path:
    """Find the project root directory containing data or configs."""
    # Check current working directory and parent paths
    cwd = Path.cwd().resolve()
    for candidate in [cwd, cwd / "LitScope-Lite", *cwd.parents]:
        if (candidate / "data" / "scifact").exists() or (candidate / "LitScope-Lite" / "data" / "scifact").exists():
            if (candidate / "data" / "scifact").exists():
                return candidate
            if (candidate / "LitScope-Lite" / "data" / "scifact").exists():
                return candidate / "LitScope-Lite"
    # Fallback to the directory containing src
    return Path(__file__).resolve().parent.parent


def get_data_path(filename: str, dataset_subdir: str = "data/scifact") -> Path:
    """Resolve data file path for SciFact dataset.
    
    Args:
        filename: Name of the file (e.g. 'claims_train.jsonl', 'claims_dev.jsonl', 'corpus.jsonl')
        dataset_subdir: Relative subdirectory path to the SciFact dataset.
        
    Returns:
        Path to the requested data file.
    """
    root = get_project_root()
    path = root / dataset_subdir / filename
    if not path.exists():
        # Fallback check relative to cwd
        cwd_path = Path.cwd() / dataset_subdir / filename
        if cwd_path.exists():
            return cwd_path
        # Check subfolder scifact
        alt_path = root / "data" / filename
        if alt_path.exists():
            return alt_path
    return path


def load_jsonl(filepath: Union[str, Path]) -> list[dict[str, Any]]:
    """Load records from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file.
        
    Returns:
        List of deserialized JSON objects.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSONL file not found at: {filepath}")
    
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                records.append(json.loads(line_str))
            except json.JSONDecodeError as err:
                raise ValueError(f"Error parsing JSON on line {line_num} in {filepath}: {err}") from err
    return records


def save_jsonl(filepath: Union[str, Path], records: Iterable[dict[str, Any]]) -> None:
    """Save records to a JSONL file.
    
    Args:
        filepath: Output path.
        records: Iterable of dictionaries to serialize.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def min_max_scale(scores: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Perform min-max normalization on an array of scores.
    
    Args:
        scores: 1D numpy array of numerical scores.
        eps: Small epsilon to prevent division by zero.
        
    Returns:
        Normalized array with values scaled to [0.0, 1.0].
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    min_val = np.min(scores)
    max_val = np.max(scores)
    range_val = max_val - min_val
    if range_val > eps:
        return (scores - min_val) / range_val
    return np.zeros_like(scores)


def compute_recall_at_k(
    retrieved_doc_ids: list[Union[int, str]],
    ground_truth_doc_ids: Iterable[Union[int, str]],
    k: int = 5,
) -> float:
    """Compute Recall@k for retrieved document IDs against ground truth document IDs.
    
    Args:
        retrieved_doc_ids: Ordered list of retrieved document IDs (best first).
        ground_truth_doc_ids: Set/list of ground truth relevant document IDs.
        k: Cutoff rank for evaluation.
        
    Returns:
        Recall@k score as a float between 0.0 and 1.0.
    """
    gt_set = {int(doc_id) for doc_id in ground_truth_doc_ids}
    if not gt_set:
        return 0.0
    top_k_retrieved = {int(doc_id) for doc_id in retrieved_doc_ids[:k]}
    hits = len(gt_set.intersection(top_k_retrieved))
    return hits / len(gt_set)

