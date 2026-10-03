"""Validation Test for Module 2: SciFact Hybrid Evidence Retrieval.

Loads data/scifact/claims_dev.jsonl, takes the first 20 valid claims (with evidence annotations),
runs retrieve_top_k(claim, k=5), and evaluates Recall@5 against ground-truth evidence doc_ids.
"""

import os
import sys
from pathlib import Path

# Add src and project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval import SciFactRetriever
from src.utils import get_data_path, load_jsonl, compute_recall_at_k


def test_hybrid_retrieval_recall_at_5():
    """Run retrieval on the first 20 valid dev claims and compute Recall@5."""
    print("=" * 70)
    print("LitScope-Lite: Validating Hybrid Evidence Retrieval (Recall@5)")
    print("=" * 70)

    # 1. Load dev claims
    dev_path = get_data_path("claims_dev.jsonl")
    print(f"Loading claims from: {dev_path}")
    all_claims = load_jsonl(dev_path)

    # Filter for valid claims with evidence
    valid_claims = [c for c in all_claims if c.get("evidence") and len(c["evidence"]) > 0]
    eval_claims = valid_claims[:20]
    print(f"Total dev claims: {len(all_claims)}")
    print(f"Valid claims with evidence: {len(valid_claims)}")
    print(f"Evaluating on first {len(eval_claims)} valid claims.\n")

    # 2. Initialize Hybrid Retriever
    print("Initializing SciFactRetriever (TF-IDF + all-MiniLM-L6-v2 hybrid)...")
    retriever = SciFactRetriever()
    print("Retriever initialized successfully!\n")

    # 3. Evaluate each claim
    recall_scores = []
    print(f"{'#':<3} | {'Claim ID':<8} | {'GT Doc IDs':<16} | {'Retrieved Top 5 Doc IDs':<28} | {'Recall@5':<8}")
    print("-" * 75)

    for idx, claim_item in enumerate(eval_claims, start=1):
        claim_id = claim_item.get("id")
        claim_text = claim_item.get("claim", "")
        gt_doc_ids = [int(doc_id) for doc_id in claim_item["evidence"].keys()]

        # Retrieve top 5
        results = retriever.retrieve_top_k(claim_text, k=5)
        retrieved_doc_ids = [res["doc_id"] for res in results]

        # Verify output structure
        assert len(results) == 5, f"Expected 5 retrieved docs, got {len(results)}"
        assert "doc_id" in results[0] and "score" in results[0], "Missing keys in retrieved item"

        # Compute recall
        recall = compute_recall_at_k(retrieved_doc_ids, gt_doc_ids, k=5)
        recall_scores.append(recall)

        gt_str = str(gt_doc_ids)
        ret_str = str(retrieved_doc_ids)
        print(f"{idx:<3} | {claim_id:<8} | {gt_str:<16} | {ret_str:<28} | {recall:.4f}")

    # 4. Compute average Recall@5
    mean_recall_at_5 = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    print("-" * 75)
    print(f"\nFinal Evaluation Results ({len(eval_claims)} claims):")
    print(f"  >>> Average Recall@5: {mean_recall_at_5:.4f} ({mean_recall_at_5 * 100:.2f}%) <<<")
    print("=" * 70)

    # Basic validity assertion
    assert mean_recall_at_5 > 0.5, f"Expected Recall@5 > 0.5, got {mean_recall_at_5:.4f}"
    return mean_recall_at_5


if __name__ == "__main__":
    test_hybrid_retrieval_recall_at_5()

