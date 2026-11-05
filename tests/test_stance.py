"""Integration Test for Module 4: Stance Classification with Two-Tier Routing.

Pipeline: Module 2 (Retrieve) -> Module 3 (Select Rationales) -> Module 4 (Predict Stance).
Evaluates first 5 valid dev claims against ground-truth labels from claims_dev.jsonl.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval import SciFactRetriever
from src.rationale_selection import RationaleSelector
from src.stance import StanceClassifier
from src.utils import get_data_path, load_jsonl


def test_stance_classification():
    print("=" * 80)
    print("LitScope-Lite: Module 4 — Stance Classifier (Two-Tier Routing)")
    print("=" * 80)

    dev_path = get_data_path("claims_dev.jsonl")
    all_claims = load_jsonl(dev_path)
    valid_claims = [c for c in all_claims if c.get("evidence")]
    test_claims = valid_claims[:5]
    print(f"Evaluating on {len(test_claims)} valid dev claims.\n")

    print("Initializing pipeline: SciFactRetriever + RationaleSelector + StanceClassifier...")
    retriever = SciFactRetriever()
    selector = RationaleSelector(model=retriever.dense_model, threshold=0.40, top_k=3)
    
    # Initialize the updated two-tier classifier
    classifier = StanceClassifier(confidence_threshold=0.60)
    print("Pipeline ready!\n")

    for i, claim_item in enumerate(test_claims, start=1):
        claim_id = claim_item["id"]
        claim_text = claim_item["claim"]
        gt_evidence = claim_item.get("evidence", {})

        print("=" * 80)
        print(f"CLAIM #{i} [ID: {claim_id}]: {claim_text}")
        print("-" * 80)
        print("Ground Truth:")
        for doc_id_str, ev_list in gt_evidence.items():
            for ev in ev_list:
                print(f"  Doc {doc_id_str} | {ev['label']} | Sentences {ev['sentences']}")

        retrieved_docs = retriever.retrieve_top_k(claim_text, k=3)
        print("\nRetrieved Candidate Stance Predictions:")

        for r_idx, doc in enumerate(retrieved_docs, start=1):
            doc_id = doc["doc_id"]
            is_gold = str(doc_id) in gt_evidence
            gold_label = None
            if is_gold:
                gold_label = gt_evidence[str(doc_id)][0].get("label", "N/A")

            rationale_result = selector.select_rationales_for_doc(
                claim=claim_text, doc=doc, threshold=0.40, top_k=3
            )
            rationale_text = " ".join(rationale_result["rationales"]).strip()
            
            # Predict stance and capture the new model_used key
            pred = classifier.predict_stance(claim_text, rationale_text)
            pl = pred["label"]
            conf = pred["confidence"]
            probs = pred["probabilities"]
            model_used = pred["model_used"]

            gold_tag = f"[GOLD: {gold_label}]" if is_gold else "[DISTRACTOR]"
            match = "✓" if (is_gold and pl == gold_label) else ("✗" if is_gold else " ")

            print(f"\n  #{r_idx} Doc {doc_id} {gold_tag}")
            print(f"    Selected Rationale Sentences: {rationale_result['sentence_indices']}")
            for s_idx, s_text in zip(rationale_result["sentence_indices"], rationale_result["rationales"]):
                print(f"      [Sent #{s_idx}]: \"{s_text}\"")
            print(f"    Predicted: {pl} (conf={conf:.4f}) {match}")
            print(f"    Model Used: {model_used}")
            print(f"    Probabilities -> SUPPORT: {probs['SUPPORT']:.4f} | CONTRADICT: {probs['CONTRADICT']:.4f} | NOT_ENOUGH_INFO: {probs['NOT_ENOUGH_INFO']:.4f}")

        print()

    print("=" * 80)
    print("Stance classification integration test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_stance_classification()