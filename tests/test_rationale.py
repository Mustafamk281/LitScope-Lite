"""Integration Test for Module 3: Sentence-Level Evidence / Rationale Selection.

Loads top retrieved documents from Module 2 (SciFactRetriever) for the first 5 valid
claims in data/scifact/claims_dev.jsonl, runs RationaleSelector, and compares selected
rationales with ground-truth evidence sentences.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output encoding in Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval import SciFactRetriever
from src.rationale_selection import RationaleSelector
from src.utils import get_data_path, load_jsonl


def test_rationale_selection():
    print("=" * 80)
    print("LitScope-Lite: Integration Test — Rationale / Evidence Sentence Selection")
    print("=" * 80)

    # 1. Load dev claims
    dev_path = get_data_path("claims_dev.jsonl")
    all_claims = load_jsonl(dev_path)
    valid_claims = [c for c in all_claims if c.get("evidence") and len(c["evidence"]) > 0]
    test_claims = valid_claims[:5]

    print(f"Loaded {len(test_claims)} valid claims for rationale selection evaluation.\n")

    # 2. Initialize Retriever and RationaleSelector
    print("Initializing SciFactRetriever & RationaleSelector...")
    retriever = SciFactRetriever()
    # Reuse dense transformer instance for memory efficiency
    selector = RationaleSelector(model=retriever.dense_model, threshold=0.40, top_k=3)
    print("Modules initialized successfully!\n")

    # 3. Process each claim
    for i, claim_item in enumerate(test_claims, start=1):
        claim_id = claim_item.get("id")
        claim_text = claim_item.get("claim", "")
        gt_evidence = claim_item.get("evidence", {})

        print("=" * 80)
        print(f"CLAIM #{i} [ID: {claim_id}]:")
        print(f"  \"{claim_text}\"")
        print("-" * 80)

        # Ground Truth summary
        print("GROUND TRUTH EVIDENCE:")
        for doc_id_str, ev_list in gt_evidence.items():
            print(f"  Doc ID: {doc_id_str}")
            for ev in ev_list:
                label = ev.get("label", "N/A")
                sent_indices = ev.get("sentences", [])
                print(f"    Label: {label} | Sentence Indices: {sent_indices}")

        # Retrieve top-k documents
        retrieved_docs = retriever.retrieve_top_k(claim_text, k=3)
        print(f"\nTOP RETRIEVED CANDIDATES & SELECTED RATIONALES (threshold=0.40, top_k=3):")

        for r_idx, doc in enumerate(retrieved_docs, start=1):
            doc_id = doc["doc_id"]
            is_gt_doc = str(doc_id) in gt_evidence
            gt_tag = "[GOLD EVIDENCE DOC]" if is_gt_doc else "[DISTRACTOR DOC]"

            # Select rationales for this doc
            doc_rationale = selector.select_rationales_for_doc(
                claim=claim_text,
                doc=doc,
                threshold=0.40,
                top_k=3,
            )

            print(f"\n  Candidate #{r_idx}: Doc ID {doc_id} {gt_tag} (Score: {doc['score']:.4f})")
            print(f"    Title: {doc.get('title', 'N/A')}")
            print(f"    Selected Sentence Indices: {doc_rationale['sentence_indices']}")

            for s_info in doc_rationale["selected_sentences"]:
                s_idx = s_info["sentence_idx"]
                s_score = s_info["score"]
                s_text = s_info["sentence"]
                
                # Check if this sentence was part of gold evidence
                gold_match = False
                if is_gt_doc:
                    for ev in gt_evidence[str(doc_id)]:
                        if s_idx in ev.get("sentences", []):
                            gold_match = True
                            break
                match_tag = " -> [GOLD RATIONALE MATCH]" if gold_match else ""
                print(f"      [Sent #{s_idx} | Sim: {s_score:.4f}]: \"{s_text}\"{match_tag}")

        print()

    print("=" * 80)
    print("Rationale selection integration test finished successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_rationale_selection()
