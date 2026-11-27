"""Phase 2 Comprehensive Evaluation Script for LitScope-Lite.

Evaluates:
1. Module 2 (Hybrid Retrieval): Recall@K, Precision@K, MRR on SciFact dev set.
2. Module 4 (Stance Classification & Two-Tier Routing):
   - Oracle-rationale evaluation (Base-only vs. Two-Tier vs. Large-only)
   - Confidence threshold sweep (escalation rate vs. accuracy/macro-F1/latency)
3. End-to-End Pipeline (Modules 1-4): Retrieved-rationale verification on dev claims.
4. Proposal Research Checks:
   - Controlled claim edits (negation flip & paraphrase consistency)
   - Cross-source agreement/disagreement inspection
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.retrieval import SciFactRetriever
from src.rationale_selection import RationaleSelector
from src.stance import StanceClassifier


DATA_DIR = Path("data/scifact")
OUT_DIR = Path("outputs/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_retrieval(retriever: SciFactRetriever, dev_claims: List[Dict[str, Any]], ks=(1, 3, 5, 10)):
    print("\n[1/4] Evaluating Hybrid Retrieval on SciFact dev set...")
    max_k = max(ks)
    recall_hits = {k: [] for k in ks}
    precision_scores = {k: [] for k in ks}
    mrr_scores = []

    claims_with_gold = [c for c in dev_claims if c.get("evidence")]
    t0 = time.perf_counter()

    for idx, item in enumerate(claims_with_gold, 1):
        claim = item["claim"]
        gold_docs = {int(doc_id) for doc_id in item["evidence"].keys()}
        retrieved = retriever.retrieve_top_k(claim, k=max_k)
        retrieved_ids = [int(d["doc_id"]) for d in retrieved]

        # MRR
        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in gold_docs:
                rr = 1.0 / rank
                break
        mrr_scores.append(rr)

        for k in ks:
            top_k_ids = set(retrieved_ids[:k])
            hits = len(gold_docs & top_k_ids)
            recall_hits[k].append(hits / len(gold_docs))
            precision_scores[k].append(hits / k)

        if idx % 50 == 0 or idx == len(claims_with_gold):
            print(f"  Processed {idx}/{len(claims_with_gold)} claims...")

    elapsed = time.perf_counter() - t0
    summary = {
        "num_dev_claims_total": len(dev_claims),
        "num_dev_claims_with_gold_evidence": len(claims_with_gold),
        "mrr_at_10": round(float(np.mean(mrr_scores)), 4),
        "elapsed_seconds": round(elapsed, 2),
        "metrics_by_k": {
            str(k): {
                "recall": round(float(np.mean(recall_hits[k])), 4),
                "precision": round(float(np.mean(precision_scores[k])), 4),
            }
            for k in ks
        },
    }
    print("  Retrieval Summary:", json.dumps(summary["metrics_by_k"], indent=2))
    print(f"  MRR@10: {summary['mrr_at_10']} ({summary['elapsed_seconds']}s)")
    return summary


def evaluate_stance_and_routing(classifier: StanceClassifier, dev_claims: List[Dict[str, Any]], corpus_map: Dict[int, Dict[str, Any]]):
    print("\n[2/4] Evaluating Stance Classification & Two-Tier Routing on gold dev rationales...")
    rows = []

    # Build gold (claim, rationale, label) pairs
    pairs = []
    for item in dev_claims:
        claim_id = item["id"]
        claim = item["claim"]
        for doc_id_str, ev_list in item.get("evidence", {}).items():
            doc_id = int(doc_id_str)
            doc = corpus_map.get(doc_id)
            if not doc:
                continue
            abstract_sents = doc.get("abstract", [])
            for ev in ev_list:
                label = ev["label"]
                sent_indices = [i for i in ev["sentences"] if 0 <= i < len(abstract_sents)]
                if not sent_indices:
                    continue
                rationale = " ".join(abstract_sents[i] for i in sent_indices).strip()
                pairs.append((claim_id, doc_id, claim, rationale, label))

    print(f"  Found {len(pairs)} gold (claim, rationale) pairs in dev set.")

    for idx, (claim_id, doc_id, claim, rationale, gold_label) in enumerate(pairs, 1):
        t_b0 = time.perf_counter()
        base_probs, base_pred = classifier._get_probs(classifier.base_model, classifier.base_tokenizer, claim, rationale)
        base_ms = (time.perf_counter() - t_b0) * 1000.0
        base_conf = base_probs[base_pred]

        t_l0 = time.perf_counter()
        large_probs, large_pred = classifier._get_probs(classifier.large_model, classifier.large_tokenizer, claim, rationale)
        large_ms = (time.perf_counter() - t_l0) * 1000.0
        large_conf = large_probs[large_pred]

        rows.append({
            "claim_id": claim_id,
            "doc_id": doc_id,
            "claim": claim,
            "rationale": rationale,
            "gold_label": gold_label,
            "base_pred": base_pred,
            "base_conf": round(base_conf, 4),
            "base_ms": round(base_ms, 2),
            "large_pred": large_pred,
            "large_conf": round(large_conf, 4),
            "large_ms": round(large_ms, 2),
        })

        if idx % 50 == 0 or idx == len(pairs):
            print(f"  Classified {idx}/{len(pairs)} pairs...")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "stance_oracle_predictions.csv", index=False)

    # Sweep thresholds for two-tier routing
    thresholds = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.01]
    sweep_results = []
    labels_2class = ["SUPPORT", "CONTRADICT"]

    for tau in thresholds:
        escalated = df["base_conf"] < tau
        preds = np.where(escalated, df["large_pred"], df["base_pred"])
        latencies = np.where(escalated, df["base_ms"] + df["large_ms"], df["base_ms"])

        acc_all = accuracy_score(df["gold_label"], preds)
        f1_macro_2 = f1_score(df["gold_label"], preds, labels=labels_2class, average="macro", zero_division=0)
        nei_rate = float(np.mean(preds == "NOT_ENOUGH_INFO"))
        esc_rate = float(np.mean(escalated))

        sweep_results.append({
            "threshold": tau,
            "mode": "base_only" if tau == 0.0 else ("large_only" if tau > 1.0 else f"two_tier_{tau:.2f}"),
            "accuracy": round(float(acc_all), 4),
            "macro_f1_support_contradict": round(float(f1_macro_2), 4),
            "escalation_rate": round(esc_rate, 4),
            "nei_abstention_rate": round(nei_rate, 4),
            "avg_latency_ms": round(float(np.mean(latencies)), 2),
        })

    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(OUT_DIR / "routing_threshold_sweep.csv", index=False)

    # Detailed report at default threshold 0.60
    esc_60 = df["base_conf"] < 0.60
    preds_60 = np.where(esc_60, df["large_pred"], df["base_pred"])
    cls_report = classification_report(
        df["gold_label"], preds_60,
        labels=["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"],
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(
        df["gold_label"], preds_60,
        labels=["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]
    ).tolist()

    print("\n  Routing Sweep Preview:")
    print(sweep_df[["mode", "accuracy", "macro_f1_support_contradict", "escalation_rate", "avg_latency_ms"]].to_string(index=False))

    return {
        "num_gold_pairs": len(df),
        "default_threshold": 0.60,
        "classification_report_at_0.60": cls_report,
        "confusion_matrix_labels": ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"],
        "confusion_matrix_at_0.60": cm,
        "threshold_sweep": sweep_results,
    }


def evaluate_end_to_end_sample(
    retriever: SciFactRetriever,
    selector: RationaleSelector,
    classifier: StanceClassifier,
    dev_claims: List[Dict[str, Any]],
    sample_size: int = 60,
):
    print(f"\n[3/4] Evaluating End-to-End Pipeline (Retrieval -> Rationale -> Stance) on {sample_size} dev claims...")
    claims_with_gold = [c for c in dev_claims if c.get("evidence")][:sample_size]
    records = []

    for idx, item in enumerate(claims_with_gold, 1):
        claim = item["claim"]
        gold_labels = [ev["label"] for ev_list in item["evidence"].values() for ev in ev_list]
        gold_claim_label = max(set(gold_labels), key=gold_labels.count)
        gold_doc_ids = {int(d) for d in item["evidence"].keys()}

        retrieved_docs = retriever.retrieve_top_k(claim, k=3)
        ev_preds = []
        gold_doc_hit = False

        for doc in retrieved_docs:
            doc_id = int(doc["doc_id"])
            if doc_id in gold_doc_ids:
                gold_doc_hit = True
            rat = selector.select_rationales_for_doc(claim=claim, doc=doc, threshold=0.40, top_k=3)
            if not rat["rationales"]:
                continue
            rationale_text = " ".join(rat["rationales"]).strip()
            st = classifier.predict_stance(claim, rationale_text)
            ev_preds.append({
                "doc_id": doc_id,
                "label": st["label"],
                "confidence": st["confidence"],
                "model_used": st["model_used"],
            })

        decisive = [e for e in ev_preds if e["label"] in ("SUPPORT", "CONTRADICT")]
        if decisive:
            best = max(decisive, key=lambda x: x["confidence"])
            pred_label = best["label"]
            pred_conf = best["confidence"]
        else:
            pred_label = "NOT_ENOUGH_INFO"
            pred_conf = 1.0

        unique_decisive = {e["label"] for e in decisive}
        cross_source_conflict = ("SUPPORT" in unique_decisive and "CONTRADICT" in unique_decisive)

        records.append({
            "claim_id": item["id"],
            "claim": claim,
            "gold_label": gold_claim_label,
            "pred_label": pred_label,
            "pred_conf": round(float(pred_conf), 4),
            "gold_doc_retrieved_top3": gold_doc_hit,
            "num_docs_with_rationales": len(ev_preds),
            "cross_source_conflict": cross_source_conflict,
        })

        if idx % 20 == 0 or idx == len(claims_with_gold):
            print(f"  Completed {idx}/{len(claims_with_gold)} end-to-end claims...")

    e2e_df = pd.DataFrame(records)
    e2e_df.to_csv(OUT_DIR / "end_to_end_sample_predictions.csv", index=False)

    acc = accuracy_score(e2e_df["gold_label"], e2e_df["pred_label"])
    f1_m = f1_score(e2e_df["gold_label"], e2e_df["pred_label"], labels=["SUPPORT", "CONTRADICT"], average="macro", zero_division=0)
    summary = {
        "sample_size": len(e2e_df),
        "end_to_end_accuracy": round(float(acc), 4),
        "end_to_end_macro_f1_support_contradict": round(float(f1_m), 4),
        "gold_doc_recall_at_3": round(float(e2e_df["gold_doc_retrieved_top3"].mean()), 4),
        "nei_abstention_rate": round(float((e2e_df["pred_label"] == "NOT_ENOUGH_INFO").mean()), 4),
        "cross_source_conflict_cases": int(e2e_df["cross_source_conflict"].sum()),
    }
    print("  End-to-End Summary:", json.dumps(summary, indent=2))
    return summary


def evaluate_consistency_edits(classifier: StanceClassifier):
    print("\n[4/4] Running Controlled Claim-Edit Consistency Check...")
    test_cases = [
        (
            "Vitamin D deficiency is associated with an increased risk of osteoporosis.",
            "Vitamin D deficiency is not associated with an increased risk of osteoporosis.",
            "Low vitamin D levels are linked to a higher risk of developing osteoporosis.",
            "Multiple clinical studies demonstrate that vitamin D deficiency significantly increases the risk of osteoporosis and bone fractures."
        ),
        (
            "Aspirin reduces the risk of myocardial infarction in high-risk patients.",
            "Aspirin does not reduce the risk of myocardial infarction in high-risk patients.",
            "Taking aspirin lowers the likelihood of heart attacks among high-risk individuals.",
            "Randomized trials show daily low-dose aspirin significantly reduces the incidence of myocardial infarction in high-risk cardiovascular patients."
        ),
        (
            "Elevated blood glucose levels impair wound healing in diabetic patients.",
            "Elevated blood glucose levels improve wound healing in diabetic patients.",
            "High blood sugar slows down wound repair among people with diabetes.",
            "Hyperglycemia directly impairs angiogenesis and immune function, leading to delayed wound healing in patients with diabetes."
        ),
        (
            "Regular aerobic exercise lowers resting systolic blood pressure.",
            "Regular aerobic exercise increases resting systolic blood pressure.",
            "Consistent aerobic workouts reduce baseline systolic blood pressure.",
            "Meta-analyses confirm that structured aerobic exercise training produces clinically meaningful reductions in resting systolic blood pressure."
        ),
        (
            "Antibiotics are effective against viral respiratory infections.",
            "Antibiotics are ineffective against viral respiratory infections.",
            "Antibiotic drugs can cure respiratory infections caused by viruses.",
            "Antibiotics target bacterial pathways and have no therapeutic efficacy against viral respiratory tract infections."
        ),
    ]

    results = []
    paraphrase_consistent = 0
    negation_flipped = 0

    for orig, neg, para, rat in test_cases:
        p_orig = classifier.predict_stance(orig, rat)
        p_neg = classifier.predict_stance(neg, rat)
        p_para = classifier.predict_stance(para, rat)

        is_para_same = (p_orig["label"] == p_para["label"])
        is_neg_flip = (
            (p_orig["label"] == "SUPPORT" and p_neg["label"] == "CONTRADICT")
            or (p_orig["label"] == "CONTRADICT" and p_neg["label"] == "SUPPORT")
        )
        paraphrase_consistent += int(is_para_same)
        negation_flipped += int(is_neg_flip)

        results.append({
            "original_claim": orig,
            "orig_label": p_orig["label"],
            "orig_conf": round(p_orig["confidence"], 4),
            "paraphrase_label": p_para["label"],
            "paraphrase_consistent": is_para_same,
            "negated_label": p_neg["label"],
            "negation_flipped": is_neg_flip,
        })

    pd.DataFrame(results).to_csv(OUT_DIR / "consistency_check.csv", index=False)
    summary = {
        "num_cases": len(test_cases),
        "paraphrase_consistency_rate": round(paraphrase_consistent / len(test_cases), 4),
        "negation_flip_rate": round(negation_flipped / len(test_cases), 4),
        "cases": results,
    }
    print(f"  Paraphrase consistency: {paraphrase_consistent}/{len(test_cases)} | Negation flip: {negation_flipped}/{len(test_cases)}")
    return summary


def main():
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    corpus_map = {int(d["doc_id"]): d for d in corpus}
    dev_claims = load_jsonl(DATA_DIR / "claims_dev.jsonl")

    print("Initializing models once for Phase 2 evaluation...")
    retriever = SciFactRetriever()
    selector = RationaleSelector(model=retriever.dense_model, threshold=0.40, top_k=3)
    classifier = StanceClassifier(confidence_threshold=0.60)

    retrieval_summary = evaluate_retrieval(retriever, dev_claims)
    stance_summary = evaluate_stance_and_routing(classifier, dev_claims, corpus_map)
    e2e_summary = evaluate_end_to_end_sample(retriever, selector, classifier, dev_claims, sample_size=60)
    consistency_summary = evaluate_consistency_edits(classifier)

    full_report = {
        "retrieval": retrieval_summary,
        "stance_and_routing": stance_summary,
        "end_to_end_sample": e2e_summary,
        "consistency_edits": consistency_summary,
    }
    with open(OUT_DIR / "phase2_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nDONE! All Phase 2 evaluation artifacts saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()