
"""Phase 2 comprehensive evaluation for LitScope-Lite."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

from src.recompute_routing_metrics import (
    summarize_routing,
    SELECTED_THRESHOLD,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "scifact"
OUT_DIR = ROOT / "outputs" / "evaluation"

SUMMARY_PATH = OUT_DIR / "phase2_evaluation_summary.json"

LABELS = ["SUPPORT", "CONTRADICT"]

RETRIEVAL_K = 3
RATIONALE_THRESHOLD = 0.40
RATIONALE_TOP_K = 3

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    """Load a JSON Lines dataset."""

    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def load_summary():
    """Preserve previously completed experiments."""

    if SUMMARY_PATH.exists():
        return json.loads(
            SUMMARY_PATH.read_text(encoding="utf-8")
        )

    return {}


def save_summary(summary):
    """Save evaluation summary."""

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def evaluate_retrieval(
    retriever,
    dev_claims,
    ks=(1, 3, 5, 10),
):
    """Evaluate hybrid document retrieval."""

    print("\n[1/4] Evaluating hybrid retrieval...")

    max_k = max(ks)

    recall_hits = {k: [] for k in ks}
    precision_scores = {k: [] for k in ks}
    hit_rates = {k: [] for k in ks}

    mrr_scores = []

    claims_with_gold = [
        c for c in dev_claims
        if c.get("evidence")
    ]

    if not claims_with_gold:
        raise ValueError("No gold-annotated claims found.")

    t0 = time.perf_counter()

    for idx, item in enumerate(claims_with_gold, 1):

        claim = item["claim"]

        gold_docs = {
            int(doc_id)
            for doc_id in item["evidence"]
        }

        retrieved = retriever.retrieve_top_k(
            claim,
            k=max_k,
        )

        retrieved_ids = [
            int(doc["doc_id"])
            for doc in retrieved
        ]

        reciprocal_rank = 0.0

        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in gold_docs:
                reciprocal_rank = 1.0 / rank
                break

        mrr_scores.append(reciprocal_rank)

        for k in ks:

            top_k_ids = set(retrieved_ids[:k])

            hits = len(gold_docs & top_k_ids)

            recall_hits[k].append(
                hits / len(gold_docs)
            )

            precision_scores[k].append(
                hits / k
            )

            hit_rates[k].append(
                int(hits > 0)
            )

        if idx % 50 == 0 or idx == len(claims_with_gold):
            print(
                f"  Processed "
                f"{idx}/{len(claims_with_gold)} claims"
            )

    elapsed = time.perf_counter() - t0

    summary = {
        "split": "dev",
        "retrieval_method": "hybrid",
        "num_dev_claims_total": len(dev_claims),
        "num_dev_claims_with_gold_evidence": (
            len(claims_with_gold)
        ),

        f"mrr_at_{max_k}": round(
            float(np.mean(mrr_scores)),
            4,
        ),

        "elapsed_seconds": round(elapsed, 2),

        "metrics_by_k": {
            str(k): {
                "recall": round(
                    float(np.mean(recall_hits[k])),
                    4,
                ),

                "precision": round(
                    float(np.mean(precision_scores[k])),
                    4,
                ),

                "hit_rate": round(
                    float(np.mean(hit_rates[k])),
                    4,
                ),
            }
            for k in ks
        },
    }

    print(json.dumps(summary, indent=2))

    return summary


def evaluate_stance_and_routing(
    classifier,
    dev_claims,
    corpus_map,
):
    """Evaluate both NLI models on gold rationales."""

    print("\n[2/4] Evaluating oracle stance classification...")

    pairs = []

    for item in dev_claims:

        claim_id = item["id"]
        claim = item["claim"]

        for doc_id_str, ev_list in item.get(
            "evidence", {}
        ).items():

            doc_id = int(doc_id_str)

            doc = corpus_map.get(doc_id)

            if not doc:
                continue

            abstract_sents = doc.get("abstract", [])

            for ev in ev_list:

                label = ev["label"]

                if label not in LABELS:
                    continue

                sent_indices = [
                    i for i in ev["sentences"]
                    if 0 <= i < len(abstract_sents)
                ]

                if not sent_indices:
                    continue

                rationale = " ".join(
                    abstract_sents[i]
                    for i in sent_indices
                ).strip()

                if not rationale:
                    continue

                pairs.append((
                    claim_id,
                    doc_id,
                    claim,
                    rationale,
                    label,
                ))

    if not pairs:
        raise ValueError("No valid gold rationale pairs.")

    print(f"  Found {len(pairs)} gold rationale pairs.")

    rows = []

    for idx, pair in enumerate(pairs, 1):

        claim_id, doc_id, claim, rationale, gold_label = pair

        # Smaller model inference
        t0 = time.perf_counter()

        base_probs, base_pred = classifier._get_probs(
            classifier.base_model,
            classifier.base_tokenizer,
            claim,
            rationale,
        )

        base_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        # Larger model inference
        t0 = time.perf_counter()

        large_probs, large_pred = classifier._get_probs(
            classifier.large_model,
            classifier.large_tokenizer,
            claim,
            rationale,
        )

        large_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        rows.append({
            "claim_id": claim_id,
            "doc_id": doc_id,
            "claim": claim,
            "rationale": rationale,
            "gold_label": gold_label,
            "base_pred": base_pred,
            "base_conf": base_probs[base_pred],
            "base_ms": base_ms,
            "large_pred": large_pred,
            "large_conf": large_probs[large_pred],
            "large_ms": large_ms,
        })

        if idx % 50 == 0 or idx == len(pairs):
            print(
                f"  Classified {idx}/{len(pairs)} pairs"
            )

    df = pd.DataFrame(rows)

    # Shared corrected evaluation implementation.
    sweep_df, summary = summarize_routing(df)

    df.to_csv(
        OUT_DIR / "stance_oracle_predictions.csv",
        index=False,
    )

    sweep_df.to_csv(
        OUT_DIR / "routing_threshold_sweep.csv",
        index=False,
    )

    summary["split"] = "dev"
    summary["device"] = str(classifier.device)

    print("\nRouting sweep:")

    print(
        sweep_df[[
            "mode",
            "accuracy",
            "macro_f1_support_contradict",
            "large_model_usage_rate",
            "avg_latency_ms",
        ]].to_string(index=False)
    )

    return summary


def summarize_end_to_end(df):
    """Calculate claim-level evaluation metrics."""

    if df.empty:
        raise ValueError("No end-to-end predictions found.")

    gold = df["gold_label"]
    preds = df["pred_label"]

    correct = gold == preds

    decisive = preds.isin(LABELS)

    # A hit means at least one annotated document
    # was retrieved, not necessarily all gold documents.
    gold_hit = df["gold_doc_retrieved_top3"].eq(True)

    joint_correct = correct & gold_hit

    return {
        "sample_size": len(df),

        "end_to_end_accuracy": round(
            float(accuracy_score(gold, preds)),
            4,
        ),

        "end_to_end_macro_f1_support_contradict": round(
            float(
                f1_score(
                    gold,
                    preds,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),

        # Retained for backward compatibility.
        "gold_doc_recall_at_3": round(
            float(gold_hit.mean()),
            4,
        ),

        "gold_doc_hit_rate_at_3": round(
            float(gold_hit.mean()),
            4,
        ),

        "joint_label_and_gold_doc_hit_rate": round(
            float(joint_correct.mean()),
            4,
        ),

        "joint_label_and_gold_doc_hit_count": int(
            joint_correct.sum()
        ),

        "nei_abstention_rate": round(
            float((~decisive).mean()),
            4,
        ),

        "decisive_coverage": round(
            float(decisive.mean()),
            4,
        ),

        "decisive_accuracy": (
            round(
                float(correct[decisive].mean()),
                4,
            )
            if decisive.any()
            else None
        ),

        "correct_count": int(correct.sum()),

        "wrong_polarity_count": int(
            (decisive & ~correct).sum()
        ),

        "nei_count": int(
            (~decisive).sum()
        ),

        "cross_source_conflict_cases": int(
            df["cross_source_conflict"].sum()
        ),
    }


def evaluate_end_to_end_sample(
    retriever,
    selector,
    classifier,
    dev_claims,
    sample_size=60,
):
    """Evaluate retrieval, evidence selection and stance."""

    print("\n[3/4] Evaluating end-to-end pipeline...")

    print(
        f"  Routing threshold: {classifier.threshold}"
    )

    claims_with_gold = [
        c for c in dev_claims
        if c.get("evidence")
    ][:sample_size]

    records = []

    for idx, item in enumerate(claims_with_gold, 1):

        claim = item["claim"]

        gold_labels = [
            ev["label"]
            for ev_list in item["evidence"].values()
            for ev in ev_list
            if ev["label"] in LABELS
        ]

        if not gold_labels:
            continue

        counts = Counter(gold_labels)

        # Deterministic majority-label selection.
        gold_claim_label = sorted(
            counts,
            key=lambda label: (
                -counts[label],
                label,
            ),
        )[0]

        gold_doc_ids = {
            int(doc_id)
            for doc_id in item["evidence"]
        }

        retrieved_docs = retriever.retrieve_top_k(
            claim,
            k=RETRIEVAL_K,
        )

        retrieved_ids = {
            int(doc["doc_id"])
            for doc in retrieved_docs
        }

        gold_doc_hit = bool(
            retrieved_ids & gold_doc_ids
        )

        evidence_predictions = []

        for doc in retrieved_docs:

            rationale_result = (
                selector.select_rationales_for_doc(
                    claim=claim,
                    doc=doc,
                    threshold=RATIONALE_THRESHOLD,
                    top_k=RATIONALE_TOP_K,
                )
            )

            if not rationale_result["rationales"]:
                continue

            rationale_text = " ".join(
                rationale_result["rationales"]
            ).strip()

            if not rationale_text:
                continue

            stance = classifier.predict_stance(
                claim,
                rationale_text,
            )

            evidence_predictions.append({
                "doc_id": int(doc["doc_id"]),
                "label": stance["label"],
                "confidence": stance["confidence"],
                "model_used": stance["model_used"],
            })

        decisive = [
            ev for ev in evidence_predictions
            if ev["label"] in LABELS
        ]

        # Claim-level aggregation for evaluation.
        # The website also exposes individual source verdicts.
        if decisive:

            best = max(
                decisive,
                key=lambda ev: ev["confidence"],
            )

            pred_label = best["label"]
            pred_conf = best["confidence"]
            selected_doc_id = best["doc_id"]

        else:

            pred_label = "NOT_ENOUGH_INFO"

            # Preserve actual model confidence.
            # Do not invent 100% confidence.
            pred_conf = (
                max(
                    ev["confidence"]
                    for ev in evidence_predictions
                )
                if evidence_predictions
                else None
            )

            selected_doc_id = None

        unique_decisive = {
            ev["label"]
            for ev in decisive
        }

        cross_source_conflict = (
            "SUPPORT" in unique_decisive
            and "CONTRADICT" in unique_decisive
        )

        records.append({
            "claim_id": item["id"],
            "claim": claim,
            "gold_label": gold_claim_label,
            "pred_label": pred_label,

            "pred_conf": (
                round(float(pred_conf), 4)
                if pred_conf is not None
                else None
            ),

            "gold_doc_retrieved_top3": gold_doc_hit,

            "num_docs_with_rationales": len(
                evidence_predictions
            ),

            "cross_source_conflict": cross_source_conflict,

            "routing_threshold": classifier.threshold,

            "selected_doc_id": selected_doc_id,

            "selected_doc_is_gold": (
                selected_doc_id in gold_doc_ids
                if selected_doc_id is not None
                else False
            ),

            "source_predictions": json.dumps(
                evidence_predictions
            ),
        })

        if idx % 10 == 0 or idx == len(claims_with_gold):
            print(
                f"  Completed "
                f"{idx}/{len(claims_with_gold)} claims"
            )

    df = pd.DataFrame(records)

    summary = summarize_end_to_end(df)

    summary.update({
        "split": "dev",
        "sampling_method": (
            "First eligible gold-annotated dev claims"
        ),
        "routing_threshold": classifier.threshold,
        "retrieval_k": RETRIEVAL_K,
        "rationale_threshold": RATIONALE_THRESHOLD,
        "rationale_top_k": RATIONALE_TOP_K,
        "aggregation": (
            "Highest-confidence decisive source prediction; "
            "otherwise NOT_ENOUGH_INFO"
        ),
        "selected_gold_source_and_correct_label_rate": round(
            float(
                (
                    (df["gold_label"] == df["pred_label"])
                    & df["selected_doc_is_gold"]
                ).mean()
            ),
            4,
        ),
    })

    df.to_csv(
        OUT_DIR / "end_to_end_sample_predictions.csv",
        index=False,
    )

    print(json.dumps(summary, indent=2))

    return summary


def evaluate_consistency_edits(classifier):
    """Controlled stance consistency experiments."""

    print("\n[4/4] Evaluating controlled claim edits...")

    # Original five controlled examples are preserved.
    test_cases = [
        (
            "Vitamin D deficiency is associated with an increased risk of osteoporosis.",
            "Vitamin D deficiency is not associated with an increased risk of osteoporosis.",
            "Low vitamin D levels are linked to a higher risk of developing osteoporosis.",
            "Multiple clinical studies demonstrate that vitamin D deficiency significantly increases the risk of osteoporosis and bone fractures.",
        ),
        (
            "Aspirin reduces the risk of myocardial infarction in high-risk patients.",
            "Aspirin does not reduce the risk of myocardial infarction in high-risk patients.",
            "Taking aspirin lowers the likelihood of heart attacks among high-risk individuals.",
            "Randomized trials show daily low-dose aspirin significantly reduces the incidence of myocardial infarction in high-risk cardiovascular patients.",
        ),
        (
            "Elevated blood glucose levels impair wound healing in diabetic patients.",
            "Elevated blood glucose levels improve wound healing in diabetic patients.",
            "High blood sugar slows down wound repair among people with diabetes.",
            "Hyperglycemia directly impairs angiogenesis and immune function, leading to delayed wound healing in patients with diabetes.",
        ),
        (
            "Regular aerobic exercise lowers resting systolic blood pressure.",
            "Regular aerobic exercise increases resting systolic blood pressure.",
            "Consistent aerobic workouts reduce baseline systolic blood pressure.",
            "Meta-analyses confirm that structured aerobic exercise training produces clinically meaningful reductions in resting systolic blood pressure.",
        ),
        (
            "Antibiotics are effective against viral respiratory infections.",
            "Antibiotics are ineffective against viral respiratory infections.",
            "Antibiotic drugs can cure respiratory infections caused by viruses.",
            "Antibiotics target bacterial pathways and have no therapeutic efficacy against viral respiratory tract infections.",
        ),
    ]

    results = []

    for orig, neg, para, rationale in test_cases:

        p_orig = classifier.predict_stance(
            orig,
            rationale,
        )

        p_neg = classifier.predict_stance(
            neg,
            rationale,
        )

        p_para = classifier.predict_stance(
            para,
            rationale,
        )

        para_same = (
            p_orig["label"] == p_para["label"]
        )

        neg_flip = (
            (
                p_orig["label"] == "SUPPORT"
                and p_neg["label"] == "CONTRADICT"
            )
            or
            (
                p_orig["label"] == "CONTRADICT"
                and p_neg["label"] == "SUPPORT"
            )
        )

        results.append({
            "original_claim": orig,
            "negated_claim": neg,
            "paraphrased_claim": para,
            "rationale": rationale,
            "orig_label": p_orig["label"],
            "orig_conf": round(
                p_orig["confidence"],
                4,
            ),
            "paraphrase_label": p_para["label"],
            "paraphrase_consistent": para_same,
            "negated_label": p_neg["label"],
            "negation_flipped": neg_flip,
            "routing_threshold": classifier.threshold,
        })

    df = pd.DataFrame(results)

    df.to_csv(
        OUT_DIR / "consistency_check.csv",
        index=False,
    )

    summary = {
        "num_cases": len(results),

        "routing_threshold": classifier.threshold,

        "evaluation_scope": (
            "Five manually constructed fixed-evidence "
            "examples; not an end-to-end robustness test"
        ),

        "paraphrase_consistency_rate": round(
            float(df["paraphrase_consistent"].mean()),
            4,
        ),

        "negation_flip_rate": round(
            float(df["negation_flipped"].mean()),
            4,
        ),

        "cases": results,
    }

    print(json.dumps(summary, indent=2))

    return summary


def reuse_saved_predictions():
    """Recompute derived metrics without model inference."""

    print("\nRecomputing metrics from saved predictions...")

    summary = load_summary()

    stance_df = pd.read_csv(
        OUT_DIR / "stance_oracle_predictions.csv"
    )

    sweep, routing = summarize_routing(stance_df)

    # Preserve any existing experiment metadata.
    summary.setdefault("stance_and_routing", {})
    summary["stance_and_routing"].update(routing)

    sweep.to_csv(
        OUT_DIR / "routing_threshold_sweep.csv",
        index=False,
    )

    e2e_path = (
        OUT_DIR / "end_to_end_sample_predictions.csv"
    )

    if e2e_path.exists():

        df = pd.read_csv(e2e_path)

        summary.setdefault("end_to_end_sample", {})

        summary["end_to_end_sample"].update(
            summarize_end_to_end(df)
        )

        # The original evaluator used threshold 0.60.
        # Do not relabel historical predictions as 0.95.
        if "routing_threshold" in df.columns:

            thresholds = df[
                "routing_threshold"
            ].dropna().unique()

            if len(thresholds) != 1:
                raise ValueError(
                    "Unexpected mixed routing thresholds."
                )

            summary["end_to_end_sample"][
                "routing_threshold"
            ] = float(thresholds[0])

        else:

            summary["end_to_end_sample"].setdefault(
                "routing_threshold",
                0.60,
            )

        summary["end_to_end_sample"].setdefault(
            "split",
            "dev",
        )

        summary["end_to_end_sample"].setdefault(
            "sampling_method",
            "First eligible gold-annotated dev claims",
        )

    if "consistency_edits" in summary:

        summary["consistency_edits"].setdefault(
            "routing_threshold",
            0.60,
        )

    save_summary(summary)

    print("\nRouting metrics:")

    print(
        sweep[[
            "mode",
            "accuracy",
            "macro_f1_support_contradict",
            "avg_latency_ms",
        ]].to_string(index=False)
    )

    print(
        "\nCorrected latency saving:",
        routing["selected_latency_saving_percent"],
        "%",
    )

    if "end_to_end_sample" in summary:

        print("\nSaved end-to-end results:")

        print(
            json.dumps(
                summary["end_to_end_sample"],
                indent=2,
            )
        )

    print("\nRaw predictions preserved.")

    print("No model inference performed.")


def main():

    parser = argparse.ArgumentParser(
        description="LitScope-Lite Phase 2 evaluation"
    )

    parser.add_argument(
        "--only",
        choices=[
            "all",
            "saved-routing",
            "retrieval",
            "stance",
            "end-to-end",
            "consistency",
        ],
        default="all",
        help="Run a specific evaluation section.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=60,
        help="Number of end-to-end development claims.",
    )

    args = parser.parse_args()

    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")

    if args.only == "saved-routing":
        reuse_saved_predictions()
        return

    summary = load_summary()

    selected = args.only

    run_retrieval = selected in (
        "all",
        "retrieval",
    )

    run_stance = selected in (
        "all",
        "stance",
    )

    run_e2e = selected in (
        "all",
        "end-to-end",
    )

    run_consistency = selected in (
        "all",
        "consistency",
    )

    dev_claims = None
    retriever = None
    selector = None
    classifier = None

    if run_retrieval or run_stance or run_e2e:

        dev_claims = load_jsonl(
            DATA_DIR / "claims_dev.jsonl"
        )

    if run_retrieval or run_e2e:

        from src.retrieval import SciFactRetriever

        print("Initializing retrieval model...")

        retriever = SciFactRetriever(
            corpus_path=DATA_DIR / "corpus.jsonl"
        )

    if run_e2e:

        from src.rationale_selection import RationaleSelector

        selector = RationaleSelector(
            model=retriever.dense_model,
            threshold=RATIONALE_THRESHOLD,
            top_k=RATIONALE_TOP_K,
        )

    if run_stance or run_e2e or run_consistency:

        from src.stance import StanceClassifier

        print("Initializing stance models...")

        classifier = StanceClassifier(
            confidence_threshold=SELECTED_THRESHOLD
        )

        print(
            "Selected routing threshold:",
            SELECTED_THRESHOLD,
        )

    if run_retrieval:

        summary["retrieval"] = evaluate_retrieval(
            retriever,
            dev_claims,
        )

        save_summary(summary)

    if run_stance:

        corpus = load_jsonl(
            DATA_DIR / "corpus.jsonl"
        )

        corpus_map = {
            int(doc["doc_id"]): doc
            for doc in corpus
        }

        summary["stance_and_routing"] = (
            evaluate_stance_and_routing(
                classifier,
                dev_claims,
                corpus_map,
            )
        )

        save_summary(summary)

    if run_e2e:

        summary["end_to_end_sample"] = (
            evaluate_end_to_end_sample(
                retriever,
                selector,
                classifier,
                dev_claims,
                sample_size=args.sample_size,
            )
        )

        save_summary(summary)

    if run_consistency:

        summary["consistency_edits"] = (
            evaluate_consistency_edits(
                classifier
            )
        )

        save_summary(summary)

    print(
        "\nEvaluation completed successfully."
    )

    print(
        "Results saved to:",
        OUT_DIR.resolve(),
    )


if __name__ == "__main__":
    main()
