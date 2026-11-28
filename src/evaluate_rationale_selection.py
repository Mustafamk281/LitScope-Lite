
"""Evaluate sentence-level evidence selection on SciFact dev."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.rationale_selection import RationaleSelector
from src.utils import get_data_path, load_jsonl


ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "outputs" / "evaluation"
IMAGE_DIR = ROOT / "images"

SUMMARY_PATH = OUT_DIR / "phase2_evaluation_summary.json"

THRESHOLD = 0.40
TOP_K = 3

METHODS = [
    "First-3 baseline",
    "MiniLM selector",
]


def compute_metrics(predicted, gold, gold_sets):
    """Calculate sentence-level evidence metrics."""

    predicted = set(predicted)
    gold = set(gold)

    if not gold:
        raise ValueError("Gold evidence cannot be empty.")

    true_positives = len(predicted & gold)

    false_positives = len(predicted - gold)

    false_negatives = len(gold - predicted)

    precision = (
        true_positives / len(predicted)
        if predicted
        else 0.0
    )

    recall = true_positives / len(gold)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": precision,

        "recall": recall,

        "f1": f1,

        "hit": int(true_positives > 0),

        "exact_match": int(predicted == gold),

        "complete_rationale": int(
            any(
                gold_set.issubset(predicted)
                for gold_set in gold_sets
            )
        ),

        "true_positives": true_positives,

        "false_positives": false_positives,

        "false_negatives": false_negatives,

        "num_selected_sentences": len(predicted),

        "num_gold_sentences": len(gold),
    }


def validate_metrics():
    """Verify metric calculations on known examples."""

    result = compute_metrics(
        predicted={1, 3, 4},
        gold={1, 4},
        gold_sets=[{1, 4}],
    )

    assert np.isclose(
        result["precision"],
        2 / 3,
    )

    assert result["recall"] == 1.0

    assert np.isclose(
        result["f1"],
        0.8,
    )

    assert result["hit"] == 1

    assert result["complete_rationale"] == 1

    assert result["exact_match"] == 0

    result = compute_metrics(
        predicted={2},
        gold={1, 4},
        gold_sets=[{1, 4}],
    )

    assert result["f1"] == 0.0

    assert result["hit"] == 0

    assert result["complete_rationale"] == 0

    result = compute_metrics(
        predicted={1},
        gold={1, 4},
        gold_sets=[{1}, {4}],
    )

    assert result["complete_rationale"] == 1

    assert result["recall"] == 0.5

    print("Rationale metric checks passed.")


def build_evaluation_pairs(dev_claims, corpus_map):
    """Build one evaluation pair per claim and gold document."""

    pairs = []

    for item in dev_claims:

        for doc_id_str, annotations in item.get(
            "evidence", {}
        ).items():

            doc_id = int(doc_id_str)

            if doc_id not in corpus_map:

                raise ValueError(
                    f"Missing gold document: {doc_id}"
                )

            doc = corpus_map[doc_id]

            abstract = doc.get("abstract", [])

            if not isinstance(abstract, list):

                raise ValueError(
                    f"Expected sentence list for {doc_id}"
                )

            # Preserve original SciFact sentence indices.

            original_indices = [
                i for i, sentence in enumerate(abstract)
                if sentence and sentence.strip()
            ]

            if not original_indices:

                raise ValueError(
                    f"Empty gold abstract: {doc_id}"
                )

            gold_sets = []

            for annotation in annotations:

                sentence_ids = set(
                    annotation.get("sentences", [])
                )

                if not sentence_ids:
                    continue

                if not sentence_ids.issubset(
                    set(original_indices)
                ):

                    raise ValueError(
                        f"Invalid gold sentence indices "
                        f"for document {doc_id}"
                    )

                gold_sets.append(sentence_ids)

            if not gold_sets:
                continue

            gold_union = set().union(*gold_sets)

            pairs.append({
                "claim_id": item["id"],

                "claim": item["claim"],

                "doc_id": doc_id,

                "doc": doc,

                "original_indices": original_indices,

                "gold_sets": gold_sets,

                "gold_union": gold_union,
            })

    return pairs


def evaluate(selector, pairs):
    """Evaluate semantic selection and positional baseline."""

    rows = []

    start = time.perf_counter()

    for idx, pair in enumerate(pairs, 1):

        result = selector.select_rationales_for_doc(
            claim=pair["claim"],
            doc=pair["doc"],
            threshold=THRESHOLD,
            top_k=TOP_K,
        )

        original_indices = pair["original_indices"]

        # The selector returns indices relative to the
        # filtered nonempty sentence list. Convert them
        # back to the original SciFact indices.

        selected_indices = [
            original_indices[i]
            for i in result["sentence_indices"]
        ]

        selected_scores = [
            float(item["score"])
            for item in result["selected_sentences"]
        ]

        fallback_used = bool(
            selected_scores
            and all(
                score < THRESHOLD
                for score in selected_scores
            )
        )

        baseline_indices = original_indices[:TOP_K]

        predictions = {
            "First-3 baseline": baseline_indices,

            "MiniLM selector": selected_indices,
        }

        for method, predicted in predictions.items():

            metrics = compute_metrics(
                predicted=predicted,

                gold=pair["gold_union"],

                gold_sets=pair["gold_sets"],
            )

            rows.append({
                "claim_id": pair["claim_id"],

                "claim": pair["claim"],

                "doc_id": pair["doc_id"],

                "method": method,

                "gold_sentence_indices": json.dumps(
                    sorted(pair["gold_union"])
                ),

                "gold_rationale_sets": json.dumps([
                    sorted(group)
                    for group in pair["gold_sets"]
                ]),

                "predicted_sentence_indices": json.dumps(
                    predicted
                ),

                "fallback_used": (
                    fallback_used
                    if method == "MiniLM selector"
                    else None
                ),

                "selected_similarity_scores": json.dumps(
                    selected_scores
                    if method == "MiniLM selector"
                    else []
                ),

                **metrics,
            })

        if idx % 25 == 0 or idx == len(pairs):

            elapsed = time.perf_counter() - start

            print(
                f"Processed {idx}/{len(pairs)} pairs "
                f"| {elapsed:.1f}s elapsed",
                flush=True,
            )

    return pd.DataFrame(rows)


def summarize(df):
    """Calculate macro and micro sentence-selection metrics."""

    rows = []

    for method in METHODS:

        subset = df[df["method"] == method]

        tp = int(
            subset["true_positives"].sum()
        )

        fp = int(
            subset["false_positives"].sum()
        )

        fn = int(
            subset["false_negatives"].sum()
        )

        micro_precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        micro_recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        micro_f1 = (
            2 * tp / (2 * tp + fp + fn)
            if 2 * tp + fp + fn > 0
            else 0.0
        )

        rows.append({
            "method": method,

            "num_pairs": len(subset),

            "macro_precision": round(
                float(subset["precision"].mean()),
                4,
            ),

            "macro_recall": round(
                float(subset["recall"].mean()),
                4,
            ),

            "macro_f1": round(
                float(subset["f1"].mean()),
                4,
            ),

            "micro_precision": round(
                micro_precision,
                4,
            ),

            "micro_recall": round(
                micro_recall,
                4,
            ),

            "micro_f1": round(
                micro_f1,
                4,
            ),

            "sentence_hit_rate": round(
                float(subset["hit"].mean()),
                4,
            ),

            "exact_match_rate": round(
                float(subset["exact_match"].mean()),
                4,
            ),

            "complete_rationale_rate": round(
                float(
                    subset["complete_rationale"].mean()
                ),
                4,
            ),

            "mean_selected_sentences": round(
                float(
                    subset["num_selected_sentences"].mean()
                ),
                4,
            ),

            "mean_gold_sentences": round(
                float(
                    subset["num_gold_sentences"].mean()
                ),
                4,
            ),

            "true_positives": tp,

            "false_positives": fp,

            "false_negatives": fn,
        })

    return pd.DataFrame(rows)


def create_plot(summary_df):
    """Generate rationale selection comparison figure."""

    columns = [
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "sentence_hit_rate",
        "complete_rationale_rate",
    ]

    labels = [
        "Precision",
        "Recall",
        "F1",
        "Sentence hit",
        "Complete rationale",
    ]

    x = np.arange(len(columns))

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(12, 6),
    )

    for idx, method in enumerate(METHODS):

        row = summary_df[
            summary_df["method"] == method
        ].iloc[0]

        values = [
            float(row[column])
            for column in columns
        ]

        bars = ax.bar(
            x + (idx - 0.5) * width,
            values,
            width,
            label=method,
        )

        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=9,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(labels)

    ax.set_ylim(0, 1.12)

    ax.set_ylabel("Score")

    ax.set_title(
        "Sentence-Level Evidence Selection on SciFact Dev"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
        linestyle="--",
    )

    fig.tight_layout()

    path = (
        IMAGE_DIR
        / "04_07_rationale_selection_comparison.png"
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_metrics()

    print("\nLoading SciFact data...")

    dev_claims = load_jsonl(
        get_data_path("claims_dev.jsonl")
    )

    corpus = load_jsonl(
        get_data_path("corpus.jsonl")
    )

    corpus_map = {
        int(doc["doc_id"]): doc
        for doc in corpus
    }

    pairs = build_evaluation_pairs(
        dev_claims,
        corpus_map,
    )

    if not pairs:

        raise ValueError(
            "No valid rationale evaluation pairs."
        )

    print(
        "Unique annotated claims:",
        len({
            pair["claim_id"]
            for pair in pairs
        }),
    )

    print(
        "Gold claim-document pairs:",
        len(pairs),
    )

    print(
        "Gold rationale sets:",
        sum(
            len(pair["gold_sets"])
            for pair in pairs
        ),
    )

    print("\nInitializing MiniLM rationale selector...")

    selector = RationaleSelector(
        threshold=THRESHOLD,
        top_k=TOP_K,
    )

    print("Rationale selector ready.")

    start = time.perf_counter()

    df = evaluate(
        selector,
        pairs,
    )

    elapsed = time.perf_counter() - start

    assert len(df) == len(pairs) * len(METHODS)

    summary_df = summarize(df)

    semantic_df = df[
        df["method"] == "MiniLM selector"
    ]

    fallback_count = int(
        semantic_df["fallback_used"].eq(True).sum()
    )

    fallback_rate = (
        fallback_count / len(semantic_df)
    )

    df.to_csv(
        OUT_DIR / "rationale_selection_per_pair.csv",
        index=False,
    )

    summary_df.to_csv(
        OUT_DIR / "rationale_selection_summary.csv",
        index=False,
    )

    image_path = create_plot(
        summary_df
    )

    if SUMMARY_PATH.exists():

        summary = json.loads(
            SUMMARY_PATH.read_text(
                encoding="utf-8"
            )
        )

    else:

        summary = {}

    summary["rationale_selection"] = {

        "split": "dev",

        "evaluation_type": (
            "Oracle-document sentence selection"
        ),

        "num_claims": len({
            pair["claim_id"]
            for pair in pairs
        }),

        "num_claim_document_pairs": len(pairs),

        "num_gold_rationale_sets": sum(
            len(pair["gold_sets"])
            for pair in pairs
        ),

        "threshold": THRESHOLD,

        "top_k": TOP_K,

        "model": (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),

        "gold_annotation_method": (
            "Union of annotated sentence indices "
            "per claim-document pair"
        ),

        "complete_rationale_definition": (
            "At least one complete annotated rationale "
            "set is contained in the predicted sentences"
        ),

        "baseline": (
            "First three nonempty abstract sentences"
        ),

        "macro_averaging": (
            "Equal weight per claim-document pair"
        ),

        "fallback_count": fallback_count,

        "fallback_rate": round(
            fallback_rate,
            4,
        ),

        "elapsed_seconds": round(
            elapsed,
            2,
        ),

        "results": summary_df.to_dict(
            orient="records"
        ),

        "limitation": (
            "Gold documents are supplied directly. "
            "Results isolate sentence selection "
            "and do not measure retrieval failures "
            "or end-to-end verification accuracy."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 80)

    print("RATIONALE SELECTION RESULTS")

    print("=" * 80)

    print(
        summary_df[[
            "method",
            "num_pairs",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "micro_f1",
        ]].to_string(index=False)
    )

    print("\nAdditional evidence coverage metrics:")

    print(
        summary_df[[
            "method",
            "sentence_hit_rate",
            "exact_match_rate",
            "complete_rationale_rate",
            "mean_selected_sentences",
        ]].to_string(index=False)
    )

    print(
        "\nMiniLM fallback count:",
        fallback_count,
    )

    print(
        "MiniLM fallback rate:",
        f"{fallback_rate:.2%}",
    )

    print(
        "\nEvaluation time:",
        f"{elapsed:.2f} seconds",
    )

    print(
        "Comparison figure saved:",
        image_path,
    )

    print(
        "Per-pair evaluation rows:",
        len(df),
    )

    print(
        "Existing evaluation results preserved."
    )

    print(
        "No DeBERTa inference or model training performed."
    )

    print(
        "Rationale selection evaluation completed successfully."
    )


if __name__ == "__main__":
    main()
