
"""Compare lexical, semantic and hybrid retrieval on SciFact dev."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.retrieval import SciFactRetriever
from src.utils import get_data_path, load_jsonl


ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "outputs" / "evaluation"
IMAGE_DIR = ROOT / "images"

SUMMARY_PATH = OUT_DIR / "phase2_evaluation_summary.json"

KS = (1, 3, 5, 10)

CONFIGURATIONS = {
    "TF-IDF": {
        "lexical_weight": 1.0,
        "dense_weight": 0.0,
    },
    "MiniLM": {
        "lexical_weight": 0.0,
        "dense_weight": 1.0,
    },
    "Hybrid": {
        "lexical_weight": 0.5,
        "dense_weight": 0.5,
    },
}


def compute_metrics(retrieved_ids, gold_ids, k):
    """Calculate document retrieval metrics for one claim."""

    if not gold_ids:
        raise ValueError("Gold document set cannot be empty.")

    top_k = retrieved_ids[:k]

    hits = len(set(top_k) & gold_ids)

    reciprocal_rank = 0.0

    for rank, doc_id in enumerate(top_k, 1):

        if doc_id in gold_ids:

            reciprocal_rank = 1.0 / rank

            break

    return {
        "recall": hits / len(gold_ids),

        "precision": hits / k,

        "hit_rate": float(hits > 0),

        "reciprocal_rank": reciprocal_rank,
    }


def validate_metrics():
    """Small deterministic check of metric calculations."""

    result = compute_metrics(
        retrieved_ids=[10, 20, 30],
        gold_ids={20, 40},
        k=3,
    )

    assert np.isclose(result["recall"], 0.5)

    assert np.isclose(result["precision"], 1 / 3)

    assert result["hit_rate"] == 1.0

    assert result["reciprocal_rank"] == 0.5

    missing = compute_metrics(
        retrieved_ids=[10, 20, 30],
        gold_ids={40},
        k=3,
    )

    assert all(
        value == 0.0
        for value in missing.values()
    )

    print("Retrieval metric checks passed.")


def evaluate_ablation(retriever, dev_claims):
    """Evaluate each configuration on identical claims."""

    eligible = [
        item for item in dev_claims
        if item.get("evidence")
    ]

    if not eligible:
        raise ValueError("No gold-annotated claims found.")

    print(
        f"\nTotal development claims: "
        f"{len(dev_claims)}"
    )

    print(
        f"Gold-annotated evaluation claims: "
        f"{len(eligible)}"
    )

    print(
        f"Corpus documents: "
        f"{len(retriever.corpus)}"
    )

    per_claim_rows = []

    max_k = max(KS)

    start = time.perf_counter()

    for idx, item in enumerate(eligible, 1):

        claim = item["claim"]

        gold_ids = {
            int(doc_id)
            for doc_id in item["evidence"]
        }

        for method, weights in CONFIGURATIONS.items():

            results = retriever.retrieve_top_k(
                claim=claim,
                k=max_k,
                lexical_weight=weights["lexical_weight"],
                dense_weight=weights["dense_weight"],
            )

            retrieved_ids = [
                int(doc["doc_id"])
                for doc in results
            ]

            for k in KS:

                metrics = compute_metrics(
                    retrieved_ids,
                    gold_ids,
                    k,
                )

                per_claim_rows.append({
                    "claim_id": item["id"],

                    "claim": claim,

                    "method": method,

                    "lexical_weight": (
                        weights["lexical_weight"]
                    ),

                    "dense_weight": (
                        weights["dense_weight"]
                    ),

                    "k": k,

                    "gold_doc_count": len(gold_ids),

                    "gold_doc_ids": json.dumps(
                        sorted(gold_ids)
                    ),

                    "retrieved_doc_ids": json.dumps(
                        retrieved_ids[:k]
                    ),

                    **metrics,
                })

        if idx % 20 == 0 or idx == len(eligible):

            elapsed = time.perf_counter() - start

            print(
                f"Processed {idx}/{len(eligible)} "
                f"claims | {elapsed:.1f}s elapsed",
                flush=True,
            )

    df = pd.DataFrame(per_claim_rows)

    expected_rows = (
        len(eligible)
        * len(CONFIGURATIONS)
        * len(KS)
    )

    if len(df) != expected_rows:

        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"but generated {len(df)}."
        )

    return df


def summarize_results(df):
    """Macro-average metrics across development claims."""

    summary_rows = []

    for method, weights in CONFIGURATIONS.items():

        for k in KS:

            subset = df[
                (df["method"] == method)
                & (df["k"] == k)
            ]

            summary_rows.append({
                "method": method,

                "lexical_weight": (
                    weights["lexical_weight"]
                ),

                "dense_weight": (
                    weights["dense_weight"]
                ),

                "k": k,

                "num_claims": len(subset),

                "recall": round(
                    float(subset["recall"].mean()),
                    4,
                ),

                "precision": round(
                    float(subset["precision"].mean()),
                    4,
                ),

                "hit_rate": round(
                    float(subset["hit_rate"].mean()),
                    4,
                ),

                "mrr": round(
                    float(
                        subset["reciprocal_rank"].mean()
                    ),
                    4,
                ),
            })

    return pd.DataFrame(summary_rows)


def create_comparison_plot(summary_df):
    """Save the retrieval ablation comparison figure."""

    colors = {
        "TF-IDF": "#64748b",
        "MiniLM": "#2563eb",
        "Hybrid": "#059669",
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    metrics = [
        ("recall", "Recall@K"),
        ("precision", "Precision@K"),
        ("hit_rate", "Hit Rate@K"),
        ("mrr", "Mean Reciprocal Rank@K"),
    ]

    for ax, (column, title) in zip(
        axes.flat,
        metrics,
    ):

        for method in CONFIGURATIONS:

            subset = summary_df[
                summary_df["method"] == method
            ].sort_values("k")

            ax.plot(
                subset["k"],
                subset[column],
                marker="o",
                linewidth=2,
                markersize=6,
                label=method,
                color=colors[method],
            )

        ax.set_title(
            title,
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xlabel("Retrieved documents (K)")

        ax.set_ylabel("Score")

        ax.set_xticks(KS)

        ax.set_ylim(0, 1.05)

        ax.grid(
            alpha=0.25,
            linestyle="--",
        )

        ax.legend(loc="best")

    fig.suptitle(
        "SciFact Development Set: Retrieval Ablation Study",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    image_path = (
        IMAGE_DIR / "04_06_retrieval_ablation.png"
    )

    fig.savefig(
        image_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    return image_path


def compare_historical_hybrid(summary_df, previous):
    """Check whether hybrid results reproduce earlier metrics."""

    old = previous.get("retrieval", {})

    old_metrics = old.get("metrics_by_k", {})

    print("\nHistorical hybrid retrieval comparison:")

    comparisons = []

    for k in KS:

        old_row = old_metrics.get(str(k), {})

        if "recall" not in old_row:
            continue

        new_row = summary_df[
            (summary_df["method"] == "Hybrid")
            & (summary_df["k"] == k)
        ].iloc[0]

        old_recall = float(old_row["recall"])

        new_recall = float(new_row["recall"])

        difference = new_recall - old_recall

        comparisons.append({
            "k": k,
            "historical_recall": old_recall,
            "current_recall": new_recall,
            "difference": round(difference, 4),
        })

        print(
            f"Recall@{k}: "
            f"previous={old_recall:.4f}, "
            f"current={new_recall:.4f}, "
            f"difference={difference:+.4f}"
        )

    return comparisons


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

    if SUMMARY_PATH.exists():

        previous_summary = json.loads(
            SUMMARY_PATH.read_text(
                encoding="utf-8"
            )
        )

    else:

        previous_summary = {}

    print("\nLoading SciFact development data...")

    dev_claims = load_jsonl(
        get_data_path("claims_dev.jsonl")
    )

    print("Initializing retrieval indices...")

    retriever = SciFactRetriever(
        use_cache=True
    )

    print("Retriever ready.")

    per_claim_df = evaluate_ablation(
        retriever,
        dev_claims,
    )

    summary_df = summarize_results(
        per_claim_df
    )

    # Save individual predictions for reproducibility.

    per_claim_df.to_csv(
        OUT_DIR / "retrieval_ablation_per_claim.csv",
        index=False,
    )

    # Save aggregate metrics for notebooks and reports.

    summary_df.to_csv(
        OUT_DIR / "retrieval_ablation_summary.csv",
        index=False,
    )

    image_path = create_comparison_plot(
        summary_df
    )

    historical_comparison = (
        compare_historical_hybrid(
            summary_df,
            previous_summary,
        )
    )

    # Identify the strongest configuration for each metric.

    top3 = summary_df[
        summary_df["k"] == 3
    ]

    top10 = summary_df[
        summary_df["k"] == 10
    ]

    best_recall3 = top3["recall"].max()

    best_mrr10 = top10["mrr"].max()

    best_recall_methods = top3.loc[
        top3["recall"] == best_recall3,
        "method",
    ].tolist()

    best_mrr_methods = top10.loc[
        top10["mrr"] == best_mrr10,
        "method",
    ].tolist()

    ablation_summary = {

        "split": "dev",

        "num_dev_claims_total": len(dev_claims),

        "num_evaluated_claims": int(
            per_claim_df["claim_id"].nunique()
        ),

        "num_corpus_documents": len(
            retriever.corpus
        ),

        "configurations": CONFIGURATIONS,

        "retrieval_cutoffs": list(KS),

        "averaging_method": (
            "Macro-average across gold-annotated claims"
        ),

        "evaluation_scope": (
            "Claims with annotated gold evidence only. "
            "Unannotated documents are not necessarily "
            "irrelevant."
        ),

        "results": summary_df.to_dict(
            orient="records"
        ),

        "historical_hybrid_comparison": (
            historical_comparison
        ),

        "best_recall_at_3": {
            "methods": best_recall_methods,
            "score": float(best_recall3),
        },

        "best_mrr_at_10": {
            "methods": best_mrr_methods,
            "score": float(best_mrr10),
        },
    }

    # Preserve all previously saved experiments.

    previous_summary[
        "retrieval_ablation"
    ] = ablation_summary

    SUMMARY_PATH.write_text(
        json.dumps(
            previous_summary,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 80)

    print("RETRIEVAL ABLATION RESULTS")

    print("=" * 80)

    print(
        summary_df[[
            "method",
            "k",
            "num_claims",
            "recall",
            "precision",
            "hit_rate",
            "mrr",
        ]].to_string(index=False)
    )

    print("\nRecall@3 comparison:")

    print(
        top3[[
            "method",
            "recall",
            "precision",
            "hit_rate",
        ]].to_string(index=False)
    )

    print("\nMRR@10 comparison:")

    print(
        top10[[
            "method",
            "mrr",
        ]].to_string(index=False)
    )

    print(
        "\nBest Recall@3:",
        ", ".join(best_recall_methods),
        f"({best_recall3:.4f})",
    )

    print(
        "Best MRR@10:",
        ", ".join(best_mrr_methods),
        f"({best_mrr10:.4f})",
    )

    print(
        "\nComparison figure saved:",
        image_path,
    )

    print(
        "Per-claim evaluation rows:",
        len(per_claim_df),
    )

    print(
        "Existing stance and end-to-end results preserved."
    )

    print(
        "No DeBERTa inference or model training performed."
    )

    print(
        "Retrieval ablation completed successfully."
    )


if __name__ == "__main__":
    main()
