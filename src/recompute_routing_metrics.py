
"""Recompute routing metrics from saved SciFact development predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "evaluation"

LABELS = [
    "SUPPORT",
    "CONTRADICT",
    "NOT_ENOUGH_INFO",
]

THRESHOLDS = [
    0.0,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    1.01,
]

SELECTED_THRESHOLD = 0.95


def summarize_routing(df):
    """Calculate routing metrics from saved model predictions."""

    required = {
        "gold_label",
        "base_pred",
        "base_conf",
        "base_ms",
        "large_pred",
        "large_ms",
    }

    missing = required - set(df.columns)

    if missing or df.empty:
        raise ValueError(
            f"Missing columns: {sorted(missing)}; rows: {len(df)}"
        )

    if not df["gold_label"].isin(LABELS[:2]).all():
        raise ValueError(
            "Oracle evaluation expects SUPPORT and CONTRADICT gold labels."
        )

    for col in ("base_pred", "large_pred"):
        if not df[col].isin(LABELS).all():
            raise ValueError(f"Invalid labels in {col}")

    for col in ("base_conf", "base_ms", "large_ms"):
        if (
            not np.isfinite(df[col]).all()
            or (df[col] < 0).any()
        ):
            raise ValueError(f"Invalid numeric values in {col}")

    if (df["base_conf"] > 1).any():
        raise ValueError(
            "Confidence must be between zero and one."
        )

    gold = df["gold_label"].to_numpy()

    small = df["base_pred"].to_numpy()
    large = df["large_pred"].to_numpy()

    confidence = df["base_conf"].to_numpy()

    small_ms = df["base_ms"].to_numpy()
    large_ms = df["large_ms"].to_numpy()

    results = []

    summary = {
        "num_gold_pairs": len(df),
        "default_threshold": 0.60,
        "selected_threshold": SELECTED_THRESHOLD,
        "evaluation_scope": (
            "Gold SUPPORT/CONTRADICT rationale pairs "
            "on SciFact dev; no gold NEI examples."
        ),
        "latency_method": (
            "Estimated per-pair classifier latency from "
            "saved timings; excludes retrieval, startup and UI."
        ),
    }

    for tau in THRESHOLDS:

        # Configuration 1: Smaller model only
        if tau == 0.0:
            mode = "base_only"

            routed = np.zeros(len(df), dtype=bool)

            preds = small
            latency = small_ms
            large_usage = 0.0

        # Configuration 2: Larger model only
        elif tau > 1.0:
            mode = "large_only"

            routed = np.zeros(len(df), dtype=bool)

            preds = large

            # Standalone latency excludes smaller-model inference.
            latency = large_ms

            large_usage = 1.0

        # Configuration 3: Confidence-based two-tier routing
        else:
            mode = f"two_tier_{tau:.2f}"

            routed = confidence < tau

            preds = np.where(
                routed,
                large,
                small,
            )

            latency = small_ms + np.where(
                routed,
                large_ms,
                0.0,
            )

            large_usage = float(routed.mean())

        # Prediction outcome analysis
        decisive = preds != "NOT_ENOUGH_INFO"
        correct = preds == gold

        results.append({
            "threshold": tau,
            "mode": mode,

            "accuracy": round(
                float(correct.mean()),
                4,
            ),

            "macro_f1_support_contradict": round(
                float(
                    f1_score(
                        gold,
                        preds,
                        labels=LABELS[:2],
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            ),

            "escalation_rate": (
                None
                if mode == "large_only"
                else round(float(routed.mean()), 4)
            ),

            "nei_abstention_rate": round(
                float((~decisive).mean()),
                4,
            ),

            "avg_latency_ms": round(
                float(latency.mean()),
                2,
            ),

            "large_model_usage_rate": round(
                large_usage,
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

            "correct_count": int(
                correct.sum()
            ),

            "wrong_polarity_count": int(
                (decisive & ~correct).sum()
            ),

            "nei_count": int(
                (~decisive).sum()
            ),
        })

        # Detailed reports for the original and selected thresholds.
        if tau in (0.60, SELECTED_THRESHOLD):

            key = f"{tau:.2f}"

            summary[f"classification_report_at_{key}"] = (
                classification_report(
                    gold,
                    preds,
                    labels=LABELS,
                    output_dict=True,
                    zero_division=0,
                )
            )

            summary[f"confusion_matrix_at_{key}"] = (
                confusion_matrix(
                    gold,
                    preds,
                    labels=LABELS,
                ).tolist()
            )

    # Majority-class baseline
    majority = df["gold_label"].value_counts().idxmax()

    majority_preds = np.repeat(
        majority,
        len(df),
    )

    selected = next(
        r for r in results
        if r["threshold"] == SELECTED_THRESHOLD
    )

    standalone = next(
        r for r in results
        if r["mode"] == "large_only"
    )

    if standalone["avg_latency_ms"] <= 0:
        raise ValueError(
            "Standalone latency must be positive."
        )

    summary.update({

        "confusion_matrix_labels": LABELS,

        "threshold_sweep": results,

        "majority_baseline": {

            "label": majority,

            "accuracy": round(
                float(
                    accuracy_score(
                        gold,
                        majority_preds,
                    )
                ),
                4,
            ),

            "macro_f1_support_contradict": round(
                float(
                    f1_score(
                        gold,
                        majority_preds,
                        labels=LABELS[:2],
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            ),
        },

        "selected_latency_saving_percent": round(
            100 * (
                1
                - selected["avg_latency_ms"]
                / standalone["avg_latency_ms"]
            ),
            2,
        ),

    })

    return pd.DataFrame(results), summary


def main():
    """Update derived evaluation artifacts without rerunning inference."""

    raw_path = OUT_DIR / "stance_oracle_predictions.csv"

    summary_path = (
        OUT_DIR / "phase2_evaluation_summary.json"
    )

    df = pd.read_csv(raw_path)

    report = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    sweep, routing = summarize_routing(df)

    # Preserve retrieval, end-to-end and consistency results.
    report["stance_and_routing"] = routing

    sweep.to_csv(
        OUT_DIR / "routing_threshold_sweep.csv",
        index=False,
    )

    summary_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        sweep[[
            "mode",
            "accuracy",
            "macro_f1_support_contradict",
            "large_model_usage_rate",
            "avg_latency_ms",
        ]].to_string(index=False)
    )

    print(
        "\nSelected threshold:",
        SELECTED_THRESHOLD,
    )

    print(
        "Majority baseline:",
        routing["majority_baseline"],
    )

    print(
        "Selected latency saving (%):",
        routing["selected_latency_saving_percent"],
    )

    print(
        "Confusion matrix at 0.95:",
        routing["confusion_matrix_at_0.95"],
    )

    print("Raw model predictions unchanged.")


if __name__ == "__main__":
    main()
