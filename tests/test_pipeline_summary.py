
"""Tests for claim aggregation and cross-source comparison."""

from src.pipeline import summarize_claim_evidence


def make_evidence(doc_id, label, confidence):

    return {
        "doc_id": doc_id,
        "predicted_stance": label,
        "confidence": confidence,
    }


def test_supporting_sources():

    evidence = [
        make_evidence(1, "SUPPORT", 0.90),
        make_evidence(2, "SUPPORT", 0.95),
    ]

    result = summarize_claim_evidence(evidence)

    assert result["predicted_stance"] == "SUPPORT"

    assert result["selected_doc_id"] == 2

    assert result["confidence"] == 0.95

    assert result["cross_source_conflict"] is False

    assert result["stance_counts"]["SUPPORT"] == 2


def test_conflicting_sources():

    evidence = [
        make_evidence(1, "SUPPORT", 0.85),
        make_evidence(2, "CONTRADICT", 0.95),
        make_evidence(3, "NOT_ENOUGH_INFO", 0.90),
    ]

    result = summarize_claim_evidence(evidence)

    assert result["cross_source_conflict"] is True

    assert result["predicted_stance"] == "CONTRADICT"

    assert result["selected_doc_id"] == 2

    assert result["stance_counts"]["SUPPORT"] == 1

    assert result["stance_counts"]["CONTRADICT"] == 1

    assert result["stance_counts"]["NOT_ENOUGH_INFO"] == 1


def test_only_insufficient_information():

    evidence = [
        make_evidence(1, "NOT_ENOUGH_INFO", 0.80),
        make_evidence(2, "NOT_ENOUGH_INFO", 0.92),
    ]

    result = summarize_claim_evidence(evidence)

    assert result["predicted_stance"] == "NOT_ENOUGH_INFO"

    assert result["confidence"] == 0.92

    assert result["selected_doc_id"] is None

    assert result["cross_source_conflict"] is False


def test_empty_evidence():

    result = summarize_claim_evidence([])

    assert result["predicted_stance"] == "NOT_ENOUGH_INFO"

    assert result["confidence"] is None

    assert result["selected_doc_id"] is None

    assert result["cross_source_conflict"] is False

    assert result["total_evidence_sources"] == 0


def test_decisive_prediction_preferred():

    evidence = [
        make_evidence(1, "NOT_ENOUGH_INFO", 0.99),
        make_evidence(2, "SUPPORT", 0.75),
    ]

    result = summarize_claim_evidence(evidence)

    assert result["predicted_stance"] == "SUPPORT"

    assert result["confidence"] == 0.75

    assert result["selected_doc_id"] == 2


def test_contradicting_sources():

    evidence = [
        make_evidence(1, "CONTRADICT", 0.82),
        make_evidence(2, "CONTRADICT", 0.91),
    ]

    result = summarize_claim_evidence(evidence)

    assert result["predicted_stance"] == "CONTRADICT"

    assert result["confidence"] == 0.91

    assert result["cross_source_conflict"] is False
