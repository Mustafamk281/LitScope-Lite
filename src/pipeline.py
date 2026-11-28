
"""End-to-end scientific claim verification pipeline."""

from threading import RLock
from typing import Any, Dict, Optional

from src.segmentation import segment_claims
from src.retrieval import SciFactRetriever
from src.rationale_selection import RationaleSelector
from src.stance import StanceClassifier


DEFAULT_RETRIEVAL_K = 3
DEFAULT_RATIONALE_THRESHOLD = 0.40
DEFAULT_STANCE_THRESHOLD = 0.95
RATIONALE_TOP_K = 3

DECISIVE_LABELS = {
    "SUPPORT",
    "CONTRADICT",
}


def summarize_claim_evidence(evidence):
    """Aggregate source-level predictions for one claim."""

    counts = {
        "SUPPORT": 0,
        "CONTRADICT": 0,
        "NOT_ENOUGH_INFO": 0,
    }

    for item in evidence:
        label = item["predicted_stance"]

        if label in counts:
            counts[label] += 1

    decisive = [
        item for item in evidence
        if item["predicted_stance"] in DECISIVE_LABELS
    ]

    conflict = (
        counts["SUPPORT"] > 0
        and counts["CONTRADICT"] > 0
    )

    selected_doc_id = None

    if decisive:

        best = max(
            decisive,
            key=lambda item: item["confidence"],
        )

        label = best["predicted_stance"]
        confidence = best["confidence"]
        selected_doc_id = best["doc_id"]

    else:

        label = "NOT_ENOUGH_INFO"

        confidence = (
            max(item["confidence"] for item in evidence)
            if evidence
            else None
        )

    return {
        "predicted_stance": label,

        "confidence": (
            round(float(confidence), 4)
            if confidence is not None
            else None
        ),

        "selected_doc_id": selected_doc_id,

        "cross_source_conflict": conflict,

        "stance_counts": counts,

        "total_evidence_sources": len(evidence),

        "aggregation_method": (
            "Highest-confidence decisive source; "
            "otherwise NOT_ENOUGH_INFO"
        ),
    }


class SciFactPipeline:
    """Integrates segmentation, retrieval and verification."""

    def __init__(
        self,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        rationale_threshold: float = DEFAULT_RATIONALE_THRESHOLD,
        stance_threshold: float = DEFAULT_STANCE_THRESHOLD,
    ):

        self._validate_settings(
            retrieval_k,
            rationale_threshold,
            stance_threshold,
        )

        print(
            "Initializing SciFact Verification Pipeline..."
        )

        self.retrieval_k = retrieval_k
        self.rationale_threshold = rationale_threshold

        # Protect shared models against simultaneous
        # inference requests with different settings.
        self._lock = RLock()

        # Module 2: Hybrid retrieval
        self.retriever = SciFactRetriever()

        # Module 3: Evidence sentence selection
        self.selector = RationaleSelector(
            model=self.retriever.dense_model,
            threshold=rationale_threshold,
            top_k=RATIONALE_TOP_K,
        )

        # Module 4: Confidence-based stance verification
        self.classifier = StanceClassifier(
            confidence_threshold=stance_threshold
        )

        print("Pipeline successfully initialized!")

    @staticmethod
    def _validate_settings(
        retrieval_k,
        rationale_threshold,
        stance_threshold,
    ):

        if retrieval_k < 1:
            raise ValueError(
                "retrieval_k must be at least 1."
            )

        if not 0.0 <= rationale_threshold <= 1.0:
            raise ValueError(
                "Rationale threshold must be between 0 and 1."
            )

        if not 0.0 <= stance_threshold <= 1.0:
            raise ValueError(
                "Stance threshold must be between 0 and 1."
            )

    def verify(
        self,
        text: str,
        *,
        retrieval_k: Optional[int] = None,
        rationale_threshold: Optional[float] = None,
        stance_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:

        """Verify input text against the SciFact corpus."""

        with self._lock:

            k = (
                self.retrieval_k
                if retrieval_k is None
                else retrieval_k
            )

            rationale_th = (
                self.rationale_threshold
                if rationale_threshold is None
                else rationale_threshold
            )

            stance_th = (
                self.classifier.threshold
                if stance_threshold is None
                else stance_threshold
            )

            self._validate_settings(
                k,
                rationale_th,
                stance_th,
            )

            old_threshold = self.classifier.threshold

            try:

                self.classifier.threshold = stance_th

                return self._verify_impl(
                    text,
                    k,
                    rationale_th,
                    stance_th,
                )

            finally:

                # Restore the original model settings
                # after every inference request.
                self.classifier.threshold = old_threshold

    def _verify_impl(
        self,
        text,
        retrieval_k,
        rationale_threshold,
        stance_threshold,
    ):

        # Module 1: Claim segmentation
        claims = segment_claims(text)

        report = {
            "input_text": text,

            "total_claims_detected": len(claims),

            "configuration": {
                "retrieval_k": retrieval_k,
                "rationale_threshold": rationale_threshold,
                "rationale_top_k": RATIONALE_TOP_K,
                "stance_threshold": stance_threshold,
                "retrieval_method": "TF-IDF + MiniLM",
            },

            "results": [],
        }

        for claim in claims:

            claim_result = {
                "claim": claim,
                "evidence": [],
            }

            # Module 2: Retrieve candidate abstracts
            retrieved_docs = (
                self.retriever.retrieve_top_k(
                    claim,
                    k=retrieval_k,
                )
            )

            claim_result["total_documents_retrieved"] = (
                len(retrieved_docs)
            )

            for doc in retrieved_docs:

                # Module 3: Select relevant sentences
                rationale_result = (
                    self.selector.select_rationales_for_doc(
                        claim=claim,
                        doc=doc,
                        threshold=rationale_threshold,
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

                # Module 4: Predict source-level stance
                stance_pred = (
                    self.classifier.predict_stance(
                        claim,
                        rationale_text,
                    )
                )

                claim_result["evidence"].append({
                    "doc_id": int(doc["doc_id"]),

                    "title": (
                        doc.get("title")
                        or "Unknown Title"
                    ),

                    "retrieval_score": float(
                        doc["score"]
                    ),

                    "extracted_rationales": (
                        rationale_result["rationales"]
                    ),

                    "predicted_stance": (
                        stance_pred["label"]
                    ),

                    # Preserve raw confidence for aggregation.
                    "confidence": float(
                        stance_pred["confidence"]
                    ),

                    "model_used": (
                        stance_pred["model_used"]
                    ),
                })

            # Module 5: Claim-level aggregation
            # and potential cross-source disagreement.
            claim_summary = summarize_claim_evidence(
                claim_result["evidence"]
            )

            claim_result.update(claim_summary)

            report["results"].append(claim_result)

        report["total_claims_with_conflicts"] = sum(
            item["cross_source_conflict"]
            for item in report["results"]
        )

        return report
