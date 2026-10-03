"""Module 4: Stance Classification for LitScope-Lite.

Uses HuggingFace AutoModelForSequenceClassification and AutoTokenizer
with cross-encoder/nli-deberta-v3-xsmall.

Label mapping (standard DeBERTa NLI):
    - Index 0: contradiction -> CONTRADICT
    - Index 1: entailment    -> SUPPORT
    - Index 2: neutral       -> NOT_ENOUGH_INFO
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class StanceClassifier:
    """NLI-based Stance Classifier using AutoModelForSequenceClassification."""

    # User-specified mapping: 0 -> CONTRADICT, 1 -> NOT_ENOUGH_INFO, 2 -> SUPPORT
    INDEX_TO_LABEL = {
        0: "CONTRADICT",
        1: "NOT_ENOUGH_INFO",
        2: "SUPPORT",
    }

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict_stance(self, claim: str, rationale: str) -> Dict[str, Any]:
        """Classify stance of rationale w.r.t. claim.

        Args:
            claim: Scientific claim (hypothesis).
            rationale: Evidence/rationale text (premise).

        Returns:
            dict with keys: label, confidence, probabilities.
        """
        if not claim or not rationale or not rationale.strip():
            return {
                "label": "NOT_ENOUGH_INFO",
                "confidence": 1.0,
                "probabilities": {"SUPPORT": 0.0, "CONTRADICT": 0.0, "NOT_ENOUGH_INFO": 1.0},
            }

        # NLI: (premise, hypothesis) = (rationale, claim)
        inputs = self.tokenizer(
            rationale, claim,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        prob_dict = {self.INDEX_TO_LABEL[i]: float(probs[i]) for i in range(len(probs)) if i in self.INDEX_TO_LABEL}
        predicted_label = max(prob_dict, key=prob_dict.get)

        return {
            "label": predicted_label,
            "confidence": float(prob_dict[predicted_label]),
            "probabilities": prob_dict,
        }

    def predict(self, claim: str, rationale: str) -> Dict[str, Any]:
        return self.predict_stance(claim, rationale)

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        return [self.predict_stance(c, r) for c, r in pairs]

    def predict_doc_stance(self, claim: str, doc_rationales: Union[List[str], str]) -> Dict[str, Any]:
        if isinstance(doc_rationales, list):
            doc_rationales = " ".join(doc_rationales)
        return self.predict_stance(claim, doc_rationales.strip())
