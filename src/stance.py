"""Module 4: Stance Classification for LitScope-Lite.

Uses a two-tier confidence-based routing architecture with HuggingFace 
AutoModelForSequenceClassification and AutoTokenizer.

Label mapping (standard DeBERTa NLI):
    - Index 0: contradiction -> CONTRADICT
    - Index 1: entailment    -> SUPPORT
    - Index 2: neutral       -> NOT_ENOUGH_INFO
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class StanceClassifier:
    """NLI-based Stance Classifier with two-tier confidence routing."""

    # Fixed mapping: 0 -> CONTRADICT, 1 -> SUPPORT, 2 -> NOT_ENOUGH_INFO
    INDEX_TO_LABEL = {
        0: "CONTRADICT",
        1: "SUPPORT",
        2: "NOT_ENOUGH_INFO",
    }

    def __init__(
        self,
        base_model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        large_model_name: str = "cross-encoder/nli-deberta-v3-base",
        confidence_threshold: float = 0.60,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        self.max_length = max_length
        self.threshold = confidence_threshold
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load Base Model
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
        self.base_model.to(self.device)
        self.base_model.eval()

        # Load Large Model
        self.large_tokenizer = AutoTokenizer.from_pretrained(large_model_name)
        self.large_model = AutoModelForSequenceClassification.from_pretrained(large_model_name)
        self.large_model.to(self.device)
        self.large_model.eval()

    def _get_probs(self, model, tokenizer, claim: str, rationale: str) -> Tuple[Dict[str, float], str]:
        """Helper function to run inference on a specific model."""
        # NLI: (premise, hypothesis) = (rationale, claim)
        inputs = tokenizer(
            rationale, claim,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        prob_dict = {self.INDEX_TO_LABEL[i]: float(probs[i]) for i in range(len(probs)) if i in self.INDEX_TO_LABEL}
        predicted_label = max(prob_dict, key=prob_dict.get)
        
        return prob_dict, predicted_label

    def predict_stance(self, claim: str, rationale: str) -> Dict[str, Any]:
        """Classify stance of rationale w.r.t. claim using confidence routing."""
        if not claim or not rationale or not rationale.strip():
            return {
                "label": "NOT_ENOUGH_INFO",
                "confidence": 1.0,
                "probabilities": {"SUPPORT": 0.0, "CONTRADICT": 0.0, "NOT_ENOUGH_INFO": 1.0},
                "model_used": "none"
            }

        # 1. Base Model Inference
        prob_dict, predicted_label = self._get_probs(self.base_model, self.base_tokenizer, claim, rationale)
        confidence = prob_dict[predicted_label]
        model_used = "base_model"

        # 2. Confidence-Based Routing
        if confidence < self.threshold:
            prob_dict, predicted_label = self._get_probs(self.large_model, self.large_tokenizer, claim, rationale)
            confidence = prob_dict[predicted_label]
            model_used = "large_model"

        return {
            "label": predicted_label,
            "confidence": confidence,
            "probabilities": prob_dict,
            "model_used": model_used
        }

    def predict(self, claim: str, rationale: str) -> Dict[str, Any]:
        return self.predict_stance(claim, rationale)

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        return [self.predict_stance(c, r) for c, r in pairs]

    def predict_doc_stance(self, claim: str, doc_rationales: Union[List[str], str]) -> Dict[str, Any]:
        if isinstance(doc_rationales, list):
            doc_rationales = " ".join(doc_rationales)
        return self.predict_stance(claim, doc_rationales.strip())