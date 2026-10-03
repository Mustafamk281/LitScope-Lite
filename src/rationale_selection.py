"""Module 3: Rationale / Sentence-Level Evidence Selection for LitScope-Lite.

Identifies and extracts salient evidence sentences from retrieved candidate documents
given a scientific claim using dense semantic similarity with sentence-transformers.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from .segmentation import segment_claims


class RationaleSelector:
    """Sentence-level rationale selector for scientific claim verification.
    
    Scores each sentence in a document's abstract against the query claim using
    dense embeddings from sentence-transformers/all-MiniLM-L6-v2 and extracts
    the most relevant evidence sentences.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.40,
        top_k: int = 3,
        device: Optional[str] = None,
        model: Optional[SentenceTransformer] = None,
    ) -> None:
        """Initialize RationaleSelector.
        
        Args:
            model_name: HuggingFace model identifier.
            threshold: Minimum cosine similarity score threshold for a sentence to be selected (default: 0.40).
            top_k: Maximum number of rationale sentences to retain per document (default: 3).
            device: Torch device (e.g. 'cpu', 'cuda').
            model: Optional pre-loaded SentenceTransformer instance to share weights.
        """
        self.model_name = model_name
        self.threshold = threshold
        self.top_k = top_k
        self.device = device

        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(model_name, device=device)

    def select_rationales_from_sentences(
        self,
        claim: str,
        sentences: List[str],
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Compute sentence-level similarity against claim and select top rationales.
        
        Args:
            claim: Query scientific claim string.
            sentences: List of sentences comprising the document abstract.
            threshold: Optional threshold override (defaults to self.threshold).
            top_k: Optional top_k override (defaults to self.top_k).
            
        Returns:
            List of dicts for selected sentences sorted by sentence_idx:
                [
                    {
                        "sentence_idx": int,
                        "sentence": str,
                        "score": float
                    },
                    ...
                ]
        """
        if not claim or not sentences:
            return []

        th = self.threshold if threshold is None else threshold
        k = self.top_k if top_k is None else top_k

        # Encode claim and sentences
        claim_emb = self.model.encode(
            [claim],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        sent_embs = self.model.encode(
            sentences,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Pairwise cosine similarities
        scores = np.dot(sent_embs, claim_emb.T).flatten()

        # Candidates meeting threshold
        candidate_indices = [
            i for i, score in enumerate(scores) if score >= th
        ]

        # If no sentence meets threshold, fallback to the single highest-scoring sentence if available
        if not candidate_indices and len(scores) > 0 and k > 0:
            best_idx = int(np.argmax(scores))
            candidate_indices = [best_idx]

        # Sort candidate indices by score descending, take top_k
        candidate_indices.sort(key=lambda idx: scores[idx], reverse=True)
        selected_indices = candidate_indices[:k]

        # Return ordered by sentence_idx for natural text order
        selected_indices.sort()

        results = [
            {
                "sentence_idx": int(idx),
                "sentence": sentences[idx],
                "score": float(scores[idx]),
            }
            for idx in selected_indices
        ]
        return results

    def select_rationales_for_doc(
        self,
        claim: str,
        doc: Dict[str, Any],
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Extract rationales for a single retrieved document dictionary.
        
        Args:
            claim: Query scientific claim.
            doc: Document dict containing 'doc_id', 'abstract' (list or str), and optional 'title'.
            threshold: Optional threshold override.
            top_k: Optional top-k override.
            
        Returns:
            Dictionary with document ID, selected sentence details, and summary lists.
        """
        doc_id = doc.get("doc_id")
        abstract = doc.get("abstract", [])
        title = doc.get("title", "")

        # Format abstract into sentence list
        if isinstance(abstract, list):
            sentences = [s.strip() for s in abstract if s and s.strip()]
        elif isinstance(abstract, str):
            sentences = segment_claims(abstract)
        else:
            sentences = []

        selected = self.select_rationales_from_sentences(
            claim=claim,
            sentences=sentences,
            threshold=threshold,
            top_k=top_k,
        )

        return {
            "doc_id": doc_id,
            "title": title,
            "selected_sentences": selected,
            "sentence_indices": [s["sentence_idx"] for s in selected],
            "rationales": [s["sentence"] for s in selected],
        }

    def select_rationales_for_docs(
        self,
        claim: str,
        retrieved_docs: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Extract rationales for a list of retrieved candidate documents."""
        results = []
        for doc in retrieved_docs:
            results.append(
                self.select_rationales_for_doc(
                    claim=claim,
                    doc=doc,
                    threshold=threshold,
                    top_k=top_k,
                )
            )
        return results
