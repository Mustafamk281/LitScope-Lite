"""Module 2: Evidence Retrieval for LitScope-Lite.

Implements hybrid lexical-dense retrieval on the SciFact corpus by combining
TF-IDF (scikit-learn) and dense semantic representations (sentence-transformers/all-MiniLM-L6-v2)
using normalized reciprocal/equal-weight score fusion.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from .utils import get_data_path, load_jsonl, min_max_scale


class SciFactRetriever:
    """Hybrid Evidence Retriever for the SciFact dataset.
    
    Combines:
    1. Lexical Retrieval via scikit-learn TfidfVectorizer.
    2. Dense Retrieval via sentence-transformers/all-MiniLM-L6-v2 embeddings.
    3. Min-max normalization and equal-weight score fusion (0.5 * Lexical + 0.5 * Dense).
    """

    def __init__(
        self,
        corpus_path: Optional[Union[str, Path]] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        """Initialize and index the SciFact corpus.
        
        Args:
            corpus_path: Path to SciFact corpus.jsonl. If None, auto-resolved via get_data_path.
            model_name: HuggingFace model name for dense embeddings.
            cache_dir: Directory to save/load precomputed dense embeddings.
            use_cache: Whether to use cached dense embeddings on disk if available.
            batch_size: Batch size for dense embedding computation.
            device: Torch device ('cpu', 'cuda', etc.).
        """
        if corpus_path is None:
            self.corpus_path = get_data_path("corpus.jsonl")
        else:
            self.corpus_path = Path(corpus_path)

        self.model_name = model_name
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.device = device

        # Set default cache path
        if cache_dir is None:
            cache_dir = self.corpus_path.parent
        self.cache_dir = Path(cache_dir)
        safe_model_name = self.model_name.replace("/", "_").replace("\\", "_")
        self.embeddings_cache_file = self.cache_dir / f"corpus_embeddings_{safe_model_name}.npy"

        # State attributes
        self.corpus: List[Dict[str, Any]] = []
        self.doc_ids: List[int] = []
        self.corpus_texts: List[str] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.dense_model: Optional[SentenceTransformer] = None
        self.corpus_embeddings: Optional[np.ndarray] = None

        # Load corpus and build indices
        self._load_corpus()
        self._build_lexical_index()
        self._build_dense_index()

    def _load_corpus(self) -> None:
        """Load document records from corpus.jsonl."""
        records = load_jsonl(self.corpus_path)
        self.corpus = []
        self.doc_ids = []
        self.corpus_texts = []

        for item in records:
            doc_id = int(item["doc_id"])
            title = (item.get("title") or "").strip()
            abstract = item.get("abstract") or []
            
            if isinstance(abstract, list):
                abstract_str = " ".join(abstract).strip()
            else:
                abstract_str = str(abstract).strip()

            if title and abstract_str:
                full_text = f"{title}. {abstract_str}"
            elif title:
                full_text = title
            else:
                full_text = abstract_str

            self.corpus.append({
                "doc_id": doc_id,
                "title": title,
                "abstract": abstract,
                "full_text": full_text,
            })
            self.doc_ids.append(doc_id)
            self.corpus_texts.append(full_text)

    def _build_lexical_index(self) -> None:
        """Build scikit-learn TF-IDF index over corpus documents."""
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_df=0.85,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus_texts)

    def _build_dense_index(self) -> None:
        """Build or load dense embeddings for corpus documents."""
        self.dense_model = SentenceTransformer(self.model_name, device=self.device)

        # Check for cached embeddings
        if self.use_cache and self.embeddings_cache_file.exists():
            try:
                cached = np.load(self.embeddings_cache_file)
                if cached.shape[0] == len(self.corpus_texts):
                    self.corpus_embeddings = cached
                    return
            except Exception:
                # If loading corrupted cache fails, recompute
                pass

        # Compute embeddings with L2 normalization for fast dot-product cosine similarity
        self.corpus_embeddings = self.dense_model.encode(
            self.corpus_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Save to cache if enabled
        if self.use_cache:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(self.embeddings_cache_file, self.corpus_embeddings)
            except Exception:
                pass

    def retrieve_top_k(
        self,
        claim: str,
        k: int = 5,
        lexical_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k most relevant documents for a given claim.
        
        Performs:
        1. Lexical cosine similarity via TF-IDF dot product.
        2. Dense cosine similarity via MiniLM embedding dot product.
        3. Min-max normalization for each score array.
        4. Equal-weight fusion (0.5 * TF-IDF + 0.5 * Dense score).
        5. Ranking and extracting the top-k results.
        
        Args:
            claim: Query claim text.
            k: Number of documents to retrieve.
            lexical_weight: Weight assigned to lexical scores (default: 0.5).
            dense_weight: Weight assigned to dense scores (default: 0.5).
            
        Returns:
            List of dictionaries containing:
                - doc_id: int
                - score: float (fused score in [0, 1])
                - lexical_score: float (raw lexical similarity)
                - dense_score: float (raw dense similarity)
                - title: str
                - abstract: list of sentences or str
        """
        if not claim or not claim.strip():
            return []

        # 1. Lexical scoring
        claim_tfidf = self.tfidf_vectorizer.transform([claim])
        lex_scores = (claim_tfidf * self.tfidf_matrix.T).toarray().flatten()

        # 2. Dense scoring
        claim_emb = self.dense_model.encode(
            [claim],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        dense_scores = np.dot(self.corpus_embeddings, claim_emb.T).flatten()

        # 3. Min-Max Score Normalization
        norm_lex = min_max_scale(lex_scores)
        norm_dense = min_max_scale(dense_scores)

        # 4. Equal-weight Fusion (0.5 * TF-IDF + 0.5 * Dense)
        fused_scores = (lexical_weight * norm_lex) + (dense_weight * norm_dense)

        # 5. Top-k Ranking
        top_k_indices = np.argsort(fused_scores)[::-1][:k]

        results = []
        for idx in top_k_indices:
            doc = self.corpus[idx]
            results.append({
                "doc_id": self.doc_ids[idx],
                "score": float(fused_scores[idx]),
                "lexical_score": float(lex_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "title": doc.get("title", ""),
                "abstract": doc.get("abstract", []),
            })

        return results

    def get_top_k_doc_ids(self, claim: str, k: int = 5) -> List[int]:
        """Convenience method returning just the list of top-k document IDs."""
        return [res["doc_id"] for res in self.retrieve_top_k(claim, k=k)]

