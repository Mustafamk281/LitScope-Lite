"""Stage 4: End-to-End Verification Pipeline for LitScope-Lite."""

from typing import Dict, Any
from src.segmentation import segment_claims
from src.retrieval import SciFactRetriever
from src.rationale_selection import RationaleSelector
from src.stance import StanceClassifier

class SciFactPipeline:
    def __init__(self, retrieval_k: int = 3, rationale_threshold: float = 0.40, stance_threshold: float = 0.60):
        print("Initializing SciFact Verification Pipeline (This may take a moment)...")
        self.retrieval_k = retrieval_k
        self.rationale_threshold = rationale_threshold
        
        # 1. Initialize Retrieval (Module 2)
        self.retriever = SciFactRetriever()
        
        # 2. Initialize Rationale Selection (Module 3)
        self.selector = RationaleSelector(
            model=self.retriever.dense_model, 
            threshold=rationale_threshold, 
            top_k=3
        )
        
        # 3. Initialize Stance Classification (Module 4)
        self.classifier = StanceClassifier(confidence_threshold=stance_threshold)
        print("Pipeline successfully initialized!")

    def verify(self, text: str) -> Dict[str, Any]:
        """Runs the full end-to-end pipeline on raw input text."""
        
        # Step 1: Segment raw text into individual scientific claims
        claims = segment_claims(text)
        
        report = {
            "input_text": text,
            "total_claims_detected": len(claims),
            "results": []
        }
        
        for claim in claims:
            claim_result = {
                "claim": claim,
                "evidence": []
            }
            
            # Step 2: Retrieve Top Candidate Documents
            retrieved_docs = self.retriever.retrieve_top_k(claim, k=self.retrieval_k)
            
            for doc in retrieved_docs:
                # Step 3: Extract Rationale Sentences from the Document
                rationale_result = self.selector.select_rationales_for_doc(
                    claim=claim, doc=doc, threshold=self.rationale_threshold, top_k=3
                )
                
                # If no sentences passed the similarity threshold, skip this document
                if not rationale_result["rationales"]:
                    continue 
                    
                rationale_text = " ".join(rationale_result["rationales"]).strip()
                
                # Step 4: Classify the Stance of the extracted rationales against the claim
                stance_pred = self.classifier.predict_stance(claim, rationale_text)
                
                claim_result["evidence"].append({
                    "doc_id": doc["doc_id"],
                    "title": doc.get("title", "Unknown Title"),
                    "extracted_rationales": rationale_result["rationales"],
                    "predicted_stance": stance_pred["label"],
                    "confidence": round(stance_pred["confidence"], 4),
                    "model_used": stance_pred["model_used"]
                })
            
            report["results"].append(claim_result)
            
        return report