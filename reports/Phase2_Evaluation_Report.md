# LitScope-Lite — Phase 2 Evaluation & Modeling Report

## 1. Pipeline Overview
LitScope-Lite verifies biomedical claims against the AllenAI SciFact corpus (5,183 scientific abstracts) through a four-module pipeline orchestrated by `src/pipeline.py` and exposed via `main.py`:

1. **Module 1 — Claim Segmentation (`src/segmentation.py`):** Rule-based scientific sentence splitter handling biomedical abbreviations (*et al.*, *i.e.*, *e.g.*), decimals, and multi-line input.
2. **Module 2 — Hybrid Evidence Retrieval (`src/retrieval.py`):** 50/50 min-max normalized fusion of sparse TF-IDF (n-grams) and dense `sentence-transformers/all-MiniLM-L6-v2` embeddings with `.npy` disk caching.
3. **Module 3 — Rationale Selection (`src/rationale_selection.py`):** Sentence-level cosine similarity filtering (`threshold = 0.40`, `top_k = 3`).
4. **Module 4 — Two-Tier Stance Classification (`src/stance.py`):** Zero-shot NLI routing from `cross-encoder/nli-deberta-v3-xsmall` (fast base tier) to `cross-encoder/nli-deberta-v3-base` (heavy tier) when base confidence falls below threshold $\tau$.

---

## 2. Quantitative Evaluation Results (SciFact Dev Set)

### 2.1 Hybrid Evidence Retrieval (188 Gold-Annotated Claims)
| Cutoff ($K$) | Recall@$K$ | Precision@$K$ |
| :--- | :---: | :---: |
| **$K = 1$** | 68.44% | 70.74% |
| **$K = 3$** | 83.26% | 29.26% |
| **$K = 5$** | **89.56%** | 19.04% |
| **$K = 10$** | **96.38%** | 10.53% |

* **MRR@10:** `0.7943` (evaluated across 188 claims in `11.45s` on CPU using cached dense embeddings).
* **Figure:** `images/04_01_retrieval_recall_precision_curve.png`

### 2.2 Stance Classification & Two-Tier Routing Ablation (338 Gold Claim-Rationale Pairs)
| Mode | Threshold ($\tau$) | Accuracy | Macro-F1 (SUPPORT/CONTRADICT) | Escalation Rate | Avg CPU Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `base_only` | 0.00 | 41.12% | 0.5712 | 0.00% | 99.34 |
| `two_tier_0.60` (default) | 0.60 | 40.53% | 0.5698 | 6.51% | 116.93 |
| `two_tier_0.80` | 0.80 | 42.31% | 0.5883 | 20.12% | 152.26 |
| **`two_tier_0.95` (calibrated)** | **0.95** | **44.67%** | **0.6106** | **45.86%** | **222.93** |
| `large_only` | 1.01 | 45.56% | 0.6250 | 100.00% | 365.58 |

* **Key Calibration Finding:** Because `nli-deberta-v3-xsmall` produces sharp softmax probabilities even on uncertain biomedical terminology, $\tau = 0.60$ escalates only 6.51% of pairs. Calibrating to **$\tau = 0.95$** recovers **98%** of `large_only` accuracy (`44.67%` vs. `45.56%`) while reducing average CPU inference latency by **39.0%** (`222.93 ms` vs. `365.58 ms`).
* **Figures:** `images/04_02_routing_accuracy_latency_tradeoff.png`, `images/04_03_stance_confusion_matrices.png`, `images/04_04_confidence_calibration.png`

### 2.3 End-to-End Pipeline Evaluation (60 Dev Claims, No Oracle Rationales)
* **Gold Document Recall@3:** `83.33%`
* **End-to-End Accuracy:** `45.00%`
* **End-to-End Macro-F1 (SUPPORT/CONTRADICT):** `0.5569`
* **NEI Abstention Rate:** `36.67%`
* **Cross-Source Conflict Cases Surfaced:** `4 / 60` claims where different top-3 abstracts produced opposing `SUPPORT` and `CONTRADICT` signals.
* **Figure:** `images/04_05_end_to_end_outcomes.png`

### 2.4 Controlled Claim-Edit Consistency Check (5 Biomedical Test Cases)
* **Paraphrase Consistency Rate:** **100% (5/5)** — semantic paraphrases preserved the original stance verdict.
* **Negation Flip Rate:** **100% (5/5)** — explicit negation/antonym edits cleanly flipped `SUPPORT` $\leftrightarrow$ `CONTRADICT`.

---

## 3. Error Analysis & Limitations
* **Zero-Shot Biomedical Domain Gap:** General-domain NLI cross-encoders were used without fine-tuning on `claims_train.jsonl`. Their dominant error mode is conservative abstention (`NOT_ENOUGH_INFO`) on indirect causal biomedical evidence rather than direct polarity reversal (`SUPPORT` $\leftrightarrow$ `CONTRADICT`).
* **Future Work:** Fine-tuning the Tier-2 model on SciFact training pairs (or swapping in a biomedical backbone such as PubMedBERT/BioLinkBERT NLI) and applying temperature scaling to base-tier logits would further reduce NEI abstentions and sharpen routing efficiency.
