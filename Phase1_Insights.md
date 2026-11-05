# LitScope-Lite Phase 1: Data Preparation and Exploratory Findings

**Project:** Evidence-Based Verification of Scientific Claims Using NLP  
**Data source:** SciFact scientific-claim verification benchmark  
**Scope:** Data auditing, cleaning, EDA, feature engineering, normalization and scaling  
**Status:** Reproducible Phase 1 data analysis; no predictive modeling or performance evaluation

## 1. Executive summary

This report documents the empirical findings from three reproducible notebooks: `01_data_cleaning.ipynb`, `02_exploratory_data_analysis.ipynb`, and `03_feature_scaling_preprocessing.ipynb`. All results are derived from the supplied SciFact corpus and official claim splits. Cleaning preserves source document IDs, abstract sentence positions, and evidence links. EDA describes evidence annotations, text-length characteristics, and possible quality risks. Numerical preprocessing produces distinct claim–document features and fits scalers exclusively on training examples..

## 2. Data structure and observation units

The corpus contains **5,183 documents and 45,952 sentence entries**. There are **809 training claims**, **300 development claims**, and **300 unlabeled test claims**. Raw corpus fields include `doc_id`, `title`, `abstract`, and `structured`. Claim fields include `id` and `claim`; train/development claims also supply citation and evidence annotations.

Three analysis units are kept separate:

1. **Claims:** 1,409 distinct source claim records across official splits. Claims may have no listed gold evidence.
2. **Gold rationale annotations:** 1,295 labeled evidence annotations across train and development splits, including 832 `SUPPORT` and 463 `CONTRADICT`. A claim may contribute more than one annotation.
3. **Cited claim–document pairs:** 1,258 unique pairs (919 train, 339 development). These include both annotated and unannotated pairs. These rows are used for numerical feature preprocessing.

The lack of a gold annotation is not a confirmed `NOT_ENOUGH_INFO` label. Test claims do not provide gold stance or cited-document lists in the supplied files; no such labels or pairs are invented.

## 3. Data quality and cleaning

### 3.1 Validation and normalization

All four JSONL files were parsed and examined for required-field completeness, document-ID uniqueness, empty titles/abstracts, cited-document availability, and legal evidence sentence indices. **No missing/invalid required values and no broken evidence references** were detected.

Text is normalized with **Unicode NFC** and whitespace standardization. **3,843 text strings** changed. The cleaning does not change gold labels, split membership, document identifiers, or sentence-list positions.

### 3.2 Review flags and special cases

| Quality indicator | Observed count | Treatment |
| --- | ---: | --- |
| Repeated sentence occurrences within abstracts | 808 | Flagged; preserved in source sentence lists |
| Abstract sentence entries of 1–2 whitespace-delimited words | 1,083 | Flagged; not deleted solely for being short |
| Abstract sentence entries above 46 words (1.5 × IQR upper fence) | 1,356 | Flagged for length review; not truncated |
| Sentence entries containing a replacement character | 3 | Flagged for possible extraction damage |
| Highly repetitive document IDs | 3 | Document IDs `12207167`, `36271512`, `86217760` preserved and identified for follow-up inspection |
| Exact cross-split shared claim texts | 2 train–dev, 1 train–test, 0 dev–test | Documented as potential later evaluation leakage |

Quality-flag counts **overlap**: one sentence may be both short and repeated. Extreme values do not necessarily represent erroneous data. In particular, deleting or reordering sentences without remapping gold evidence indices would create annotation errors, so the cleaning policy deliberately avoids destructive edits.

**Sentence-length quality distribution.** The distribution and upper IQR threshold identify unusually long sentence entries for inspection, not automatic removal.

## 4. Exploratory analysis and interpretation

### 4.1 Labeled evidence stance frequencies

Gold rationale annotations total **1,295**. The training split contains 616 `SUPPORT` and 341 `CONTRADICT` annotations; the development split contains 216 and 122, respectively. Combined, these are **832 (64.2%) support** and **463 (35.8%) contradiction** annotations.

**Gold evidence annotations by stance and split.** Both annotated splits contain more `SUPPORT` than `CONTRADICT` evidence. Future verification experiments should report performance separately by class and use an imbalance-aware aggregate measure such as macro-F1. These counts represent **annotations**, not disjoint claims.

### 4.2 Gold-evidence coverage

In training, **505 of 809 claims** have at least one gold evidence annotation and **304** have none listed. In development, **188 of 300** have gold evidence and **112** have none listed.

**Claim annotation coverage.** A substantial portion of each annotated split has no gold evidence listed. These records cannot be assigned an NEI class solely on the basis of an empty evidence dictionary; later supervised target construction needs an explicit labeling protocol.

### 4.3 Claim length distribution

The median claim length is **12 whitespace-delimited words** in each train, development, and test split. The distributions have tails extending beyond the typical short claim length.

**Claim word counts by split.** Claim lengths are comparatively short and similarly centered across the supplied splits. Length variability motivates checking token-budget assumptions but does not by itself establish classification difficulty.

### 4.4 Abstract length and sentence counts

The corpus median is **192 whitespace-delimited words per abstract** and **8 sentence entries per abstract**. Document lengths vary considerably, including extreme examples connected to extraction irregularities.

**Abstract word counts.** The distribution has a right tail. The graph restricts display to the 99th percentile for readability, rather than deleting the longest abstracts from the source dataset.

**Abstract sentence counts.** Most abstracts comprise a modest number of sentence entries, but some have implausibly large counts caused by malformed text segmentation. Sentence-count outliers remain intact and are identified by the cleaning audit.

### 4.5 Claim length across annotated stances

**Claim length by gold-annotation stance.** The `SUPPORT` and `CONTRADICT` length distributions overlap substantially. There is no clear descriptive evidence that claim length alone separates the two labels.

### 4.6 Rationale length

**1,210 of 1,295** labeled rationales contain **one evidence sentence** (93.4%). The remainder contain two to four sentences.

**Rationale sentence counts.** Single-sentence evidence predominates, supporting the relevance of sentence-level rationale analysis as a modeling design choice. The multi-sentence subset should not be discarded, and prevalence does not demonstrate retrieval performance.

### 4.7 Lexical overlap with gold rationale sentences

The median token-set Jaccard overlap between claim and gold rationale is approximately **0.105** for `SUPPORT` and **0.095** for `CONTRADICT`.

**Claim–rationale Jaccard distributions.** Both stance groups exhibit low typical token overlap and overlapping distributions. Lexical similarity can assist evidence candidate matching but cannot determine whether a sentence supports or contradicts a claim.

### 4.8 Descriptive rank correlations

The strongest absolute Spearman association in the gold-annotated descriptive features is **ρ ≈ 0.900** between abstract word count and abstract sentence count.

**Descriptive feature correlations.** Word and sentence counts contain related document-length information. Correlations in this table are observational and conditional on gold-annotated examples; they do not establish causal relationships, predictive usefulness, or independence between lexical and dense retrieval systems.

### 4.9 Cited-document counts

**Cited documents per claim.** Most training and development claims cite a single document, while a smaller subset cite multiple documents. This is a distribution of **citations**, not a ranking of model-retrieved documents or a guarantee that each cited document contains evidence.

### 4.10 Data-quality flag frequencies

**Sentence quality flags.** Long, very short, and repeated sentence entries are present. Categories overlap; their totals must not be combined into an affected-row count without deduplication.

## 5. Numerical features and preprocessing

### 5.1 Feature-table construction

The feature dataset consists of **1,258 distinct cited claim–document rows**: 919 training and 339 development. The notebook removes one duplicate cited reference occurrence from the derived development pair table; this does **not** alter the underlying cleaned JSONL sources. All 1,409 claims also receive separate claim-only descriptive features.

The 11 pair-level numerical variables are grouped as follows:

| Group | Feature names | Role |
| --- | --- | --- |
| Claim size/composition | `claim_word_count`, `claim_char_count`, `claim_numeric_token_ratio`, `claim_unique_token_ratio` | Claim length, numerals and lexical diversity |
| Abstract/title size/composition | `title_word_count`, `abstract_word_count`, `abstract_sentence_count`, `abstract_numeric_token_ratio` | Source document length and composition |
| Claim–document lexical relation | `claim_abstract_jaccard`, `claim_title_jaccard`, `tfidf_sparse_score` | Transparent lexical overlap and TF-IDF similarity |

The fixed corpus of titles and abstracts is used to construct the TF-IDF document index. This score is not a verification target. Gold stance, annotation status, IDs and split membership are retained for traceability but not used as numerical predictors. No aligned dense claim embeddings were supplied, so no dense cosine-score feature was estimated.

### 5.2 Raw feature distributions

**Unscaled training features.** Counts, ratios and lexical similarities have different native ranges and patterns. Applying one universal raw scale would complicate numerical comparison; the chart provides a descriptive basis for comparing transformations.

### 5.3 Count transformation

The preprocessing notebook applies `log1p` to five nonnegative count variables: claim words, claim characters, title words, abstract words, and abstract sentence counts. For example, **abstract word-count skewness decreases from approximately 1.352 to 0.472** after transformation, measured on the training pair table.

**Log transformation of count features.** Logarithmic compression reduces the extreme positive tail for selected count variables. It is a transparent transformation, not outlier deletion.

### 5.4 Train-fitted scaling

`StandardScaler` and `MinMaxScaler` are fitted using **only the 919 training pairs**. Fitted parameters are then applied unchanged to **339 development pairs**. The exported scaled CSV contains parallel standard-score and min–max representations of each feature. Test claim–document rows are not fabricated because the test files do not contain the cited-document IDs needed for this pair construction.

**Standard versus min–max scaling.** The graphs compare resulting feature locations and ranges. Linear scaling does not generally remove skewness, and neither method is claimed to improve verification accuracy without model experiments.

### 5.5 Feature correlation and distribution comparison

**Pair-feature Spearman correlations.** Text-length measures are associated with each other, while some overlap and TF-IDF features capture different descriptive aspects. Correlation alone does not justify removing, selecting, or weighting features.

**Training/development feature comparisons.** Selected features are compared across train and development after a transformation fitted exclusively on the training split. This supports detection of distribution differences before later evaluation but is not a robustness result.

The median TF-IDF similarity for cited training pairs is approximately **0.1272**, compared with **0.1224** for cited development pairs. These medians are descriptive of cited-pair sampling only and cannot be interpreted as retrieval accuracy, domain discrimination, or an optimal rejection threshold.

## 6. Key implications for later verification experiments

The Phase 1 observations suggest hypotheses and controls for subsequent modeling, not validated performance conclusions:

- **Class-imbalance-aware evaluation:** `SUPPORT` annotations exceed `CONTRADICT` annotations; report stance-level metrics, macro-F1 and confusion matrices when supervised experiments are conducted.
- **Careful negative examples:** Lack of a gold rationale is not a gold NEI label. Any negative-pair sampling or rejection-label construction must be explicitly defined and evaluated.
- **Evidence traceability:** Preserve original abstract sentence offsets and gold index mappings when extracting candidate rationales.
- **Text-length variability:** Inspect token truncation and long-abstract handling under the actual model tokenizer, rather than assuming IQR outliers should be discarded.
- **Lexical similarity limitations:** Word overlap cannot distinguish entailment from contradiction. Compare sparse, dense and hybrid retrievers only when aligned score data and a defined retrieval evaluation protocol are available.
- **Evaluation leakage checks:** Inspect exact cross-split claim overlap before using development/test performance for model-selection claims.
- **Scaling reproducibility:** Use training-fitted transformations for future supervised comparisons; avoid fitting feature transforms on development/evaluation data.

## 7. Reproducibility and research limitations

The notebooks retain original data structure and export machine-readable audit summaries. The raw inputs remain in `data/scifact/`; normalized copies reside in `data/processed/`. Key outputs include:

- `data/processed/cleaning_summary.json`
- `outputs/eda/eda_summary.json`
- `outputs/preprocessing/03_preprocessing_summary.json`
- `data/processed/scifact_feature_scaling_metadata.json`
- `data/processed/scifact_features_raw.csv`
- `data/processed/scifact_features_scaled.csv`
- `data/processed/scifact_scalers.joblib`

The Phase 1 analysis has several important boundaries. The data are a scientific-abstract benchmark, not a live literature crawl. Abnormal records remain flagged instead of being automatically repaired. Gold evidence annotations are not exhaustive sentence-level negative labels, and the test split lacks public gold annotations. The preprocessing sample is restricted to supplied cited pairs. Numeric feature scaling is not proof of an improved retriever, classifier, or out-of-domain abstention decision. No model training, retrieval metric computation, threshold optimization, or uncertainty calibration is reported here.

## 8. Data source

Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., and Hajishirzi, H. (2020). *Fact or Fiction: Verifying Scientific Claims*. EMNLP. [AllenAI SciFact repository](https://github.com/allenai/scifact) · [Dataset schema](https://github.com/allenai/scifact/blob/master/doc/data.md) · [Data licenses](https://github.com/allenai/scifact/blob/master/LICENSE.md).

## 9. Supporting visualizations

The corresponding plots and supporting visual evidence are available in the repository's `images/` directory. The figures are organized by analysis stage and can be consulted to verify the distributions, comparisons, quality flags, and correlations discussed in this report. No figures are embedded in this document.

**Data cleaning:** `01_cleaning_sentence_length_distribution.png`.

**Exploratory data analysis:** `02_01_gold_evidence_stance_counts.png`, `02_02_claim_annotation_coverage.png`, `02_03_claim_word_count_distribution.png`, `02_04_abstract_word_count_distribution.png`, `02_05_abstract_sentence_count_distribution.png`, `02_06_claim_length_by_annotated_stance.png`, `02_07_rationale_sentence_counts_by_stance.png`, `02_08_claim_rationale_lexical_overlap.png`, `02_09_meta_feature_spearman_heatmap.png`, `02_10_cited_documents_per_claim.png`, and `02_11_sentence_quality_flag_counts.png`.

**Feature engineering and preprocessing:** `03_01_unscaled_feature_distributions.png`, `03_02_count_log_transformation.png`, `03_03_standard_vs_minmax.png`, `03_04_feature_spearman_correlations.png`, and `03_05_train_dev_feature_comparison.png`.
