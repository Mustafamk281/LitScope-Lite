"""LitScope-Lite Interactive Web Application (Streamlit)."""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path when launched via `streamlit run web/app.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import SciFactPipeline


st.set_page_config(
    page_title="LitScope-Lite | Scientific Claim Verification",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 LitScope-Lite: Scientific Claim Verification")
st.markdown(
    "Verify biomedical claims against the **AllenAI SciFact** corpus (5,183 abstracts) using "
    "**Hybrid Retrieval (TF-IDF + MiniLM)**, **Rationale Extraction**, and **Two-Tier NLI Routing**."
)


@st.cache_resource(show_spinner="Loading SciFact corpus, retriever, and DeBERTa NLI models...")
def load_pipeline() -> SciFactPipeline:
    return SciFactPipeline(retrieval_k=3, rationale_threshold=0.40, stance_threshold=0.95)


# Sidebar controls
st.sidebar.header("Pipeline Settings")
retrieval_k = st.sidebar.slider("Top-K Abstracts to Retrieve", min_value=1, max_value=10, value=3)
rationale_threshold = st.sidebar.slider("Rationale Similarity Threshold", min_value=0.10, max_value=0.80, value=0.40, step=0.05)
stance_threshold = st.sidebar.slider(
    "Two-Tier Escalation Threshold (τ)",
    min_value=0.40,
    max_value=0.99,
    value=0.95,
    step=0.05,
    help="Base model (deberta-v3-xsmall) escalates to deberta-v3-base when confidence < τ. Calibrated optimal: 0.95.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Example Claims")
examples = [
    "Vitamin D deficiency is associated with an increased risk of osteoporosis.",
    "Aspirin reduces the risk of myocardial infarction in high-risk patients.",
    "Antibiotics are effective against viral respiratory infections.",
    "0-dimensional biomaterials lack inductive properties. Regular aerobic exercise lowers blood pressure.",
]
selected_example = st.sidebar.selectbox("Load a sample claim:", ["(Custom input)"] + examples)

default_text = "" if selected_example == "(Custom input)" else selected_example
user_text = st.text_area(
    "Enter one or more scientific claims:",
    value=default_text,
    height=110,
    placeholder="Type or paste a scientific claim here...",
)

if st.button("Verify Claims", type="primary"):
    if not user_text.strip():
        st.warning("Please enter at least one scientific claim.")
    else:
        pipeline = load_pipeline()
        # Update thresholds dynamically without reloading model weights
        pipeline.retrieval_k = retrieval_k
        pipeline.rationale_threshold = rationale_threshold
        pipeline.classifier.threshold = stance_threshold

        with st.spinner("Retrieving evidence and classifying stance..."):
            report = pipeline.verify(user_text.strip())

        st.success(f"Detected **{report['total_claims_detected']}** claim(s).")

        badge_map = {
            "SUPPORT": ("🟢 SUPPORT", "green"),
            "CONTRADICT": ("🔴 CONTRADICT", "red"),
            "NOT_ENOUGH_INFO": ("🟡 NOT ENOUGH INFO", "orange"),
        }

        for i, item in enumerate(report["results"], 1):
            st.markdown(f"### Claim {i}: {item['claim']}")
            evidence_list = item["evidence"]

            if not evidence_list:
                st.info("No rationale sentences passed the similarity threshold in the top-K retrieved abstracts.")
                continue

            for ev in evidence_list:
                label_text, _ = badge_map.get(ev["predicted_stance"], (ev["predicted_stance"], "gray"))
                tier = "Tier 2 (deberta-v3-base)" if ev["model_used"] == "large_model" else "Tier 1 (deberta-v3-xsmall)"

                with st.expander(
                    f"{label_text} | Conf: {ev['confidence']:.2%} | Doc #{ev['doc_id']} — {ev['title']}",
                    expanded=True,
                ):
                    st.caption(f"Routing: **{tier}**")
                    st.markdown("**Extracted Rationale Sentences:**")
                    for sent in ev["extracted_rationales"]:
                        st.markdown(f"- {sent}")
