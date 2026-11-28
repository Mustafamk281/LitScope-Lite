
"""LitScope-Lite interactive Streamlit application."""

import json
import sys
import time
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.pipeline import (
    SciFactPipeline,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_RATIONALE_THRESHOLD,
    DEFAULT_STANCE_THRESHOLD,
)


st.set_page_config(
    page_title="LitScope-Lite | Scientific Claim Verification",
    page_icon="🔬",
    layout="wide",
)


LABEL_NAMES = {
    "SUPPORT": "SUPPORT",
    "CONTRADICT": "CONTRADICT",
    "NOT_ENOUGH_INFO": "NOT ENOUGH INFORMATION",
}


MODEL_NAMES = {
    "base_model": "Tier 1 — DeBERTa V3 XSmall",
    "large_model": "Tier 2 — DeBERTa V3 Base",
    "none": "No model inference",
}


EXAMPLES = [
    "Vitamin D deficiency is associated with an increased risk of osteoporosis.",

    "Aspirin reduces the risk of myocardial infarction in high-risk patients.",

    "Antibiotics are effective against viral respiratory infections.",

    "0-dimensional biomaterials lack inductive properties.",

    "0-dimensional biomaterials lack inductive properties. Regular aerobic exercise lowers blood pressure.",
]


@st.cache_resource(
    show_spinner="Loading SciFact corpus and pretrained models..."
)
def load_pipeline():

    return SciFactPipeline(
        retrieval_k=DEFAULT_RETRIEVAL_K,
        rationale_threshold=DEFAULT_RATIONALE_THRESHOLD,
        stance_threshold=DEFAULT_STANCE_THRESHOLD,
    )


def format_confidence(value):

    if value is None:
        return "Unavailable"

    return f"{value:.2%}"


def render_stance(label, confidence):

    label_text = LABEL_NAMES.get(
        label,
        label,
    )

    message = (
        f"{label_text} | "
        f"Model confidence: "
        f"{format_confidence(confidence)}"
    )

    if label == "SUPPORT":
        st.success(message)

    elif label == "CONTRADICT":
        st.error(message)

    else:
        st.warning(message)


def load_example():

    example = st.session_state["selected_example"]

    if example != "(Custom input)":

        st.session_state["claim_input"] = example

    else:

        st.session_state["claim_input"] = ""


def render_report(report, elapsed_seconds):

    results = report["results"]

    st.divider()

    st.header("Verification Results")

    total_sources = sum(
        len(item["evidence"])
        for item in results
    )

    total_escalated = sum(
        ev["model_used"] == "large_model"
        for item in results
        for ev in item["evidence"]
    )

    total_conflicts = report.get(
        "total_claims_with_conflicts",
        0,
    )

    columns = st.columns(4)

    columns[0].metric(
        "Claims detected",
        report["total_claims_detected"],
    )

    columns[1].metric(
        "Evidence sources",
        total_sources,
    )

    columns[2].metric(
        "Potential conflicts",
        total_conflicts,
    )

    columns[3].metric(
        "Processing time",
        f"{elapsed_seconds:.2f} s",
    )

    configuration = report["configuration"]

    st.caption(
        f"Configuration used: "
        f"Top-K = {configuration['retrieval_k']} | "
        f"Rationale threshold = "
        f"{configuration['rationale_threshold']:.2f} | "
        f"Routing threshold = "
        f"{configuration['stance_threshold']:.2f}"
    )

    if total_sources:

        escalation_rate = (
            total_escalated / total_sources
        )

        st.caption(
            f"Tier 2 model used for "
            f"{total_escalated}/{total_sources} "
            f"source predictions "
            f"({escalation_rate:.1%})."
        )

    st.caption(
        "Processing time excludes initial model loading "
        "and includes retrieval, evidence selection "
        "and stance inference."
    )

    if not results:

        st.info(
            "No usable claims were detected."
        )

    for index, item in enumerate(results, 1):

        st.divider()

        st.subheader(f"Claim {index}")

        st.write(item["claim"])

        st.markdown("**Aggregated model prediction**")

        render_stance(
            item["predicted_stance"],
            item["confidence"],
        )

        if item["selected_doc_id"] is not None:

            st.caption(
                "Selected source: SciFact document "
                f"#{item['selected_doc_id']}. "
                "The verdict is based on the highest-confidence "
                "decisive source prediction."
            )

        else:

            st.caption(
                "No retrieved source produced a "
                "decisive SUPPORT or CONTRADICT prediction."
            )

        counts = item["stance_counts"]

        count_cols = st.columns(3)

        count_cols[0].metric(
            "Supporting predictions",
            counts["SUPPORT"],
        )

        count_cols[1].metric(
            "Contradicting predictions",
            counts["CONTRADICT"],
        )

        count_cols[2].metric(
            "Insufficient information",
            counts["NOT_ENOUGH_INFO"],
        )

        if item["cross_source_conflict"]:

            st.warning(
                "Potential cross-source disagreement detected. "
                "Different retrieved sources received opposing "
                "stance predictions. This does not establish "
                "that the original scientific publications "
                "actually disagree. Manual review is recommended."
            )

        evidence_list = item["evidence"]

        if not evidence_list:

            st.info(
                "No usable evidence sentences were extracted "
                "from the retrieved abstracts."
            )

            continue

        st.markdown("**Retrieved evidence and source predictions**")

        for rank, ev in enumerate(evidence_list, 1):

            label = LABEL_NAMES.get(
                ev["predicted_stance"],
                ev["predicted_stance"],
            )

            confidence = format_confidence(
                ev["confidence"]
            )

            with st.expander(
                f"Source {rank} | "
                f"{label} | "
                f"{confidence} | "
                f"Document #{ev['doc_id']}",
                expanded=True,
            ):

                st.markdown("**Source title**")

                st.write(ev["title"])

                render_stance(
                    ev["predicted_stance"],
                    ev["confidence"],
                )

                st.caption(
                    f"SciFact document ID: "
                    f"{ev['doc_id']}"
                )

                st.caption(
                    f"Hybrid retrieval score: "
                    f"{ev['retrieval_score']:.4f}"
                )

                st.caption(
                    "Model used: "
                    + MODEL_NAMES.get(
                        ev["model_used"],
                        ev["model_used"],
                    )
                )

                st.markdown(
                    "**Extracted evidence sentences**"
                )

                for sentence in ev["extracted_rationales"]:

                    st.write("• " + sentence)

    st.divider()

    st.download_button(
        label="Download Verification Report (JSON)",
        data=json.dumps(
            {
                **report,
                "processing_time_seconds": round(
                    elapsed_seconds,
                    4,
                ),
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        file_name="litscope_verification_report.json",
        mime="application/json",
    )


# -------------------------------------------------
# Application header
# -------------------------------------------------

st.title("LitScope-Lite")

st.subheader(
    "Evidence-Based Scientific Claim Verification"
)

st.markdown(
    "A research prototype that retrieves relevant scientific "
    "abstracts from the SciFact corpus, extracts evidence "
    "sentences and predicts whether each source supports, "
    "contradicts or provides insufficient information "
    "about a claim."
)

st.info(
    "Research prototype only. Predictions and model confidence "
    "scores do not establish scientific truth. The system is "
    "limited to its benchmark corpus and should not be used "
    "as a substitute for expert scientific or medical judgment."
)


# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.header("Pipeline Settings")

retrieval_k = st.sidebar.slider(
    "Top-K Abstracts",
    min_value=1,
    max_value=10,
    value=DEFAULT_RETRIEVAL_K,
)

rationale_threshold = st.sidebar.slider(
    "Rationale Similarity Threshold",
    min_value=0.10,
    max_value=0.80,
    value=DEFAULT_RATIONALE_THRESHOLD,
    step=0.05,
    help=(
        "Selects relevant evidence sentences. "
        "If none meet the threshold, the highest-scoring "
        "sentence is retained as a fallback."
    ),
)

stance_threshold = st.sidebar.slider(
    "Two-Tier Routing Threshold",
    min_value=0.40,
    max_value=1.00,
    value=DEFAULT_STANCE_THRESHOLD,
    step=0.05,
    help=(
        "The smaller model escalates a prediction to "
        "the larger model when its confidence falls "
        "below this threshold. The default 0.95 was "
        "selected using development-set experiments."
    ),
)

st.sidebar.divider()

st.sidebar.subheader("Example Claims")

st.sidebar.selectbox(
    "Load an example:",
    ["(Custom input)"] + EXAMPLES,
    key="selected_example",
    on_change=load_example,
)

st.sidebar.divider()

with st.sidebar.expander("About the pipeline"):

    st.write("1. Rule-based claim segmentation")
    st.write("2. TF-IDF and MiniLM hybrid retrieval")
    st.write("3. Semantic evidence sentence selection")
    st.write("4. Two-tier DeBERTa NLI verification")
    st.write("5. Claim-level and cross-source analysis")

    st.caption(
        "Corpus: AllenAI SciFact, 5,183 abstracts."
    )


# -------------------------------------------------
# User input
# -------------------------------------------------

if "claim_input" not in st.session_state:

    st.session_state["claim_input"] = ""


user_text = st.text_area(
    "Enter one or more scientific claims:",
    key="claim_input",
    height=150,
    placeholder="Type or paste scientific claims here...",
)

st.caption(
    "For the best results, use English scientific claims "
    "related to the biomedical literature."
)


# -------------------------------------------------
# Execute verification
# -------------------------------------------------

if st.button(
    "Verify Claims",
    type="primary",
):

    if not user_text.strip():

        st.warning(
            "Please enter at least one scientific claim."
        )

    else:

        # Prevent old results from being confused
        # with a failed new verification.
        st.session_state.pop(
            "verification_report",
            None,
        )

        st.session_state.pop(
            "processing_time",
            None,
        )

        try:

            pipeline = load_pipeline()

            start = time.perf_counter()

            with st.spinner(
                "Retrieving abstracts and verifying claims..."
            ):

                report = pipeline.verify(
                    user_text.strip(),

                    retrieval_k=retrieval_k,

                    rationale_threshold=rationale_threshold,

                    stance_threshold=stance_threshold,
                )

            elapsed = time.perf_counter() - start

            st.session_state[
                "verification_report"
            ] = report

            st.session_state[
                "processing_time"
            ] = elapsed

        except Exception as exc:

            st.error(
                "Verification could not be completed. "
                "Check the dataset location, installed "
                "dependencies and pretrained model availability."
            )

            st.code(str(exc))


# -------------------------------------------------
# Persist results across Streamlit reruns
# -------------------------------------------------

if "verification_report" in st.session_state:

    saved_report = st.session_state[
        "verification_report"
    ]

    saved_time = st.session_state[
        "processing_time"
    ]

    saved_config = saved_report["configuration"]

    settings_changed = (
        saved_config["retrieval_k"] != retrieval_k
        or saved_config["rationale_threshold"]
        != rationale_threshold
        or saved_config["stance_threshold"]
        != stance_threshold
    )

    if (
        saved_report["input_text"] != user_text.strip()
        or settings_changed
    ):

        st.warning(
            "The displayed results belong to the previous "
            "input or configuration. Select Verify Claims "
            "to evaluate the current settings."
        )

    render_report(
        saved_report,
        saved_time,
    )
