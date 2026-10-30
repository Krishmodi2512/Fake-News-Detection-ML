"""Streamlit interface: python -m streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from fakenews import config  # noqa: E402
from fakenews.predict import DISCLAIMER, load_model, predict  # noqa: E402

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


@st.cache_resource
def get_model():
    return load_model()


st.title("Fake News Detector")
st.caption("Estimates whether a headline or article resembles content that is likely real or potentially fake.")
st.info(DISCLAIMER)

if not config.MODEL_PATH.exists():
    st.error(f"No trained model found at {config.MODEL_PATH}. Run `python scripts/train.py` first.")
    st.stop()

text = st.text_area(
    "Paste a headline or article",
    height=220,
    placeholder="Paste text here...",
)

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
        st.stop()

    try:
        result = predict(get_model(), text)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader(result.label)
    st.progress(result.confidence, text=f"Confidence: {result.confidence:.1%}")
    st.caption(f"Calibrated probability the text is fake: {result.p_fake:.1%}")

    if result.low_confidence:
        st.warning("Low confidence. The model is not sure either way, so treat this result as inconclusive.")
    for w in result.warnings:
        st.warning(w)

    st.markdown("### Style analysis")
    for flag in result.style["flags"]:
        st.write(f"- {flag}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Terms pushing toward fake**")
        if result.terms["toward_fake"]:
            st.table([{"term": t, "contribution": c} for t, c in result.terms["toward_fake"]])
        else:
            st.write("None")
    with col_b:
        st.markdown("**Terms pushing toward real**")
        if result.terms["toward_real"]:
            st.table([{"term": t, "contribution": c} for t, c in result.terms["toward_real"]])
        else:
            st.write("None")

    with st.expander("Model details"):
        st.write(
            "Features: TF-IDF over words and bigrams. Classifier: logistic regression with "
            "sigmoid (Platt) calibration. Contributions are feature weight times TF-IDF value, "
            "so they show which words the model relied on, not why they are true or false."
        )
