"""
webapp.py — Streamlit UI for Chatbot with Sentiment, Emotion & Sarcasm Analysis

Place at project root and run:
    pip install -r requirements.txt
    streamlit run webapp.py
"""
import json
import time
from typing import List, Dict, Optional

import streamlit as st

# ---- Import project modules (fail gracefully) ----
try:
    from src.chatbot import Chatbot
    from src.sentiment import SentimentModule
    from src.utils import preprocess_user_text
    from src.sarcasm import SarcasmDetector
except Exception as e:
    st.error(
        "Failed to import project modules. Make sure you run this from the project root "
        "and that src/ contains chatbot.py, sentiment.py, utils.py, sarcasm.py.\n\n"
        f"Import error: {e}"
    )
    st.stop()

# ---- Optional: emotion & sarcasm pipeline (transformers). Load lazily and cache ----
EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
SARCASM_MODEL_ID = "mohitjain/sarcasm-roberta"


@st.cache_resource(show_spinner=False)
def load_emotion_pipe(model_id: str):
    try:
        from transformers import pipeline  # lazy import

        pipe = pipeline("text-classification", model=model_id, return_all_scores=True, device=-1)
        return pipe
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_sarcasm_detector(model_id: Optional[str]):
    """
    Returns a SarcasmDetector instance.
    If HF model can't be loaded internally, the detector will fallback to heuristic.
    """
    try:
        # SarcasmDetector handles HF availability internally
        return SarcasmDetector(use_hf=True, model_id=model_id, device=-1)
    except Exception:
        # last-resort: objective heuristic-only detector
        return SarcasmDetector(use_hf=False, model_id=None, device=-1)


# ---- Helpers -----------------------------------------------------------------
def ensure_state():
    """Ensure session_state has bot, sent_mod and messages."""
    if "bot" not in st.session_state:
        sent_mod = SentimentModule(debug=False, use_hf=True)
        st.session_state.bot = Chatbot(sentiment_module=sent_mod, ensemble_by_default=True)
        st.session_state.sent_mod = sent_mod
    if "messages" not in st.session_state:
        st.session_state.messages = []  # each record: role,text,preprocessed,scores,label,emotion,sarcasm,skipped,ts


def clear_history():
    """Clear conversation history but keep sentiment module loaded."""
    st.session_state.messages = []
    if "input_box" in st.session_state:
        st.session_state["input_box"] = ""


def score_and_record(text: str, ensemble: bool = False, emotion: bool = False, sarcasm_enabled: bool = False) -> Dict:
    """
    Score a single user text and return dict with scores, label, emotion and sarcasm info.
    If sarcasm_enabled is True, detect sarcasm and conservatively adjust positive compounds.
    """
    sent_mod: SentimentModule = st.session_state.sent_mod
    t = preprocess_user_text(text)

    # Score via HF/VADER (ensemble optional)
    try:
        if ensemble and hasattr(sent_mod, "score_text_ensemble"):
            scores = sent_mod.score_text_ensemble(t)
        else:
            scores = sent_mod.score_text(t)
    except Exception as e:
        st.warning(f"Scoring error: {e}. Returning neutral score.")
        scores = {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    label = sent_mod.classify_compound(scores.get("compound", 0.0))

    # Emotion
    emotion_label = None
    emotion_scores = None
    if emotion and t.strip():
        pipe = load_emotion_pipe(EMOTION_MODEL_ID)
        if pipe is not None:
            try:
                raw = pipe(t, truncation=True)
                if isinstance(raw, list) and raw:
                    scored = raw[0]
                    best = max(scored, key=lambda x: x.get("score", 0.0))
                    emotion_label = best.get("label")
                    emotion_scores = {d["label"]: d["score"] for d in scored}
            except Exception:
                emotion_label = None
                emotion_scores = None

    # Sarcasm detection & optional adjustment
    sarcasm_info = None
    if sarcasm_enabled:
        detector = load_sarcasm_detector(SARCASM_MODEL_ID)
        try:
            sarcasm_info = detector.detect(t)
        except Exception as e:
            sarcasm_info = {"sarcasm": False, "score": 0.0, "method": "error", "raw": str(e)}

        # Adjust sentiment conservatively if sarcasm likely and original is positive
        if sarcasm_info and sarcasm_info.get("sarcasm") and scores is not None:
            orig = float(scores.get("compound", 0.0))
            sar_score = float(sarcasm_info.get("score", 0.0))
            # adjust only when original positive and sarcasm confidence reasonable
            if orig > 0 and sar_score >= 0.5:
                # invert and attenuate strength (empirical factor)
                new_comp = -abs(orig) * 0.75
                if new_comp > 0:
                    pos = float(new_comp); neg = 0.0; neu = max(0.0, 1.0 - pos)
                elif new_comp < 0:
                    neg = float(abs(new_comp)); pos = 0.0; neu = max(0.0, 1.0 - neg)
                else:
                    pos = neg = 0.0; neu = 1.0
                scores = {"neg": neg, "neu": neu, "pos": pos, "compound": new_comp}
                label = sent_mod.classify_compound(new_comp)

    return {
        "text": text,
        "preprocessed": t,
        "scores": scores,
        "label": label,
        "emotion": emotion_label,
        "emotion_scores": emotion_scores,
        "sarcasm": sarcasm_info,
        "ts": time.time(),
    }


def plot_compounds(compounds: List[float]):
    """Return a matplotlib figure plotting compounds over message index."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 2.5))
    if compounds:
        ax.plot(range(1, len(compounds) + 1), compounds, marker="o")
    ax.axhline(0.0, linestyle="--", linewidth=0.7)
    ax.set_xlabel("Message #")
    ax.set_ylabel("Compound score")
    ax.set_title("Sentiment timeline (compound)")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig


# --- App layout --------------------------------------------------------------
st.set_page_config(page_title="Chatbot + Sentiment Demo", layout="wide")
st.title("Chatbot + Sentiment Analysis — Streamlit Demo")

ensure_state()
bot: Chatbot = st.session_state.bot

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    ensemble = st.checkbox("Use ensemble (HF + VADER, if available)", value=True)
    use_emotion = st.checkbox("Show emotion labels (if available)", value=False)
    sarcasm_enabled = st.checkbox("Enable sarcasm detection (adjust sentiment)", value=False)
    agg_method = st.selectbox("Aggregation method for overall sentiment", options=["mean", "recency"], index=0)
    stricter_thresholds = st.checkbox("Use stricter classification thresholds (±0.1)", value=False)
    debug = st.checkbox("Enable Sentiment debug (sent_mod.debug)", value=False)

    # Apply debug toggle to sentiment module (if present)
    try:
        st.session_state.sent_mod.debug = debug
    except Exception:
        pass

    st.markdown("---")
    st.markdown("Export / Import")
    if st.button("Export transcript (JSON)"):
        export = {"history": st.session_state.messages}
        st.download_button(
            label="Download transcript JSON",
            data=json.dumps(export, indent=2),
            file_name="transcript.json",
            mime="application/json",
        )

    st.markdown("")
    # Reset conversation button
    if st.button("Reset conversation"):
        clear_history()
        st.rerun()

    st.caption("Model info:")
    try:
        sm = st.session_state.sent_mod
        model_id = getattr(sm, "hf_model_id", "n/a")
        st.write(f"Sentiment model: {model_id}")
    except Exception:
        st.write("Sentiment model: unknown")

# Main columns: left = chat, right = analysis
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Conversation")
    # Display message history
    for i, m in enumerate(st.session_state.messages):
        role = m.get("role", "user")
        txt = m.get("text", "")
        if role == "user":
            st.markdown(f"**User:** {txt}")
            meta = f"Sentiment: {m.get('label', 'N/A')} (compound={m.get('scores', {}).get('compound', 0.0):.3f})"

            # Emotion info
            if use_emotion and m.get("emotion"):
                meta += f" • Emotion: {m.get('emotion')}"

            # Sarcasm info
            if m.get("sarcasm"):
                try:
                    meta += (
                        f" • Sarcasm: {m['sarcasm'].get('sarcasm')} "
                        f"(score={m['sarcasm'].get('score', 0.0):.2f})"
                    )
                except Exception:
                    pass

            if m.get("skipped"):
                meta += " [skipped]"
            st.caption(meta)
        else:
            st.markdown(f"**Bot:** {txt}")

    st.write("---")

    # Input area (callback-based to avoid session_state post-widget assignment errors)
    def handle_send(ensemble_flag: bool, use_emotion_flag: bool, sarcasm_flag: bool):
        """Button callback: process the message, score it and clear input_box."""
        user_text = st.session_state.get("input_box", "").strip()
        if not user_text:
            return
        # record user message
        st.session_state.messages.append({"role": "user", "text": user_text})
        # scoring (pass sarcasm flag)
        scored = score_and_record(user_text, ensemble=ensemble_flag, emotion=use_emotion_flag, sarcasm_enabled=sarcasm_flag)
        scored["skipped"] = scored.get("scores", {}).get("compound", 0.0) == 0.0
        st.session_state.messages[-1].update(scored)
        # generate bot reply & record
        reply_text = bot.simple_response(preprocess_user_text(user_text))
        st.session_state.messages.append({"role": "bot", "text": reply_text})
        # Clear the input field by mutating session_state inside this callback
        st.session_state["input_box"] = ""

    # render text_input with key="input_box" (binds to session_state)
    user_input = st.text_input("Your message:", key="input_box")
    st.button("Send", on_click=handle_send, args=(ensemble, use_emotion, sarcasm_enabled))

with col2:
    st.subheader("Analysis")
    user_entries = [m for m in st.session_state.messages if m.get("role") == "user"]
    # Aggregate only non-skipped entries
    compounds = [m.get("scores", {}).get("compound", 0.0) for m in user_entries if not m.get("skipped", False)]

    if user_entries:
        # Per-message table (includes sarcasm info)
        rows = []
        for idx, m in enumerate(user_entries, start=1):
            sarcasm_obj = m.get("sarcasm") or {}
            sarcasm_flag = bool(sarcasm_obj.get("sarcasm")) if isinstance(sarcasm_obj, dict) else False
            sarcasm_score = sarcasm_obj.get("score", None) if isinstance(sarcasm_obj, dict) else None

            rows.append(
                {
                    "msg#": idx,
                    "text": m.get("preprocessed", m.get("text")),
                    "label": m.get("label"),
                    "compound": round(m.get("scores", {}).get("compound", 0.0), 3),
                    "emotion": m.get("emotion") or "-",
                    "sarcasm": sarcasm_flag,
                    "sarcasm_score": round(sarcasm_score, 3) if sarcasm_score is not None else "-",
                    "skipped": m.get("skipped", False),
                }
            )
        st.table(rows)

        # Plot timeline (show all compounds including skipped as 0.0 so timeline aligns)
        all_compounds = [m.get("scores", {}).get("compound", 0.0) for m in user_entries]
        fig = plot_compounds(all_compounds)
        st.pyplot(fig)

        # Compute aggregate & trend using sentiment module
        sent_mod = st.session_state.sent_mod

        # optionally use stricter thresholds by temporarily wrapping classify_compound
        classify_fn = None
        if stricter_thresholds:
            def stricter(comp):
                if comp >= 0.1:
                    return "Positive"
                if comp <= -0.1:
                    return "Negative"
                return "Neutral"
            classify_fn = sent_mod.classify_compound
            sent_mod.classify_compound = stricter

        if not compounds:
            agg_compound, agg_label = 0.0, "Neutral"
        else:
            agg_compound, agg_label = sent_mod.aggregate_conversation(compounds, method=agg_method)
        trend = sent_mod.trend(compounds)

        st.markdown(f"**Overall conversation sentiment:** {agg_label}  \n**Aggregate score:** {agg_compound:.3f}  \n**Trend:** {trend}")

        # restore classify if monkey-patched
        if classify_fn is not None:
            sent_mod.classify_compound = classify_fn

    else:
        st.info("No user messages yet. Type something on the left to start.")

# Footer / tips
st.markdown("---")
st.caption(
    "Tips: enable 'Use ensemble' to combine HF and VADER scores (if implemented). "
    "Enable 'Show emotion labels' if an emotion model is available. "
    "Enable sarcasm detection to help correct false-positive sentiment from sarcastic text."
)
