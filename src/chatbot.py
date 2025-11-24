"""
Chatbot engine: maintains history, provides simple replies, and wraps the SentimentModule.

Features:
- dependency injection for SentimentModule (easier testing)
- ensemble scoring by default (uses score_text_ensemble)
- configurable filler list and minimal-length filtering
- helpers: reset_history, export_transcript
- safe logging and defensive guards
"""
from typing import List, Dict, Optional
import logging

from .sentiment import SentimentModule
from .utils import preprocess_user_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Chatbot:
    def __init__(self, sentiment_module: Optional[SentimentModule] = None, ensemble_by_default: bool = True):
        """
        Args:
            sentiment_module: optional SentimentModule instance (injected for tests or custom configs).
            ensemble_by_default: whether to use score_text_ensemble() by default when analyzing messages.
        """
        self.history: List[Dict[str, str]] = []
        self.sentiment = sentiment_module if sentiment_module is not None else SentimentModule()
        self.ensemble_by_default = ensemble_by_default

        # configurable rules
        self.FILLER_SET = {"bye", "exit", "quit", "ok", "okay", "thanks", "thank you"}
        self.MIN_MESSAGE_LENGTH = 3  # messages shorter than this are treated as trivial

    # ---- Conversation helpers ----
    def user_message(self, text: str) -> Dict[str, str]:
        msg = {"role": "user", "text": text}
        self.history.append(msg)
        return msg

    def bot_reply(self, text: str) -> Dict[str, str]:
        msg = {"role": "bot", "text": text}
        self.history.append(msg)
        return msg

    def reset_history(self) -> None:
        """Clear conversation history."""
        self.history = []

    def export_transcript(self) -> List[Dict[str, str]]:
        """Return a copy of the conversation history (useful for exporting)."""
        return list(self.history)

    # ---- Simple response logic (replaceable) ----
    def simple_response(self, user_text: str) -> str:
        """
        Lightweight rule-based reply generator. Replace with any logic / LLM as needed.
        """
        if not user_text:
            return "Can you say that again?"

        text = user_text.lower()
        if any(w in text for w in ["help", "problem", "issue", "stuck", "error"]):
            return "I’m sorry to hear that — can you tell me more so I can help?"
        if any(w in text for w in ["thank", "thanks", "great", "awesome"]):
            return "You’re welcome! Anything else I can help with?"
        if any(w in text for w in ["bye", "exit", "quit"]):
            return "Goodbye!"
        return "Thanks for sharing. Can you elaborate?"

    # ---- Sentiment analysis ----
    def _score_message(self, text: str, ensemble: Optional[bool] = None) -> Dict[str, float]:
        """
        Score a single preprocessed text and return scores dict.
        Uses ensemble_by_default unless ensemble is explicitly False.
        """
        ensemble_flag = self.ensemble_by_default if ensemble is None else bool(ensemble)
        # choose the best available scoring API
        try:
            if ensemble_flag and hasattr(self.sentiment, "score_text_ensemble"):
                return self.sentiment.score_text_ensemble(text)
            return self.sentiment.score_text(text)
        except Exception as e:
            logger.warning(f"Scoring failed for text={text!r}: {e}. Returning neutral scores.")
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    def analyze_user_messages(self, aggregation_method: str = "mean", ensemble: Optional[bool] = None):
        """
        Compute per-message sentiment (and an aggregated conversation sentiment + trend).

        Behavior:
          - Skips trivially short messages (records them but marks as skipped).
          - Treats filler tokens (bye/exit/thanks) as skipped for aggregation.
          - Uses ensemble scoring by default (see ensemble_by_default).
        """
        user_msgs = [m for m in self.history if m.get("role") == "user"]
        per_message = []
        compounds: List[float] = []

        for m in user_msgs:
            raw = m.get("text", "")
            text = preprocess_user_text(raw)

            # trivial / too short
            if not text or len(text) < self.MIN_MESSAGE_LENGTH:
                scores = {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
                per_message.append({"text": raw, "preprocessed": text, "label": "Neutral", "scores": scores, "skipped": True})
                continue

            # filler tokens (do not count toward aggregate)
            if text.lower() in self.FILLER_SET:
                scores = self._score_message(text, ensemble=ensemble)
                label = self.sentiment.classify_compound(scores.get("compound", 0.0))
                per_message.append({"text": raw, "preprocessed": text, "label": label, "scores": scores, "skipped": True})
                continue

            # normal scoring
            scores = self._score_message(text, ensemble=ensemble)
            label = self.sentiment.classify_compound(scores.get("compound", 0.0))
            per_message.append({"text": raw, "preprocessed": text, "label": label, "scores": scores, "skipped": False})
            compounds.append(scores.get("compound", 0.0))

        # aggregate compounds (handle empty case)
        if not compounds:
            agg_compound, agg_label = 0.0, "Neutral"
        else:
            agg_compound, agg_label = self.sentiment.aggregate_conversation(compounds, method=aggregation_method)

        trend = self.sentiment.trend(compounds)
        return {"per_message": per_message, "aggregate": {"compound": agg_compound, "label": agg_label}, "trend": trend}
