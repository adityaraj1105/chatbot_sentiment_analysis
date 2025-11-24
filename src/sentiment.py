"""
Robust Sentiment Module (HF transformer first, VADER fallback).

Usage examples:
    # CPU using HF model (if available):
    s = SentimentModule(hf_model="cardiffnlp/twitter-roberta-base-sentiment", device=-1, debug=True)

    # Force VADER fallback (useful when offline/testing):
    s = SentimentModule(use_hf=False)
"""
from typing import Dict, List, Tuple, Optional

# HF imports are optional; we check availability at import time.
try:
    from transformers import pipeline, AutoConfig  # type: ignore
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# VADER fallback
try:
    from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
except Exception:
    # Delay raising detailed error until runtime when VADER is actually needed.
    SentimentIntensityAnalyzer = None  # type: ignore

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SentimentModule:
    """
    SentimentModule provides:
      - score_text: returns dict {'neg','neu','pos','compound'}
      - score_text_ensemble: combines HF and VADER compounds (if both available)
      - classify_compound: map compound -> label (Positive/Neutral/Negative)
      - aggregate_conversation: aggregate a list of compounds (mean or recency)
      - trend: coarse trend detection
    """

    def __init__(
        self,
        hf_model: Optional[str] = "cardiffnlp/twitter-roberta-base-sentiment",
        device: Optional[int] = -1,
        use_hf: bool = True,
        debug: bool = False,
    ):
        """
        Args:
            hf_model: Hugging Face model id for sentiment (if HF available).
            device: -1 for CPU, >=0 for specific CUDA device id.
            use_hf: attempt to use HF pipeline if True and HF is available.
            debug: if True, log raw HF outputs for debugging.
        """
        self.hf_model_id = hf_model
        self.debug = bool(debug)
        self.pipeline = None
        self.id2label = None
        self.analyzer = None

        # Decide whether to attempt HF pipeline
        self.use_hf = bool(use_hf) and HF_AVAILABLE

        if self.use_hf:
            try:
                # Try to load model config to inspect label mapping (id2label)
                try:
                    cfg = AutoConfig.from_pretrained(self.hf_model_id)
                    self.id2label = getattr(cfg, "id2label", None)
                except Exception:
                    self.id2label = None

                # Initialize HF pipeline (device expects -1 for CPU)
                self.pipeline = pipeline(
                    "sentiment-analysis", model=self.hf_model_id, tokenizer=self.hf_model_id, device=device
                )
                logger.info(f"Loaded HF pipeline for model: {self.hf_model_id}")
            except Exception as e:
                logger.warning(f"Could not initialize HF pipeline ({e}). Falling back to VADER.")
                self.use_hf = False
                self.pipeline = None

        # Initialize VADER fallback if HF not used / available
        if not self.use_hf:
            if SentimentIntensityAnalyzer is None:
                raise RuntimeError(
                    "VADER not available. Install nltk and download vader_lexicon:\n"
                    "    pip install nltk\n"
                    "    python -c \"import nltk; nltk.download('vader_lexicon')\""
                )
            try:
                self.analyzer = SentimentIntensityAnalyzer()
                logger.info("Using VADER sentiment analyzer (fallback).")
            except Exception as e:
                raise RuntimeError(
                    "VADER initialization failed. Run: python -c \"import nltk; nltk.download('vader_lexicon')\""
                ) from e

    # ---- Internal helpers ----
    def _map_label_from_config(self, label: str) -> str:
        """Map LABEL_X style labels to textual labels via config id2label, if available."""
        if not label:
            return label
        label_low = label.lower()
        if label_low.startswith("label_") and self.id2label:
            try:
                idx = int(label_low.split("_", 1)[1])
                mapped = self.id2label.get(idx, label)
                return mapped.lower() if isinstance(mapped, str) else label_low
            except Exception:
                return label_low
        return label_low

    def _hf_score_to_compound(self, hf_result: Dict) -> float:
        """
        Convert HF pipeline result (e.g. {'label': 'LABEL_1','score':0.98})
        into compound float in [-1, 1].
        """
        label = str(hf_result.get("label", "")).strip()
        score = float(hf_result.get("score", 0.0))

        label_mapped = self._map_label_from_config(label)

        if self.debug:
            logger.info(f"HF raw label='{label}' mapped='{label_mapped}' score={score:.4f}")

        # textual mapping
        if "positive" in label_mapped or label_mapped.startswith("pos"):
            return min(1.0, max(0.0, score))
        if "negative" in label_mapped or label_mapped.startswith("neg"):
            return -min(1.0, max(0.0, score))
        if "neutral" in label_mapped or label_mapped.startswith("neu"):
            return 0.0

        # heuristic for LABEL_0/LABEL_1/LABEL_2 ordering (common pattern)
        if label.lower().startswith("label_"):
            try:
                idx = int(label.split("_", 1)[1])
                if idx == 0:
                    return -score
                if idx == 2:
                    return score
                return 0.0
            except Exception:
                return 0.0

        # unknown -> neutral
        return 0.0

    # ---- Public API ----
    def score_text(self, text: str) -> Dict[str, float]:
        """
        Score a single text and return VADER-like dict: {'neg','neu','pos','compound'}.
        Tries HF first (if available) and falls back to VADER.
        """
        # Try HF if configured
        if self.use_hf and self.pipeline is not None:
            try:
                res = self.pipeline(text)
                # pipeline often returns a list for batch, handle both cases
                raw = res[0] if isinstance(res, list) and res else (res if isinstance(res, dict) else {"label": "", "score": 0.0})

                if self.debug:
                    logger.info(f"HF pipeline raw output: {raw}")

                compound = self._hf_score_to_compound(raw)

                # synthesize pos/neg/neu to approximate VADER format
                if compound > 0:
                    pos = float(compound)
                    neg = 0.0
                    neu = float(max(0.0, 1.0 - pos))
                elif compound < 0:
                    neg = float(abs(compound))
                    pos = 0.0
                    neu = float(max(0.0, 1.0 - neg))
                else:
                    pos = 0.0
                    neg = 0.0
                    neu = 1.0

                return {"neg": neg, "neu": neu, "pos": pos, "compound": compound}
            except Exception as e:
                logger.warning(f"HF scoring failed ({e}). Falling back to VADER for this call.")

        # VADER fallback path
        if self.analyzer is not None:
            return self.analyzer.polarity_scores(text)

        # defensive fallback: neutral
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    def score_text_ensemble(self, text: str) -> Dict[str, float]:
        """
        Ensemble HF + VADER compounds (average) when both are available for robustness.
        Returns same dict format.
        """
        hf_scores = None
        vader_scores = None

        # Try HF score (do not recursively call ensemble)
        if self.use_hf:
            try:
                # Use HF pipeline result if available
                if self.pipeline is not None:
                    res = self.pipeline(text)
                    raw = res[0] if isinstance(res, list) and res else (res if isinstance(res, dict) else {"label": "", "score": 0.0})
                    hf_compound = self._hf_score_to_compound(raw)
                    hf_scores = {"neg": max(0.0, -hf_compound), "neu": 1.0 - abs(hf_compound), "pos": max(0.0, hf_compound), "compound": float(hf_compound)}
            except Exception:
                hf_scores = None

        # VADER
        if self.analyzer is not None:
            try:
                vader_scores = self.analyzer.polarity_scores(text)
            except Exception:
                vader_scores = None

        # Combine
        if hf_scores is not None and vader_scores is not None:
            c = (hf_scores["compound"] + vader_scores["compound"]) / 2.0
        elif hf_scores is not None:
            c = hf_scores["compound"]
        elif vader_scores is not None:
            c = vader_scores["compound"]
        else:
            c = 0.0

        if c > 0:
            pos = float(c)
            neg = 0.0
            neu = float(max(0.0, 1.0 - pos))
        elif c < 0:
            neg = float(abs(c))
            pos = 0.0
            neu = float(max(0.0, 1.0 - neg))
        else:
            pos, neg, neu = 0.0, 0.0, 1.0

        return {"neg": neg, "neu": neu, "pos": pos, "compound": float(c)}

    def classify_compound(self, compound: float, pos_thresh: float = 0.1, neg_thresh: float = -0.1) -> str:
        """
        Map compound score to label using thresholds.
        Default thresholds are stricter (±0.1). You can pass custom thresholds.
        """
        if compound >= pos_thresh:
            return "Positive"
        if compound <= neg_thresh:
            return "Negative"
        return "Neutral"

    def aggregate_conversation(self, compounds: List[float], method: str = "mean") -> Tuple[float, str]:
        """
        Aggregate a list of compound scores into one compound + label.

        method: 'mean' or 'recency'
        """
        if not compounds:
            return 0.0, "Neutral"

        if method == "mean":
            agg = sum(compounds) / len(compounds)
        elif method == "recency":
            alpha = 0.8
            n = len(compounds)
            weights = [alpha ** (n - i - 1) for i in range(n)]
            total = sum(weights)
            agg = sum(c * w for c, w in zip(compounds, weights)) / total
        else:
            raise ValueError("Unknown aggregation method")

        label = self.classify_compound(agg)
        return agg, label

    def trend(self, compounds: List[float]) -> str:
        """
        Detect simple trend by comparing first-half mean vs second-half mean.
        Returns one of: 'Improving', 'Worsening', 'Stable'
        """
        if len(compounds) < 2:
            return "Stable"
        mid = len(compounds) // 2
        first = sum(compounds[:mid]) / max(1, len(compounds[:mid]))
        second = sum(compounds[mid:]) / max(1, len(compounds[mid:]))
        delta = second - first
        if delta > 0.05:
            return "Improving"
        if delta < -0.05:
            return "Worsening"
        return "Stable"
