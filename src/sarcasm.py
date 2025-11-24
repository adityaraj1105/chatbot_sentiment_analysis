"""
Sarcasm detector module.

Behavior:
- Try to use a Hugging Face text-classification pipeline (configurable model_id).
- If HF isn't available or model load fails, fall back to a heuristic detector.
- API:
    d = SarcasmDetector(use_hf=True, model_id="mohitjain/sarcasm-roberta", device=-1)
    d.detect("Wow, great... this totally didn't waste my entire day.")
    -> {"sarcasm": True/False, "score": 0.0-1.0, "method": "hf" or "heuristic", "raw": <raw>}
"""

from typing import Dict, Optional
import re
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Try HF import lazily
try:
    from transformers import pipeline  # type: ignore
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


class SarcasmDetector:
    def __init__(self, use_hf: bool = True, model_id: Optional[str] = "mohitjain/sarcasm-roberta", device: int = -1, debug: bool = False):
        """
        Args:
            use_hf: attempt to use HF model if available
            model_id: HF model id to use (if available). You can set to None to force heuristic.
            device: -1 for CPU or >=0 for GPU id
            debug: enable logging of raw predictions
        """
        self.debug = bool(debug)
        self.model_id = model_id
        self.device = device
        self.use_hf = bool(use_hf) and HF_AVAILABLE and (model_id is not None)
        self.pipe = None

        if self.use_hf:
            try:
                # model may not exist or download may fail - handle gracefully
                self.pipe = pipeline("text-classification", model=self.model_id, return_all_scores=True, device=self.device)
                logger.info(f"Sarcasm HF pipeline loaded: {self.model_id}")
            except Exception as e:
                logger.warning(f"Could not load sarcasm HF model ({e}). Falling back to heuristic.")
                self.use_hf = False
                self.pipe = None

        # precompile heuristic patterns
        self._positive_words = re.compile(r"\b(great|fantastic|amazing|love|perfect|awesome|wonderful|best|brilliant)\b", flags=re.I)
        self._sarcasm_markers = re.compile(r"\b(yeah right|as if|sure|totally|i love when|oh great|good job)\b", flags=re.I)
        self._ellipses = re.compile(r"\.{2,}")
        self._exclaim_many = re.compile(r"(!{2,}|\?{2,})")
        # pattern: positive word + negative context words
        self._pos_neg_mix = re.compile(r"\b(great|love|amazing|fantastic)\b.*\b(waste|error|fail|broken|bug|problem|crash)\b", flags=re.I)

    def detect(self, text: str) -> Dict:
        """
        Returns:
            {
                "sarcasm": bool,
                "score": float in [0,1],
                "method": "hf" or "heuristic",
                "raw": raw_model_output_or_heuristic_info
            }
        """
        text = (text or "").strip()
        if not text:
            return {"sarcasm": False, "score": 0.0, "method": "heuristic", "raw": None}

        # HF path
        if self.use_hf and self.pipe is not None:
            try:
                res = self.pipe(text, truncation=True)
                # res is typically a list (one item) containing list of label/score dicts
                if isinstance(res, list) and res:
                    scored = res[0]  # list of dict {label, score}
                    # find label with highest score
                    best = max(scored, key=lambda x: x.get("score", 0.0))
                    label = str(best.get("label", "")).lower()
                    score = float(best.get("score", 0.0))
                    sarcasm = False
                    # heuristics: if label contains 'sarcasm' or 'sarcastic' treat accordingly
                    if "sarcasm" in label or "sarcastic" in label:
                        sarcasm = True if score >= 0.5 else False
                    elif "not_sarcasm" in label or "nosarcasm" in label:
                        sarcasm = False
                    else:
                        # some models have labels like LABEL_0 -> fallback mapping
                        sarcasm = score >= 0.6 and ("sar" in label or "yes" in label or "true" in label)
                    if self.debug:
                        logger.info(f"Sarcasm HF raw: {scored}")
                    return {"sarcasm": sarcasm, "score": score, "method": "hf", "raw": scored}
            except Exception as e:
                logger.warning(f"Sarcasm HF pipeline failed: {e}. Falling back to heuristic.")

        # Heuristic path (fast, no deps)
        score = 0.0
        reasons = []

        # marker-based boosts
        if self._ellipses.search(text):
            score += 0.15
            reasons.append("ellipses")
        if self._exclaim_many.search(text):
            score += 0.10
            reasons.append("many_punct")
        if self._sarcasm_markers.search(text):
            score += 0.35
            reasons.append("marker_phrase")
        if self._pos_neg_mix.search(text):
            score += 0.35
            reasons.append("pos_neg_mix")
        if self._positive_words.search(text) and "not" in text.lower():
            # positive word with explicit 'not' nearby
            score += 0.25
            reasons.append("pos_with_not")

        # normalize score into [0,1]
        score = max(0.0, min(1.0, score))

        sarcasm = score >= 0.5
        if self.debug:
            logger.info(f"Sarcasm heuristic score={score:.3f} reasons={reasons}")

        return {"sarcasm": sarcasm, "score": float(score), "method": "heuristic", "raw": {"reasons": reasons}}
