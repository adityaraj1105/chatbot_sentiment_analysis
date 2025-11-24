"""
Utility helpers for the Chatbot project.
Includes:
- pretty_print_report: formats sentiment analysis output for CLI
- preprocess_user_text: normalizes user input for stable sentiment scoring
"""

import json
import re
from typing import Dict, Any
from pathlib import Path


# ---------------------------------------------
# Pretty-print conversation sentiment report
# ---------------------------------------------
def pretty_print_report(report: Dict[str, Any], show_emotion: bool = True) -> str:
    """
    Convert the analysis report (as returned by Chatbot.analyze_user_messages)
    into a clean, human-readable multi-line string.

    Args:
        report: dict containing per_message, aggregate, trend
        show_emotion: include emotion labels if present
    """

    lines = []
    per = report.get("per_message", [])

    lines.append("Per-message sentiment:")
    for i, entry in enumerate(per, start=1):
        text = entry.get("text", "")
        label = entry.get("label", "Unknown")
        comp = entry.get("scores", {}).get("compound", 0.0)
        skipped = entry.get("skipped", False)
        emo_label = entry.get("emotion") if show_emotion else None

        segment = f'{i}. "{text}" → {label} (compound={comp:.3f})'
        if emo_label:
            segment += f" • Emotion: {emo_label}"
        if skipped:
            segment += " [skipped]"
        lines.append(segment)

    # Aggregate summary
    agg = report.get("aggregate", {})
    agg_label = agg.get("label", "Neutral")
    agg_comp = agg.get("compound", 0.0)
    trend = report.get("trend", "Stable")

    lines.append("")
    lines.append(f"Overall conversation sentiment: {agg_label} (score={agg_comp:.3f})")
    lines.append(f"Trend across conversation: {trend}")

    return "\n".join(lines)


# ---------------------------------------------
# Input text normalization
# ---------------------------------------------
CONTRACTION_MAP = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
}


def expand_contractions(text: str) -> str:
    """Expand common English contractions."""
    pattern = re.compile("|".join(CONTRACTION_MAP.keys()))
    def replace(match):
        return CONTRACTION_MAP.get(match.group(0), match.group(0))
    return pattern.sub(replace, text)


def preprocess_user_text(text: str) -> str:
    """
    Normalize text before sentiment scoring.

    Operations:
    - Trim whitespace
    - Remove surrounding quotes
    - Normalize apostrophes
    - Expand contractions (don’t → do not)
    - Remove repeated spaces
    - Remove excessive punctuation
    - Strip invisible Unicode characters
    """

    if not text:
        return text

    # Strip
    t = text.strip()

    # Remove surrounding quotes
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()

    # Normalize apostrophes
    t = t.replace("’", "'").replace("`", "'")

    # Expand contractions
    t = expand_contractions(t)

    # Collapse repeated punctuation like "!!", "??", "...", "----"
    t = re.sub(r"([!?.,])\1+", r"\1", t)

    # Collapse repeated spaces
    t = " ".join(t.split())

    # Remove weird invisible characters
    t = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", t)

    # Optionally: limit to readable text only (no emojis or symbols)
    # t = re.sub(r"[^\w\s.,!?']", "", t)

    return t


def save_report_as_json(report: Dict[str, Any], path: str) -> None:
    """
    Save a report (dict) to `path` as JSON (UTF-8).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
