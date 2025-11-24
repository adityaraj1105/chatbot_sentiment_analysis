#!/usr/bin/env python3
"""
run_tests.py

Usage:
    # from project root
    python run_tests.py

What it does:
- Loads all JSON transcripts from demo_transcripts/
- For each transcript, feeds user messages to Chatbot (records bot replies)
- Runs sentiment analysis and writes a combined JSON report to test_results.json
- Prints a summary to console
"""

import json
import glob
import os
from pathlib import Path
from typing import Any, Dict

from src.chatbot import Chatbot

TRANSCRIPTS_DIR = Path("demo_transcripts")
OUT_FILE = Path("test_results.json")


def load_transcript(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_single_transcript(data: Dict[str, Any], ensemble: bool = True, aggregation: str = "mean") -> Dict[str, Any]:
    """
    Runs a single transcript through Chatbot and returns the report dict:
    {
      "name": "...",
      "messages": [...],
      "per_message": [...],
      "aggregate": {...},
      "trend": "...",
      "raw_report": {...}
    }
    """
    bot = Chatbot()  # uses default SentimentModule; Chatbot uses ensemble_by_default by default
    msgs = data.get("messages", [])
    # Feed messages to bot
    for m in msgs:
        bot.user_message(m)
        reply = bot.simple_response(m)
        bot.bot_reply(reply)

    # analyze; pass aggregation + ensemble flag
    report = bot.analyze_user_messages(aggregation_method=aggregation)
    # Include original transcript and metadata
    out = {
        "name": data.get("name", path.stem if 'path' in locals() else "unknown"),
        "description": data.get("description"),
        "meta": data.get("meta", {}),
        "messages": msgs,
        "analysis": report,
    }
    return out


def main():
    if not TRANSCRIPTS_DIR.exists():
        print(f"Transcript directory not found: {TRANSCRIPTS_DIR.resolve()}")
        return

    results = {}
    files = sorted(glob.glob(str(TRANSCRIPTS_DIR / "*.json")))
    if not files:
        print(f"No transcripts found in {TRANSCRIPTS_DIR}")
        return

    for p in files:
        print(f"Running transcript: {p}")
        data = load_transcript(Path(p))
        res = run_single_transcript(data, ensemble=True, aggregation="recency")
        results[data.get("name", Path(p).stem)] = res

    # Save results
    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print brief summary
    print(f"\nSaved results to {OUT_FILE.resolve()}\n")
    for name, r in results.items():
        agg = r["analysis"]["aggregate"]
        trend = r["analysis"]["trend"]
        print(f"- {name}: aggregate_label={agg['label']} compound={agg['compound']:.3f} trend={trend}")

    print("\nDone.")


if __name__ == "__main__":
    main()
