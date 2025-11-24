"""
CLI demo for Chatbot with Sentiment Analysis.

Usage (from project root):
    # recommended (treats src as a package)
    python -m src.main

Options:
    --aggregation {mean,recency}    Aggregation method for overall sentiment (default: mean)
    --no-ensemble                   Disable HF+VADER ensemble and use single scorer
    --debug                         Enable debug logging in SentimentModule
"""
import argparse
import sys
import signal
from typing import Optional

from .chatbot import Chatbot
from .utils import pretty_print_report
from .sentiment import SentimentModule

def _make_sentiment_module(ensemble_enabled: bool, debug: bool) -> SentimentModule:
    """
    Construct a SentimentModule configured as requested.
    If ensemble_enabled is False, we'll still use SentimentModule but callers
    in Chatbot can set ensemble_by_default accordingly.
    """
    # by default try HF and fall back to VADER; SentimentModule handles fallback internally
    sm = SentimentModule(debug=debug, use_hf=True)
    return sm


def run_cli(aggregation: str = "mean", ensemble: bool = True, debug: bool = False):
    # Create sentiment module and inject into Chatbot so we can configure ensemble usage
    sent_mod = _make_sentiment_module(ensemble_enabled=ensemble, debug=debug)
    bot = Chatbot(sentiment_module=sent_mod, ensemble_by_default=ensemble)

    print("\nChatbot with Sentiment Analysis")
    print("Type 'exit' or 'quit' to end the chat. Press Ctrl+C to abort.\n")

    # Handle Ctrl+C gracefully
    def _sigint_handler(sig, frame):
        print("\n\nInterrupted. Computing report from conversation so far...\n")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while True:
            try:
                user = input("User: ").strip()
            except EOFError:
                # e.g., user pressed Ctrl+D — exit loop and show report
                print("\nEOF received. Exiting chat.")
                break

            if not user:
                continue

            bot.user_message(user)

            # exit tokens
            if user.lower() in ["exit", "quit", "bye"]:
                reply = bot.bot_reply("Goodbye!")
                print("Chatbot:", reply["text"])
                break

            # generate reply
            response = bot.simple_response(user)
            bot.bot_reply(response)
            print("Chatbot:", response)

    except KeyboardInterrupt:
        # fall through to produce report
        pass
    except Exception as e:
        print(f"\nUnhandled error: {e}", file=sys.stderr)
    finally:
        print("\n--- Sentiment Report ---\n")
        report = bot.analyze_user_messages(aggregation_method=aggregation, ensemble=ensemble)
        print(pretty_print_report(report))

def _parse_args(argv: Optional[list] = None):
    p = argparse.ArgumentParser(prog="chatbot-cli", description="Run the Chatbot CLI demo")
    p.add_argument("--aggregation", choices=["mean", "recency"], default="mean", help="Aggregation method for overall sentiment")
    p.add_argument("--no-ensemble", dest="no_ensemble", action="store_true", help="Disable HF+VADER ensemble (use single scorer)")
    p.add_argument("--debug", action="store_true", help="Enable debug logging in sentiment module")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    ensemble_flag = not args.no_ensemble
    run_cli(aggregation=args.aggregation, ensemble=ensemble_flag, debug=args.debug)
