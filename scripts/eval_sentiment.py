"""Score labeled call lines with the lexicon and (if WINNOW_URL is set) Winnow; print accuracy and latency.

    python scripts/eval_sentiment.py [tests/fixtures/sentiment_lines.json]

Each line is [speaker, text, label] with label neg / neu / pos. A score <= -0.3 counts as neg, >= 0.3 as pos.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "v2d.settings")
import django  # noqa: E402

django.setup()
from django.conf import settings  # noqa: E402

from copilot.sentiment import LexiconSentiment, WinnowSentiment  # noqa: E402


def label(score):
    return "neg" if score <= -0.3 else "pos" if score >= 0.3 else "neu"


def evaluate(scorer, lines):
    right, neg_hit, neg_total, times, misses = 0, 0, 0, [], []
    for speaker, text, gold in lines:
        t0 = time.perf_counter()
        r = scorer.score(speaker, text, [])
        times.append(time.perf_counter() - t0)
        got = label(r["score"])
        right += got == gold
        if gold == "neg":
            neg_total += 1
            neg_hit += got == "neg"
        if got != gold:
            misses.append(f"{gold}->{got} ({r['score']:+.2f}) {text}")
    times.sort()
    return {"backend": r["backend"], "accuracy": round(right / len(lines), 3), "negative_recall": round(neg_hit / neg_total, 3),
            "p50_ms": round(1000 * times[len(times) // 2], 1), "p95_ms": round(1000 * times[int(len(times) * 0.95)], 1),
            "misses": misses}


if __name__ == "__main__":
    lines = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/sentiment_lines.json"))
    scorers = [LexiconSentiment()]
    if settings.WINNOW_URL:
        w = WinnowSentiment(settings.WINNOW_URL, settings.WINNOW_API_KEY, timeout=30)
        w.score("caller", "warm up", [])
        scorers.append(w)
    for s in scorers:
        print(json.dumps(evaluate(s, lines)))
