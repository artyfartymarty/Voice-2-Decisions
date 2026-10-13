"""Measure how often speech is attributed to the right role (agent / caller).

Truth is a JSON list of {"start", "end", "role"} turns, e.g. hand-labeled from a real call.
Score = seconds of transcribed speech whose role matches the truth / seconds of transcribed speech that
overlaps any truth turn. It covers both diarization (who spoke when) and role naming (which one is the agent).
"""
import json
from pathlib import Path


def role_accuracy(segments: list[dict], truth: list[dict]) -> dict:
    """segments: [{"start", "end", "role"}] from a processed call."""
    right = total = 0.0
    per_turn = []
    for t in truth:
        t_right = t_total = 0.0
        for s in segments:
            overlap = min(s["end"], t["end"]) - max(s["start"], t["start"])
            if overlap > 0:
                t_total += overlap
                t_right += overlap if s["role"] == t["role"] else 0.0
        per_turn.append({**t, "covered_s": round(t_total, 2), "correct_s": round(t_right, 2)})
        right, total = right + t_right, total + t_total
    turns_ok = sum(1 for t in per_turn if t["covered_s"] and t["correct_s"] / t["covered_s"] >= 0.5)
    return {
        "time_accuracy": round(right / total, 4) if total else 0.0,
        "turns_correct": turns_ok, "turns": len(truth),
        "speech_covered": round(total / sum(t["end"] - t["start"] for t in truth), 4) if truth else 0.0,
        "per_turn": per_turn,
    }


def load_truth(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
