import json
from pathlib import Path

from django.core.management.base import BaseCommand

from copilot import pipeline
from copilot.evaluation import load_truth, role_accuracy
from copilot.models import Call


class Command(BaseCommand):
    help = ("Process mono recordings and score agent/caller attribution against hand-labeled turns. "
            "Each audio file needs a truth file next to it: call.wav + call.truth.json "
            '([{"start": 0.0, "end": 4.1, "role": "agent"}, ...]).')

    def add_arguments(self, parser):
        parser.add_argument("audio", nargs="+")

    def handle(self, *args, audio, **opts):
        results = []
        for path in map(Path, audio):
            truth = load_truth(path.with_suffix(".truth.json"))
            call = Call.objects.create(title=f"eval {path.name}")
            pipeline.process_recording(call, [path.read_bytes()])
            segs = [{"start": s.start, "end": s.end, "role": s.role} for s in call.segments.all()]
            score = role_accuracy(segs, truth)
            results.append(score["time_accuracy"])
            self.stdout.write(json.dumps({"file": path.name, "roles": call.roles,
                                          **{k: v for k, v in score.items() if k != "per_turn"}}))
        if results:
            self.stdout.write(f"mean time accuracy over {len(results)} call(s): {sum(results) / len(results):.3f}")
