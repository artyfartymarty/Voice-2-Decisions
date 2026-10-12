import json
from pathlib import Path

from django.core.management.base import BaseCommand

from copilot import pipeline
from copilot.models import Call


class Command(BaseCommand):
    help = "Transcribe one recording (or two legs) and print the call summary as JSON."

    def add_arguments(self, parser):
        parser.add_argument("audio", nargs="+", help="one file, or two files for the two legs of a call")
        parser.add_argument("--title", default="")
        parser.add_argument("--full", action="store_true", help="print transcript and every moment, not just the summary")

    def handle(self, *args, audio, title, full, **opts):
        call = Call.objects.create(title=title or Path(audio[0]).name)
        pipeline.process_recording(call, [Path(p).read_bytes() for p in audio])
        if call.status == "error":
            self.stderr.write(call.error)
            raise SystemExit(1)
        out = pipeline.call_payload(call) if full else pipeline.call_summary(call)
        self.stdout.write(json.dumps(out, indent=2))
