from django.db import models


class Call(models.Model):
    STATUS = [(s, s) for s in ("live", "processing", "done", "error")]

    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS, default="processing")
    error = models.TextField(blank=True)
    language = models.CharField(max_length=16, blank=True)
    duration = models.FloatField(default=0.0)
    # Live calls: unfinished sentence held by winnow until the next partial arrives.
    pending_text = models.TextField(blank=True)
    # Call-level state winnow carries between partials: customer, location, quote intent, recent sentiment.
    context = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title or f"Call {self.pk}"

    @property
    def roles(self) -> dict:
        """Diarized speaker -> role ("speaker 1" -> "agent")."""
        return (self.context or {}).get("roles", {}).get("roles", {})

    @property
    def transcript(self) -> str:
        return " ".join(s.text for s in self.segments.all()).strip()


class Segment(models.Model):
    call = models.ForeignKey(Call, related_name="segments", on_delete=models.CASCADE)
    start = models.FloatField()
    end = models.FloatField()
    text = models.TextField()
    speaker = models.CharField(max_length=50, blank=True)

    class Meta:
        ordering = ["start", "id"]

    @property
    def role(self) -> str:
        return self.call.roles.get(self.speaker, self.speaker)


class Moment(models.Model):
    call = models.ForeignKey(Call, related_name="moments", on_delete=models.CASCADE)
    kind = models.CharField(max_length=20)
    text = models.TextField()
    score = models.FloatField()
    start = models.FloatField(null=True, blank=True)
    end = models.FloatField(null=True, blank=True)
    owner = models.CharField(max_length=100, blank=True)
    due = models.CharField(max_length=100, blank=True)
    also = models.JSONField(default=list, blank=True)
    speaker = models.CharField(max_length=50, blank=True)
    tool_name = models.CharField(max_length=100)
    tool_args = models.JSONField(default=dict)
    tool_status = models.CharField(max_length=20)  # done | pending | error
    tool_result = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["id"]

    @property
    def role(self) -> str:
        return self.call.roles.get(self.speaker, self.speaker)

    def as_dict(self):
        return {
            "id": self.pk, "kind": self.kind, "text": self.text, "score": self.score,
            "start": self.start, "end": self.end, "owner": self.owner, "due": self.due, "also": self.also,
            "speaker": self.speaker, "role": self.call.roles.get(self.speaker, self.speaker),
            "tool_call": {"name": self.tool_name, "args": self.tool_args},
            "tool_status": self.tool_status, "tool_result": self.tool_result,
        }
