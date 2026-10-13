from django.apps import AppConfig


class CopilotConfig(AppConfig):
    name = "copilot"
    verbose_name = "Voice-2-Decisions copilot"

    def ready(self):
        from . import guards  # noqa: F401  registers the v2d.E001 system check
