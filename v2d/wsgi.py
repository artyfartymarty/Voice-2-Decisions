import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "v2d.settings")
application = get_wsgi_application()

from copilot import guards  # noqa: E402

guards.enforce()  # gunicorn skips Django's system checks, so refuse a fake or misspelled backend here too

from django.conf import settings  # noqa: E402

if settings.WARMUP_ON_START:
    from copilot import warmup  # noqa: E402

    warmup.start()  # load the models now so the first live call isn't the one that waits
