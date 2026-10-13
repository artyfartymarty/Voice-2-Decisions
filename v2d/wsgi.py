import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "v2d.settings")
application = get_wsgi_application()

from copilot import guards  # noqa: E402

guards.enforce()  # gunicorn skips Django's system checks, so refuse a fake or misspelled backend here too
