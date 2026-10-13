from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("calls/<int:pk>/", views.call_detail, name="call_detail"),
    path("calls/<int:pk>/panels/", views.call_panels, name="call_panels"),
    path("alerts/", views.manager_alerts, name="manager_alerts"),
    path("test-calls/<str:name>", views.test_call_file, name="test_call_file"),
    path("healthz", views.healthz, name="healthz"),
    path("api/calls/", views.api_calls, name="api_calls"),
    path("api/boost-words/", views.api_boost_words, name="api_boost_words"),
    path("api/calls/live/", views.api_live_start, name="api_live_start"),
    path("api/calls/<int:pk>/", views.api_call, name="api_call"),
    path("api/calls/<int:pk>/partial/", views.api_partial, name="api_partial"),
    path("api/calls/<int:pk>/summary/", views.api_summary, name="api_summary"),
]
