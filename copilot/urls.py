from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("calls/<int:pk>/", views.call_detail, name="call_detail"),
    path("healthz", views.healthz, name="healthz"),
    path("api/calls/", views.api_calls, name="api_calls"),
    path("api/calls/live/", views.api_live_start, name="api_live_start"),
    path("api/calls/<int:pk>/", views.api_call, name="api_call"),
    path("api/calls/<int:pk>/partial/", views.api_partial, name="api_partial"),
    path("api/calls/<int:pk>/summary/", views.api_summary, name="api_summary"),
]
