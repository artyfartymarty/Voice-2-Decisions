import json

from django.http import FileResponse, Http404, HttpResponse, HttpResponseBadRequest, JsonResponse
from django.conf import settings
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from . import callview, extract, pipeline, tools
from .asr import get_backend
from .diarize import get_diarizer
from .audio import AudioDecodeError
from .models import Call


def _prefill(data) -> dict:
    """Customer known before anyone speaks: from the telephony system's customer_id (or ANI lookup upstream).

    Pre-filled customers make quote_lookup fire as soon as a location and a price question come up, and let the
    tail-number slot check the customer's fleet.
    """
    number = str(data.get("customer_id", "") or "").strip()
    ctx = {"ani": str(data.get("ani", "") or "")[:32]} if data.get("ani") else {}
    name = extract.get().customers.get(number, (None,))[0] if number else None
    if name:
        ctx.update(customer=number, customer_name=name, customer_source="prefill")
    return ctx


def _speakers(request) -> list[str]:
    """Optional labels for the two legs (speaker_a, speaker_b); blanks fall back to agent / caller."""
    return [request.POST.get("speaker_a", "").strip()[:50], request.POST.get("speaker_b", "").strip()[:50]]


# --- pages ---

@require_http_methods(["GET", "POST"])
def index(request):
    if request.method == "POST":
        legs = [f.read() for f in (request.FILES.get("audio"), request.FILES.get("audio_b")) if f]
        if not legs:
            return render(request, "copilot/index.html", _index_context(error="Choose an audio file."), status=400)
        call = Call.objects.create(title=request.POST.get("title", "")[:200], context=_prefill(request.POST))
        pipeline.process_recording(call, legs, _speakers(request))
        return redirect("call_detail", pk=call.pk)
    return render(request, "copilot/index.html", _index_context())


def _index_context(**extra) -> dict:
    return {"calls": Call.objects.all()[:50], "test_calls": callview.test_calls(), "refresh_ms": settings.LIVE_REFRESH_MS,
            "gateway_ws_url": settings.GATEWAY_WS_URL, "fake_asr": _fake_asr(), **extra}


def _fake_asr() -> bool:
    """ASR_BACKEND=fake returns the same canned text for every chunk; the pages warn so it isn't mistaken for a bug."""
    return get_backend().name == "fake"


@require_GET
def test_call_file(request, name):
    """One of the pregenerated test calls, by file name (only names listed on the entry page are served)."""
    match = next((c for c in callview.test_calls() if c["name"] == name), None)
    if match is None:
        raise Http404("no such test call")
    return FileResponse(open(match["path"], "rb"), content_type="audio/wav")


def _call_page_context(call) -> dict:
    summary = pipeline.call_summary(call)
    return {
        "call": call, "summary": summary,
        "counts": [(kind, kind.replace("_", " "), n) for kind, n in summary["counts"].items()],
        "who": callview.who_is_calling(call), "slot": callview.slot_hint(call),
        "groups": callview.prompt_cards(call), "turns": callview.transcript(call),
        "version": callview.version(call),
    }


# Panels of the call page that the live script swaps in place; each is its own template under copilot/panels/.
PANELS = ("who", "slot", "prompts", "transcript", "summary")


@require_GET
def call_detail(request, pk):
    call = get_object_or_404(Call, pk=pk)
    ctx = _call_page_context(call)
    ctx.update(gateway_ws_url=settings.GATEWAY_WS_URL, refresh_ms=settings.LIVE_REFRESH_MS, fake_asr=_fake_asr())
    return render(request, "copilot/call_detail.html", ctx)


@require_GET
def call_panels(request, pk):
    """The call page's panels as server-rendered HTML, for the live script. ?v=<version> -> 204 when nothing changed."""
    call = get_object_or_404(Call, pk=pk)
    version = callview.version(call)
    if request.GET.get("v") == version:
        return HttpResponse(status=204)
    ctx = _call_page_context(call)
    return JsonResponse({"version": version, "status": call.status,
                         "panels": {name: render_to_string(f"copilot/panels/{name}.html", ctx, request) for name in PANELS}})


@require_GET
def manager_alerts(request):
    """Managers' and dispatchers' view: sentiment and quote alerts across calls, newest first. ?live=1 for live calls only."""
    live_only = request.GET.get("live") == "1"
    ctx = {"alerts": callview.manager_alerts(live_only=live_only), "live_only": live_only,
           "gateway_ws_url": settings.GATEWAY_WS_URL, "refresh_ms": settings.LIVE_REFRESH_MS}
    if request.GET.get("partial") == "1":
        return render(request, "copilot/panels/alerts.html", ctx)
    return render(request, "copilot/alerts.html", ctx)


# --- JSON API (csrf-exempt: called by scripts, the live recorder and other services) ---

@require_GET
def healthz(request):
    info = get_backend().info()
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["cuda"] = False
    diarizer = get_diarizer()
    if request.GET.get("load") == "1":
        if hasattr(get_backend(), "load"):
            get_backend().load()
            info.update(get_backend().info())
        if diarizer:
            diarizer.load()
    info["tools"] = tools.available()
    return JsonResponse({"ok": True, "asr": info, "diarizer": diarizer.info() if diarizer else {"backend": "off"}})


@csrf_exempt
@require_http_methods(["GET", "POST"])
def api_calls(request):
    """GET: list calls. POST multipart: audio (required), audio_b (optional second leg), title -> processed call."""
    if request.method == "GET":
        return JsonResponse({"calls": [{"id": c.pk, "title": str(c), "status": c.status} for c in Call.objects.all()[:100]]})
    legs = [f.read() for f in (request.FILES.get("audio"), request.FILES.get("audio_b")) if f]
    if not legs:
        return JsonResponse({"error": "send an 'audio' file (and optionally 'audio_b' for the second leg)"}, status=400)
    call = Call.objects.create(title=request.POST.get("title", "")[:200], context=_prefill(request.POST))
    pipeline.process_recording(call, legs, _speakers(request))
    return JsonResponse(pipeline.call_payload(call), status=201 if call.status == "done" else 422)


@csrf_exempt
@require_POST
def api_live_start(request):
    body = json.loads(request.body or b"{}") if request.content_type == "application/json" else request.POST
    call = Call.objects.create(title=str(body.get("title", ""))[:200], status="live", context=_prefill(body))
    pipeline.reset_live_state(call.pk)
    return JsonResponse(pipeline.call_payload(call), status=201)


@csrf_exempt
@require_POST
def api_partial(request, pk):
    """Feed a live call. JSON {"text", "speaker", "final"} or multipart with an 'audio' chunk (+ optional 'final').

    Audio chunks are diarized server-side. 'speaker' is only for text partials from a source that already knows who spoke.
    """
    call = get_object_or_404(Call, pk=pk)
    if call.status not in ("live", "done"):
        return HttpResponseBadRequest("call is not live")
    if request.content_type == "application/json":
        body = json.loads(request.body or b"{}")
        text, audio_bytes, final = str(body.get("text", "")), None, bool(body.get("final"))
        speaker = str(body.get("speaker", ""))[:50]
    else:
        f = request.FILES.get("audio")
        text, audio_bytes = request.POST.get("text", ""), (f.read() if f else None)
        final = request.POST.get("final", "").lower() in {"1", "true", "yes"}
        speaker = request.POST.get("speaker", "")[:50]
    try:
        new = pipeline.process_partial(call, text=text, audio_bytes=audio_bytes, final=final, speaker=speaker)
    except AudioDecodeError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    return JsonResponse({"new_moments": [m.as_dict() for m in new], "pending_text": call.pending_text,
                         "status": call.status, "speakers": call.roles})


@require_GET
def api_boost_words(request):
    """Phrases to bias the ASR toward (customer names, FBOs, airports, codes), built from the reference data."""
    return JsonResponse({"phrases": extract.get().boost_words()})


@require_GET
def api_call(request, pk):
    return JsonResponse(pipeline.call_payload(get_object_or_404(Call, pk=pk)))


@require_GET
def api_summary(request, pk):
    return JsonResponse(pipeline.call_summary(get_object_or_404(Call, pk=pk)))
