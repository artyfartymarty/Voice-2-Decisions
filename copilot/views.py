import json

from django.http import Http404, HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from . import extract, pipeline, tools
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
            return render(request, "copilot/index.html", {"calls": Call.objects.all()[:50], "error": "Choose an audio file."}, status=400)
        call = Call.objects.create(title=request.POST.get("title", "")[:200], context=_prefill(request.POST))
        pipeline.process_recording(call, legs, _speakers(request))
        return redirect("call_detail", pk=call.pk)
    return render(request, "copilot/index.html", {"calls": Call.objects.all()[:50]})


@require_GET
def call_detail(request, pk):
    call = get_object_or_404(Call, pk=pk)
    summary = pipeline.call_summary(call)
    counts = [(kind, kind.replace("_", " "), n) for kind, n in summary["counts"].items()]
    return render(request, "copilot/call_detail.html", {"call": call, "summary": summary, "counts": counts})


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
