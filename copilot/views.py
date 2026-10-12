import json

from django.http import Http404, HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from . import pipeline, tools
from .asr import get_backend
from .audio import AudioDecodeError
from .models import Call


# --- pages ---

@require_http_methods(["GET", "POST"])
def index(request):
    if request.method == "POST":
        legs = [f.read() for f in (request.FILES.get("audio"), request.FILES.get("audio_b")) if f]
        if not legs:
            return render(request, "copilot/index.html", {"calls": Call.objects.all()[:50], "error": "Choose an audio file."}, status=400)
        call = Call.objects.create(title=request.POST.get("title", "")[:200])
        pipeline.process_recording(call, legs)
        return redirect("call_detail", pk=call.pk)
    return render(request, "copilot/index.html", {"calls": Call.objects.all()[:50]})


@require_GET
def call_detail(request, pk):
    call = get_object_or_404(Call, pk=pk)
    return render(request, "copilot/call_detail.html", {"call": call, "summary": pipeline.call_summary(call)})


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
    if request.GET.get("load") == "1" and hasattr(get_backend(), "load"):
        get_backend().load()
        info.update(get_backend().info())
    info["tools"] = tools.available()
    return JsonResponse({"ok": True, "asr": info})


@csrf_exempt
@require_http_methods(["GET", "POST"])
def api_calls(request):
    """GET: list calls. POST multipart: audio (required), audio_b (optional second leg), title -> processed call."""
    if request.method == "GET":
        return JsonResponse({"calls": [{"id": c.pk, "title": str(c), "status": c.status} for c in Call.objects.all()[:100]]})
    legs = [f.read() for f in (request.FILES.get("audio"), request.FILES.get("audio_b")) if f]
    if not legs:
        return JsonResponse({"error": "send an 'audio' file (and optionally 'audio_b' for the second leg)"}, status=400)
    call = Call.objects.create(title=request.POST.get("title", "")[:200])
    pipeline.process_recording(call, legs)
    return JsonResponse(pipeline.call_payload(call), status=201 if call.status == "done" else 422)


@csrf_exempt
@require_POST
def api_live_start(request):
    body = json.loads(request.body or b"{}") if request.content_type == "application/json" else request.POST
    call = Call.objects.create(title=str(body.get("title", ""))[:200], status="live")
    return JsonResponse(pipeline.call_payload(call), status=201)


@csrf_exempt
@require_POST
def api_partial(request, pk):
    """Feed a live call. JSON {"text": "...", "final": false} or multipart with an 'audio' chunk (+ optional 'final')."""
    call = get_object_or_404(Call, pk=pk)
    if call.status not in ("live", "done"):
        return HttpResponseBadRequest("call is not live")
    if request.content_type == "application/json":
        body = json.loads(request.body or b"{}")
        text, audio_bytes, final = str(body.get("text", "")), None, bool(body.get("final"))
    else:
        f = request.FILES.get("audio")
        text, audio_bytes = request.POST.get("text", ""), (f.read() if f else None)
        final = request.POST.get("final", "").lower() in {"1", "true", "yes"}
    try:
        new = pipeline.process_partial(call, text=text, audio_bytes=audio_bytes, final=final)
    except AudioDecodeError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    return JsonResponse({"new_moments": [m.as_dict() for m in new], "pending_text": call.pending_text, "status": call.status})


@require_GET
def api_call(request, pk):
    return JsonResponse(pipeline.call_payload(get_object_or_404(Call, pk=pk)))


@require_GET
def api_summary(request, pk):
    return JsonResponse(pipeline.call_summary(get_object_or_404(Call, pk=pk)))
