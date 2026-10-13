"""Simulate a live call: stream a mono WAV to the live API in real time and report latency.

    python scripts/stream_wav.py tests/fixtures/quote_call_phone_mono.wav [--base http://localhost:8000]
        [--chunk 2.0] [--no-pace] [--customer-id 204511] [--json]

Each chunk is POSTed to /api/calls/<id>/partial/ as soon as that much audio has "happened" (paced at real time
unless --no-pace). For every chunk it prints the processing latency and any new moments; at the end it reports,
per use case, when it fired in call time and how long after the audio was sent. Standard library only.
"""
import argparse
import io
import json
import time
import urllib.request
import uuid
import wave


def post_json(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def post_chunk(url, wav_bytes, final):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"final\"\r\n\r\n{'true' if final else ''}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"chunk.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def chunks(path, seconds):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit("expects 16-bit PCM WAV")
        rate, channels = w.getframerate(), w.getnchannels()
        step = int(rate * seconds)
        pos = 0
        while True:
            frames = w.readframes(step)
            if not frames:
                break
            n = len(frames) // (2 * channels)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as out:
                out.setnchannels(channels)
                out.setsampwidth(2)
                out.setframerate(rate)
                out.writeframes(frames)
            yield pos / rate, (pos + n) / rate, buf.getvalue()
            pos += n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--chunk", type=float, default=2.0, help="seconds of audio per request")
    ap.add_argument("--no-pace", action="store_true", help="send as fast as possible instead of real time")
    ap.add_argument("--customer-id", default="", help="pre-fill the customer (as the phone system would)")
    ap.add_argument("--json", action="store_true", help="print only the final JSON report")
    a = ap.parse_args()

    call = post_json(f"{a.base}/api/calls/live/", {"title": f"stream {a.wav}", "customer_id": a.customer_id})
    url = f"{a.base}/api/calls/{call['id']}/partial/"
    all_chunks = list(chunks(a.wav, a.chunk))
    latencies, fired = [], {}
    t0 = time.perf_counter()
    for i, (start, end, data) in enumerate(all_chunks):
        if not a.no_pace:
            time.sleep(max(0.0, end - (time.perf_counter() - t0)))  # the chunk is complete at call time `end`
        sent = time.perf_counter()
        r = post_chunk(url, data, final=i == len(all_chunks) - 1)
        done = time.perf_counter()
        latencies.append(done - sent)
        for m in r["new_moments"]:
            fired.setdefault(m["kind"], {"audio_end_s": round(end, 2), "wall_s": round(done - t0, 2),
                                         "lag_after_chunk_s": round(done - sent, 3), "text": m["text"][:80]})
        if not a.json:
            kinds = ", ".join(m["kind"] for m in r["new_moments"]) or "-"
            print(f"{start:6.1f}-{end:5.1f}s  {1000 * (done - sent):6.0f} ms  {kinds}")
    lat = sorted(latencies)
    report = {"call_id": call["id"], "chunks": len(lat), "chunk_s": a.chunk, "paced": not a.no_pace,
              "latency_ms": {"p50": round(1000 * lat[len(lat) // 2]), "p95": round(1000 * lat[int(len(lat) * 0.95)]),
                             "max": round(1000 * lat[-1])},
              "first_fired": fired, "speakers": r.get("speakers")}
    print(json.dumps(report, indent=None if a.json else 2))


if __name__ == "__main__":
    main()
