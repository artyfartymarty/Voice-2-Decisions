"""Act as a call handler: play each demo call from the entry page in a headless browser and record what shows up.

    ASR_BACKEND=demo python manage.py runserver   # in one shell
    python scripts/demo_walkthrough.py --out demo-run   # in another

Writes a video and screenshots per call, plus timeline.json: when each scripted line showed up in the transcript
(audio seconds after the line ended) and when each prompt card appeared.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def norm(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower().replace("\n", " "))


def tail_words(text, n=3):
    return " ".join(norm(text).split()[-n:])


def run_call(browser, base, name, out: Path, shots: dict):
    script = json.loads((FIXTURES / f"{name}.script.json").read_text())
    ctx = browser.new_context(viewport={"width": 1366, "height": 900}, record_video_dir=str(out / "video"),
                              record_video_size={"width": 1366, "height": 900})
    page = ctx.new_page()
    errors = []
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.on("requestfailed", lambda r: errors.append(f"failed {r.url}"))
    page.on("response", lambda r: r.status >= 400 and errors.append(f"{r.status} {r.url}"))
    page.goto(base + "/")
    page.click(f"button.case[data-name='{name}.wav']")
    page.wait_for_function("document.getElementById('sim-status').textContent.includes('Press Play')", timeout=20000)
    page.screenshot(path=str(out / f"{name}-0-loaded.png"))
    page.click("#sim-play")
    t0 = time.time()
    seen_lines, cards, slot_states, samples = {}, {}, [], []
    shot_at = dict(shots.get(name, {}))
    while True:
        state = page.evaluate("""() => ({
            status: document.getElementById('sim-status').textContent,
            transcript: (document.getElementById('panel-transcript')||{}).innerText || '',
            prompts: [...document.querySelectorAll('#panel-prompts .pcard')].map(e => e.closest('.group').querySelector('h3').firstChild.textContent.trim() + ': ' + e.querySelector('.head').innerText),
            prompts_text: (document.getElementById('panel-prompts')||{}).innerText || '',
            slot: (document.getElementById('panel-slot')||{}).innerText || '',
            who: (document.getElementById('panel-who')||{}).innerText || '',
        })""")
        m = re.match(r"([\d.]+) / ([\d.]+) s", state["status"])
        at = float(m.group(1)) if m else None
        wall = time.time() - t0
        tnorm = norm(state["transcript"])
        for i, line in enumerate(script):
            if i not in seen_lines and tail_words(line["text"]) in tnorm:
                seen_lines[i] = {"audio_t": at, "wall": round(wall, 2)}
        for p in state["prompts"]:
            cards.setdefault(p, {"audio_t": at, "wall": round(wall, 2)})
        if state["slot"].strip() and (not slot_states or slot_states[-1]["text"] != state["slot"].strip()):
            slot_states.append({"audio_t": at, "wall": round(wall, 2), "text": state["slot"].strip()})
        samples.append({"wall": round(wall, 2), "audio_t": at})
        for label, t in list(shot_at.items()):
            if at is not None and at >= t:
                page.screenshot(path=str(out / f"{name}-{label}.png"))
                del shot_at[label]
        if "Call ended" in state["status"] or wall > 300:
            break
        time.sleep(0.25)
    page.wait_for_timeout(2500)  # the last poll of the panels
    final = page.evaluate("(document.getElementById('panel-transcript')||{}).innerText || ''")
    for i, line in enumerate(script):  # lines that only landed with the final chunk
        if i not in seen_lines and tail_words(line["text"]) in norm(final):
            seen_lines[i] = {"audio_t": None, "wall": round(time.time() - t0, 2), "after_end": True}
    page.screenshot(path=str(out / f"{name}-9-ended.png"), full_page=True)
    call_href = page.get_attribute("#sim-link", "href")
    page.goto(base + call_href)
    page.wait_for_timeout(800)
    page.screenshot(path=str(out / f"{name}-call-page.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_timeout(300)
    page.screenshot(path=str(out / f"{name}-call-page-phone.png"), full_page=True)
    ctx.close()
    video = sorted((out / "video").glob("*.webm"), key=lambda p: p.stat().st_mtime)[-1]
    video.rename(out / f"{name}.webm")
    lines = []
    for i, line in enumerate(script):
        s = seen_lines.get(i)
        lag = round(s["audio_t"] - line["end"], 2) if s and s.get("audio_t") is not None else None
        lines.append({"speaker": line["speaker"], "text": line["text"], "ended_at": line["end"], "shown": s, "lag_s": lag})
    return {"call": call_href, "lines": lines, "cards": cards, "slot": slot_states, "errors": errors,
            "final_transcript": final}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="demo-run")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.only or sorted(p.name.removesuffix(".script.json") for p in FIXTURES.glob("demo_*.script.json"))
    shots = {"demo_frustrated_caller": {"1-angry": 36, "2-cutoff": 44, "3-reask": 52}, "demo_quote_call": {"1-quote": 20},
             "demo_dispute_location": {"1-dispute": 28}}
    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None,
                                    args=["--autoplay-policy=no-user-gesture-required"])
        for name in names:
            print("playing", name, flush=True)
            results[name] = run_call(browser, args.base, name, out, shots)
        page = browser.new_page(viewport={"width": 1366, "height": 900})
        page.goto(args.base + "/alerts/")
        page.screenshot(path=str(out / "alerts.png"), full_page=True)
        browser.close()
    (out / "timeline.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({n: [(l["lag_s"], l["text"][:40]) for l in r["lines"]] for n, r in results.items()}, indent=1))


if __name__ == "__main__":
    main()
