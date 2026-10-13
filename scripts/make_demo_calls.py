"""Write the demo test calls: tests/fixtures/demo_*.wav plus a .script.json (what was said, when, by whom) and a
.truth.json (agent / caller turns) next to each.

With ASR_BACKEND=demo the app transcribes these calls from their scripts, so a demo on a machine without a GPU
shows each chunk's real words, timed to the audio.

    pip install kokoro-onnx
    # model files from https://github.com/thewh1teagle/kokoro-onnx/releases (model-files-v1.0)
    python scripts/make_demo_calls.py --model kokoro-v1.0.int8.onnx --voices voices-v1.0.bin

Each line is (speaker, what the TTS says, what the ASR would write if different). A line ending in "--" is cut
off mid-word, like a caller dropping out.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
AGENT_VOICE = "am_michael"

SCENARIOS = {
    "demo_quote_call": {
        "caller_voice": "af_sarah",
        "lines": [
            ("agent", "Thanks for calling G F L fuel desk, this is Mike. How can I help?", "Thanks for calling GFL fuel desk, this is Mike. How can I help?"),
            ("caller", "Hi Mike, this is Dana with Northwind Aviation, account two zero four five one one.", "Hi Mike, this is Dana with Northwind Aviation, account 204511."),
            ("caller", "We're flying into Van Nuys on Friday morning, and I'd like a fuel quote. What's your price per gallon?"),
            ("agent", "Sure, let me pull that up for you. Which aircraft is it?"),
            ("caller", "Tail number November eight eight November Whiskey."),
            ("agent", "Got it. The ramp fee at Sunset Aviation is waived over a hundred and fifty gallons. I'll send you the quote by end of day."),
            ("caller", "Perfect, thanks Mike."),
            ("agent", "Is there anything else I can help with? Have a good flight."),
        ],
    },
    "demo_dispute_location": {
        "caller_voice": "bf_emma",
        "lines": [
            ("agent", "Thanks for calling G F L, this is Mike. How can I help?", "Thanks for calling GFL, this is Mike. How can I help?"),
            ("caller", "Hi, it's Priya from Blue Heron Jet Charter, account three one eight eight four two.", "Hi, it's Priya from Blue Heron Jet Charter, account 318842."),
            ("caller", "We're landing at Paris Le Bourget on Saturday evening, around nine P M local, and we need fuel on arrival."),
            ("agent", "Okay, let me check Le Bourget for you."),
            ("caller", "Last time we were charged an overtime fee we weren't expecting. Can you tell me what the fees will be this time?"),
            ("caller", "And what's your price per gallon there?"),
            ("agent", "I understand. The F B O is open until ten, so a nine P M arrival should be inside hours. I'll put the overtime and callout fees in writing on the quote.",
             "I understand. The FBO is open until ten, so a 9 PM arrival should be inside hours. I'll put the overtime and callout fees in writing on the quote."),
            ("caller", "Great, that works. Thanks."),
        ],
    },
    "demo_frustrated_caller": {
        "caller_voice": "af_nicole",
        "lines": [
            ("agent", "Thanks for calling G F L fuel desk, this is Mike. How can I help?", "Thanks for calling GFL fuel desk, this is Mike. How can I help?"),
            ("caller", "Yeah, this is Dana at Northwind Aviation, account two zero four five one one. I'm calling about our Teterboro invoice.",
             "Yeah, this is Dana at Northwind Aviation, account 204511. I'm calling about our Teterboro invoice."),
            ("caller", "Honestly, I'm really frustrated. You charged us a callout fee again, and that was never agreed."),
            ("caller", "This is the third time. It's unacceptable, and I'm about ready to take our business elsewhere."),
            ("agent", "I'm sorry about that, I understand. Let me pull up the invoice. What's your tail number for that trip?"),
            ("caller", "Tail number is bravo tango--", "Tail number is bravo tango."),
            ("agent", "Sorry, you cut out there. What was the tail number again?"),
            ("caller", "November five one two bravo tango."),
            ("agent", "Thank you. I'll raise a dispute on the callout fee and get back to you by end of day."),
            ("caller", "Fine. I also need a fuel quote at Teterboro for next week, and I want every fee spelled out."),
            ("agent", "Absolutely. I'll send the quote with the callout and ramp fees listed."),
        ],
    },
}


def synth(tts, text: str, voice: str) -> np.ndarray:
    from scipy.signal import resample_poly

    cut = text.endswith("--")
    samples, sr = tts.create(text.rstrip("-") if not cut else text[:-2], voice=voice, speed=1.05, lang="en-us")
    samples = resample_poly(samples, SR, sr).astype(np.float32) if sr != SR else samples.astype(np.float32)
    nz = np.flatnonzero(np.abs(samples) > 0.01)  # trim the TTS's own leading/trailing silence
    if len(nz):
        samples = samples[max(0, nz[0] - 800): nz[-1] + 1600]
    if cut:  # the line drops: lose the last 15% and end on a short burst of static
        samples = samples[: int(len(samples) * 0.85)]
        samples = np.concatenate([samples, 0.02 * np.random.default_rng(0).standard_normal(int(0.25 * SR)).astype(np.float32)])
    return samples


def build(tts, name: str, spec: dict):
    rng = np.random.default_rng(len(name))
    parts, script, t = [np.zeros(int(0.5 * SR), dtype=np.float32)], [], 0.5
    for line in spec["lines"]:
        speaker, say = line[0], line[1]
        heard = line[2] if len(line) > 2 else say
        voice = AGENT_VOICE if speaker == "agent" else spec["caller_voice"]
        clip = synth(tts, say, voice)
        script.append({"start": round(t, 3), "end": round(t + len(clip) / SR, 3), "speaker": speaker, "text": heard})
        gap = float(rng.uniform(0.45, 0.8))
        parts += [clip, np.zeros(int(gap * SR), dtype=np.float32)]
        t += len(clip) / SR + gap
    samples = np.concatenate(parts)
    samples += 0.0015 * rng.standard_normal(len(samples)).astype(np.float32)  # phone-line hiss, so silence isn't digital zero
    samples = (0.8 * samples / max(1e-6, float(np.abs(samples).max()))).astype(np.float32)
    sf.write(OUT / f"{name}.wav", samples, SR, subtype="PCM_16")
    (OUT / f"{name}.script.json").write_text(json.dumps(script, indent=1) + "\n")
    truth = [{"start": s["start"], "end": s["end"], "role": s["speaker"]} for s in script]
    (OUT / f"{name}.truth.json").write_text(json.dumps(truth, indent=1) + "\n")
    print(f"{name}: {len(samples) / SR:.1f} s, {len(script)} lines")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="kokoro-v1.0(.int8).onnx")
    ap.add_argument("--voices", required=True, help="voices-v1.0.bin")
    ap.add_argument("--only", nargs="*", help="scenario names (default: all)")
    args = ap.parse_args()
    from kokoro_onnx import Kokoro

    tts = Kokoro(args.model, args.voices)
    for name, spec in SCENARIOS.items():
        if not args.only or name in args.only:
            build(tts, name, spec)


if __name__ == "__main__":
    main()
