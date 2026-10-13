#!/bin/sh
# End-to-end smoke test against a running server. Runs inside the web container (it has curl, ffmpeg, python):
#   docker compose exec web sh scripts/smoke_test.sh
# Exits non-zero on the first failure.
set -eu
BASE="${BASE_URL:-http://localhost:8000}"
SAMPLE="${1:-tests/fixtures/sample_call.wav}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1], {}, {'d': d}))" "$1"; }

echo "1. health (loads the model on first call)"
curl -fsS "$BASE/healthz?load=1" | tee "$TMP/health.json"; echo
[ "$(json 'd["ok"]' < "$TMP/health.json")" = "True" ]

echo "2. upload a whole recording"
curl -fsS -F title="smoke upload" -F audio=@"$SAMPLE" "$BASE/api/calls/" > "$TMP/call.json"
json 'd["status"], d["duration"], d["transcript"]' < "$TMP/call.json"
[ "$(json 'd["status"]' < "$TMP/call.json")" = "done" ]
ID="$(json 'd["id"]' < "$TMP/call.json")"
curl -fsS "$BASE/api/calls/$ID/summary/" > "$TMP/summary.json"
json 'd["headline"], d["counts"]' < "$TMP/summary.json"
[ "$(json 'sum(d["counts"].values()) > 0' < "$TMP/summary.json")" = "True" ]

echo "3. live call: stream the same audio in 5 s chunks"
ffmpeg -loglevel error -i "$SAMPLE" -f segment -segment_time 5 -ar 16000 -ac 1 "$TMP/chunk%02d.wav"
LIVE="$(curl -fsS -H 'Content-Type: application/json' -d '{"title":"smoke live"}' "$BASE/api/calls/live/" | json 'd["id"]')"
LAST="$(ls "$TMP"/chunk*.wav | tail -n1)"
for f in "$TMP"/chunk*.wav; do
  FINAL=false; [ "$f" = "$LAST" ] && FINAL=true
  curl -fsS -F audio=@"$f" -F final=$FINAL "$BASE/api/calls/$LIVE/partial/" | json '"  +%d moment(s), waiting on: %r" % (len(d["new_moments"]), d["pending_text"])'
done
curl -fsS "$BASE/api/calls/$LIVE/summary/" > "$TMP/live.json"
json 'd["status"], d["counts"]' < "$TMP/live.json"
[ "$(json 'd["status"]' < "$TMP/live.json")" = "done" ]

echo "4. mono quote call (phone quality): speakers, quote lookup, dispute guidance, sentiment alert"
curl -fsS -F title="smoke quote" -F audio=@tests/fixtures/quote_call_phone_mono.wav "$BASE/api/calls/" | json 'd["id"]' > "$TMP/qid"
curl -fsS "$BASE/api/calls/$(cat "$TMP/qid")/summary/" > "$TMP/quote.json"
json 'd["speakers"], d["speaker_roles_confident"]' < "$TMP/quote.json"
json '[(q["customer"], q["icao"], q.get("offer_range_gppg")) for q in d["quotes"]]' < "$TMP/quote.json"
json '[w["prompt_agent"] for w in d["dispute_warnings"]]' < "$TMP/quote.json"
json '[(a["speaker"], a["avg_sentiment"]) for a in d["sentiment_alerts"]]' < "$TMP/quote.json"
[ "$(json 'bool(d["quotes"] and d["dispute_warnings"] and d["sentiment_alerts"])' < "$TMP/quote.json")" = "True" ]

echo "SMOKE TEST PASSED"
