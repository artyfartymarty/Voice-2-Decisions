# API contract

The service takes the transcript text Nova Sonic produces and returns the variables it has parsed, the functions that
fired, and the result the decision model picked for what the caller asked. It is plain JSON over HTTP on port 8000
(`PORT` to change). Nothing here touches audio or renders UI.

One call is one `call_id`, any string you choose (the Nova Sonic session id works). State is kept in memory per
`call_id` until you `DELETE` it or the service restarts.

| Method | Path | What it does |
|---|---|---|
| `POST` | `/calls/<call_id>/events` | Send transcript text. Returns what happened. |
| `GET` | `/calls/<call_id>` | The whole state of the call. |
| `DELETE` | `/calls/<call_id>` | Forget the call. |
| `GET` | `/functions` | The functions and the variables each one needs. |
| `GET` | `/healthz` | Liveness and which decision model is in use. |

CORS is on for every origin by default, so a browser UI can call the service directly. Set `CORS_ORIGIN` to one
origin (for example `https://ui.example.com`) to restrict it, or to an empty value to send no CORS headers.

## POST /calls/<call_id>/events

### Request

The body is one event or a JSON array of events, sent in the order they happened. Any of these shapes works, and they
can be mixed:

```jsonc
// 1. Nova Sonic raw output events, forwarded as they arrive. contentStart gives the role and stage; the
//    textOutput with the same contentId carries the text. Other events (audioOutput, contentEnd, ...) are ignored.
{"event": {"contentStart": {"contentId": "c1", "type": "TEXT", "role": "USER",
                            "additionalModelFields": "{\"generationStage\":\"FINAL\"}"}}}
{"event": {"textOutput": {"contentId": "c1", "role": "USER", "content": "What is your price per gallon?"}}}

// 2. A textOutput flattened by your code
{"role": "USER", "content": "What is your price per gallon?", "generationStage": "FINAL"}

// 3. Plain
{"speaker": "caller", "text": "What is your price per gallon?"}
```

| Field | Meaning |
|---|---|
| `role` | `USER` is the caller and `ASSISTANT` is the agent. Change the mapping with `ROLE_MAP` (default `USER:caller,ASSISTANT:agent`). |
| `generationStage` | `FINAL` text fills variables and can produce a decision. `SPECULATIVE` text is only stored as a preview. Missing means `FINAL`. |
| `speaker` | `caller` or `agent`, for the plain shape. |

Send agent lines too. They are not decided on, but they give context, for example the agent asking "what's the tail
number?" tells the service the caller's next line is likely a tail. The `{"interrupted": true}` content Nova Sonic sends on
barge-in is ignored.

### Response

```jsonc
{
  "call_id": "demo",
  "steps": [ ... ],            // one per event that carried text, in order (see below)
  "decision": { ... } | null,  // the call's current decision, possibly from an earlier request
  "waiting_for": "fuel_quote" | null,  // a function the caller asked for that still lacks variables
  "variables": {"customer_number": "204511", "icao": "KTEB"}  // every variable filled so far, value only
}
```

`steps[].decision` is set only when that line produced a new decision. The top-level `decision` is the latest one on
the call, so a UI that only wants "what should the agent say now" can read that field alone.

A **final** step:

| Field | Meaning |
|---|---|
| `final` | `true` |
| `line` | `{speaker, text, at}`, where `at` is seconds since the call's first event |
| `variables_changed` | Names of the variables this line filled or changed |
| `fired` | Functions called because their variables are now filled: `[{function, args, result, at}]` |
| `prompts` | Things the agent should say or check, such as "Read the tail number back to the caller: N88NW." |
| `decision` | A new decision (below), or `null` |

A **speculative** step is just `{"final": false, "preview": "caller: Northwind"}`.

### The decision

The decision model picks the function whose result answers what the caller asked for. Two forms:

**Ready**: every variable the function needs is filled, so its result is included.

```json
{"function": "fuel_quote", "ready": true,
 "say": "Quote NORTHWIND AVIATION, LLC at KTEB: current 0.3125 GPPG. Don't go below the minimum, 0.0850 GPPG (0.2275 of room).",
 "result": {"customer": "NORTHWIND AVIATION, LLC", "customer_number": "204511", "icao": "KTEB",
            "fbo": "Meadowlark Jet Center", "current_gppg": 0.3125, "floor_gppg": 0.085, "...": "..."},
 "because": "Account 204511 at Teterboro.",
 "by": "winnow", "p": 0.93, "probabilities": {"fuel_quote": 0.93, "dispute_check": 0.04, "...": 0.0}, "ms": 41}
```

**Not ready**: the caller asked for something, but variables are missing. `say` tells the agent what to ask for.
When the missing variables arrive, a ready decision for the same function comes back on that later line with
`"by": "waiting"`, without the caller asking again.

```json
{"function": "fuel_quote", "ready": false, "missing": ["customer_number", "icao"],
 "say": "Ask the caller for the account number or company name and which airport or FBO.",
 "because": "What is your price per gallon?", "by": "rules", "p": 1.0}
```

| Field | Meaning |
|---|---|
| `function` | One of the names from `GET /functions` |
| `ready` | Whether `result` is included |
| `say` | One line for the agent: the answer, or what to ask for |
| `result` | The function's full result (ready only). Every result also has its own `say`. |
| `missing` | Variables still needed (not ready only) |
| `because` | The caller line that led to this decision |
| `by` | `winnow`, `rules` (keyword fallback, no `WINNOW_URL`), `rules (winnow unavailable)`, or `waiting` |
| `p` | Probability of the picked function. Picks below `DECIDE_MIN_P` (default 0.5) produce no decision. |
| `probabilities`, `ms` | Winnow only: the probability of every option, and the request time in milliseconds |

## Variables and functions

| Variable | Example value | Filled from |
|---|---|---|
| `customer_number` | `204511` | An account number said as digits, or a company name (GPT 6 Luna reads it, then it must match customers.csv) |
| `icao` | `KTEB` | ICAO or IATA code, spelled or phonetic, or an FBO or airport name |
| `tail_number` | `N88NW` | GPT 6 Luna reads it when Winnow says a full tail was said; it must be a valid registration on the customer's fleet |

| Function | Needs | Called |
|---|---|---|
| `customer_profile` | customer_number | automatically |
| `location_info` | icao | automatically |
| `fuel_quote` | customer_number, icao | automatically |
| `dispute_check` | icao (+ customer_number) | automatically |
| `tail_check` | tail_number (+ customer_number) | automatically |
| `escalate` | nothing | only when the decision model picks it |

Functions are called as soon as their variables are filled and again when an input changes, so their results are
usually ready before the caller asks.

## GET /calls/<call_id>

The full state, for a UI that polls or reconnects.

```jsonc
{
  "call_id": "demo",
  "variables": {
    "customer_number": {"value": "204511", "label": "NORTHWIND AVIATION, LLC", "confidence": 0.98,
                        "heard": "204511", "speaker": "caller"},
    "icao": {"value": "KTEB", "label": "Meadowlark Jet Center", "confidence": 1.0,
             "heard": "Teterboro", "speaker": "caller"}
    // a variable whose fuzzy match Winnow settled also has "settled_by": "winnow"
  },
  "results": {"fuel_quote": {"args": {...}, "result": {...}, "at": 0.01}, ...},
  "decision": { ... },       // as above
  "waiting_for": null,
  "prompts": ["..."],         // every prompt so far
  "transcript": [{"speaker": "caller", "text": "...", "at": 0.0}, ...],
  "preview": ""              // the latest speculative text, cleared when a final line arrives
}
```

Unknown `call_id` returns `404 {"error": "not found"}`.

## The other endpoints

- `DELETE /calls/<call_id>` returns `{"deleted": true}`, or 404 with `{"deleted": false}`.
- `GET /functions` returns `[{"name", "needs", "optional", "auto", "description"}]`.
- `GET /healthz` returns `{"ok": true, "decider": "winnow" | "rules", "calls": 1}`.
- A body that isn't JSON returns `400 {"error": "body must be JSON"}`.

## Pushed decisions (optional)

Set `DECISION_WEBHOOK_URL` and every new decision is also POSTed there as `{"call_id": "...", ...decision}`, the same
fields as above. It is fire and forget with a 3 second timeout; failures are only logged.

## Example

```bash
curl -s localhost:8000/calls/demo/events -H 'Content-Type: application/json' \
  -d '{"role":"USER","content":"What is your price per gallon?","generationStage":"FINAL"}'
# decision: fuel_quote, ready false, missing customer_number and icao

curl -s localhost:8000/calls/demo/events -H 'Content-Type: application/json' \
  -d '{"role":"USER","content":"Account 204511 at Teterboro.","generationStage":"FINAL"}'
# fired: fuel_quote, dispute_check, location_info, customer_profile; decision: fuel_quote, ready true, by "waiting"
```
