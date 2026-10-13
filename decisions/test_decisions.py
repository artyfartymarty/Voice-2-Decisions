"""Tests for the text-to-decisions service. Standard library only, no GPU or model:

    python -m unittest decisions.test_decisions -v
    docker compose run --rm api python -m unittest decisions.test_decisions
"""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .decider import RuleDecider, WinnowDecider
from .engine import Call, load_extractor
from .events import EventReader
from .server import App

ROOT = Path(__file__).resolve().parent.parent
EXTRACTOR = load_extractor(ROOT / "reference")


def play(name: str, decider=None) -> list[dict]:
    call = Call(name, EXTRACTOR, decider or RuleDecider())
    script = json.loads((ROOT / "demos" / f"{name}.json").read_text())
    return [call.feed(line["speaker"], line["text"]) for line in script], call


class Events(unittest.TestCase):
    def test_raw_nova_stream_uses_content_start_stage(self):
        r = EventReader()
        start = {"event": {"contentStart": {"contentId": "a", "role": "USER",
                                            "additionalModelFields": "{\"generationStage\":\"SPECULATIVE\"}"}}}
        text = {"event": {"textOutput": {"contentId": "a", "role": "USER", "content": "quote please"}}}
        self.assertEqual(r.read([start, text]), [{"speaker": "caller", "text": "quote please", "final": False}])

    def test_flat_and_plain_events(self):
        r = EventReader()
        self.assertEqual(r.read({"role": "ASSISTANT", "content": "hi", "generationStage": "FINAL"})[0]["speaker"], "agent")
        self.assertEqual(r.read({"speaker": "caller", "text": "hello"}), [{"speaker": "caller", "text": "hello", "final": True}])
        self.assertEqual(r.read({"role": "USER", "content": '{ "interrupted" : true }'}), [])


class Engine(unittest.TestCase):
    def test_quote_call(self):
        steps, call = play("demo_quote_call")
        self.assertEqual({k: v["value"] for k, v in call.variables.items()},
                         {"customer_number": "204511", "icao": "KVNY", "tail_number": "N88NW"})
        fired = [f["function"] for s in steps for f in s["fired"]]
        self.assertEqual(fired.count("fuel_quote"), 1)  # called once, as soon as customer + location were both known
        decisions = [s["decision"]["function"] for s in steps if s["decision"]]
        self.assertEqual(decisions, ["fuel_quote", "tail_check"])
        self.assertIn("0.2650", steps[2]["decision"]["say"])

    def test_dispute_call_surfaces_fees_then_price(self):
        steps, _ = play("demo_dispute_location")
        self.assertEqual([s["decision"]["function"] for s in steps if s["decision"]], ["dispute_check", "fuel_quote"])

    def test_frustrated_call(self):
        steps, call = play("demo_frustrated_caller")
        decisions = [s["decision"]["function"] for s in steps if s["decision"]]
        self.assertIn("escalate", decisions)
        self.assertEqual(call.variables["tail_number"]["value"], "N512BT")  # re-asked after a cut-off "bravo tango"
        self.assertTrue(any("incomplete" in p for s in steps for p in s["prompts"]))

    def test_ask_before_variables_then_surface_when_ready(self):
        call = Call("t", EXTRACTOR, RuleDecider())
        first = call.feed("caller", "What's your price per gallon?")
        self.assertFalse(first["decision"]["ready"])
        self.assertEqual(first["decision"]["missing"], ["customer_number", "icao"])
        call.feed("caller", "It's Northwind Aviation.")
        last = call.feed("caller", "We're going into Teterboro.")
        self.assertEqual(last["decision"]["function"], "fuel_quote")
        self.assertTrue(last["decision"]["ready"])
        self.assertEqual(last["decision"]["by"], "waiting")

    def test_misheard_name_is_matched_and_read_back(self):
        call = Call("t", EXTRACTOR, RuleDecider())
        out = call.feed("caller", "Hi, it's Priya from Blue Herring Jet Charter.")
        self.assertEqual(call.variables["customer_number"]["value"], "318842")
        self.assertIn("Confirm the account", out["prompts"][0])
        self.assertEqual(call.feed("caller", "Account 318842.")["prompts"], [])  # the number itself needs no read-back

    def test_tail_numbers_said_every_way(self):
        cases = {
            "Tail number November eight eight November Whiskey.": "N88NW",       # phonetic + digit words
            "Tail is November niner fife tree alpha bravo.": "N953AB",            # aviation digits
            "Tail number November one twenty three alpha bravo.": "N123AB",      # "one twenty three"
            "Tail number November seven seven hundred whiskey.": "N7700W",       # "seven hundred"
            "Tail number N 5 1 2 B T.": "N512BT",                                # letters, as Nova writes them
            "Tail number N512BT.": "N512BT",                                     # already a registration
            "Tail is five one two bravo tango.": "N512BT",                       # "November" dropped
            "Registration golf alpha bravo charlie delta.": "GABCD",             # non-US prefix
        }
        for said, tail in cases.items():
            call = Call("t", EXTRACTOR, RuleDecider())  # no customer, so no fleet list to check against
            call.feed("caller", said)
            self.assertEqual(call.variables.get("tail_number", {}).get("value"), tail, said)

    def test_tail_checked_against_the_customers_fleet(self):
        call = Call("t", EXTRACTOR, RuleDecider())
        call.feed("caller", "Northwind, account 204511.")
        out = call.feed("caller", "Tail number November one twenty three alpha bravo.")
        self.assertNotIn("tail_number", call.variables)  # valid format, but not one of Northwind's tails
        self.assertIn("isn't on file", out["prompts"][0])
        call.feed("caller", "Sorry, November eight eight November whiskey.")
        self.assertEqual(call.variables["tail_number"]["value"], "N88NW")

    def test_tail_self_corrections(self):
        cases = [  # (lines, tail committed or None, text expected in the last prompt)
            (["Tail number November five one two bravo delta.", "No, bravo tango, not bravo delta."], "N512BT", None),
            (["Tail number November five one two bravo delta, sorry I meant bravo tango."], "N512BT", None),
            (["Tail is November five one two, scratch that, November eight eight November whiskey."], "N88NW", None),
            (["Tail number November five one two bravo... no wait, November eight eight November whiskey."], "N88NW", None),
            (["Tail number November eight eight November whiskey.",
              "Sorry, that's wrong, it's November seven seven zero zero whiskey."], "N7700W", None),
            # doesn't add up: nothing committed, the agent reads it back (and the fallback model may be asked)
            (["Tail is November eight eight, oh wait forget the November, I meant tango eight eight November whiskey."],
             None, "Read the tail back"),
        ]
        for lines, tail, prompt in cases:
            call = Call("t", EXTRACTOR, RuleDecider())
            call.feed("caller", "Northwind, account 204511.")
            prompts = [p for line in lines for p in call.feed("caller", line)["prompts"]]
            self.assertEqual(call.variables.get("tail_number", {}).get("value"), tail, lines)
            if prompt:
                self.assertIn(prompt, prompts[-1])
        call = Call("t", EXTRACTOR, RuleDecider())  # one-letter fix, no fleet to check against
        call.feed("caller", "Tail number November eight eight November whiskey, I mean Victor.")
        self.assertEqual(call.variables["tail_number"]["value"], "N88NV")

    def test_invalid_n_numbers_are_rejected(self):
        call = Call("t", EXTRACTOR, RuleDecider())
        call.feed("caller", "Tail number November zero one two three.")  # N-numbers never start with 0
        self.assertNotIn("tail_number", call.variables)

    def test_speculative_text_changes_nothing(self):
        call = Call("t", EXTRACTOR, RuleDecider())
        call.feed("caller", "account 204511 quote at Teterboro", final=False)
        self.assertEqual(call.variables, {})
        self.assertEqual(call.results, {})

    def test_agent_lines_fill_variables_but_are_not_decided(self):
        call = Call("t", EXTRACTOR, RuleDecider())
        out = call.feed("agent", "Is that for Northwind at Van Nuys? What price did you get last time?")
        self.assertEqual(out["decision"], None)
        self.assertEqual([f["function"] for f in out["fired"]][0], "fuel_quote")


class FakeModel:
    """Stands in for the external extract model; records which kinds it was asked for."""

    def __init__(self, **answers):
        self.answers, self.asked = answers, []

    def extract(self, kind, text, recent):
        self.asked.append(kind)
        return self.answers.get(kind)


class FakeWinnow(BaseHTTPRequestHandler):
    pick = "fuel_quote"
    tail_said = 0.9
    customer_said = 0.9
    which = {}  # "which_customer_number" -> the option to pick (default: the first, i.e. the fuzzy match)
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeWinnow.seen.append(body)
        keys = list(body["questions"]["answer_with"]["criteria"])
        probs = {k: (0.8 if k == self.pick else 0.2 / (len(keys) - 1)) for k in keys}
        answers = {"answer_with": {"type": "choice", "choice": self.pick, "probabilities": probs, "confidence": 0.6}}
        for qid, q in body["questions"].items():
            if qid.startswith("which_"):
                pick = self.which.get(qid, next(iter(q["criteria"])))
                answers[qid] = {"type": "choice", "choice": pick, "probabilities": {k: 0.9 if k == pick else 0.02 for k in q["criteria"]}}
        for flag in ("tail_said", "customer_said"):
            if flag in body["questions"]:
                answers[flag] = {"type": "noul", "noul": getattr(self, flag)}
        out = json.dumps({"answers": answers}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class Winnow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeWinnow)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_choice_request_and_answer(self):
        call = Call("t", EXTRACTOR, WinnowDecider(self.url))
        call.feed("caller", "Northwind, account 204511.")
        out = call.feed("caller", "Flying into Van Nuys, what would that run us?")
        self.assertEqual(out["decision"]["function"], "fuel_quote")
        self.assertEqual(out["decision"]["by"], "winnow")
        self.assertAlmostEqual(out["decision"]["p"], 0.8)
        q = FakeWinnow.seen[-1]["questions"]["answer_with"]
        self.assertEqual(q["type"], "choice")
        self.assertIn("none", q["criteria"])
        self.assertEqual(FakeWinnow.seen[-1]["state"]["known_so_far"]["icao"], "Sunset Aviation Services")

    def test_none_and_low_confidence_surface_nothing(self):
        FakeWinnow.pick = "none"
        try:
            call = Call("t", EXTRACTOR, WinnowDecider(self.url))
            self.assertIsNone(call.feed("caller", "Thanks, have a good one.")["decision"])
        finally:
            FakeWinnow.pick = "fuel_quote"
        call = Call("t", EXTRACTOR, WinnowDecider(self.url), min_p=0.9)
        self.assertIsNone(call.feed("caller", "Northwind at Van Nuys, price?")["decision"])

    def test_fuzzy_name_is_settled_in_the_same_request(self):
        call = Call("t", EXTRACTOR, WinnowDecider(self.url))
        call.feed("caller", "This is Priya at Blue Herring Jet Charter, quote at Le Bourget please.")
        q = FakeWinnow.seen[-1]["questions"]
        self.assertEqual(len(FakeWinnow.seen[-1]["questions"]), 2)  # intent + which customer, one request
        self.assertIn("318842", q["which_customer_number"]["criteria"])
        self.assertIn("none", q["which_customer_number"]["criteria"])
        self.assertEqual(call.variables["customer_number"]["settled_by"], "winnow")

    def test_winnow_overrides_a_wrong_fuzzy_match_and_functions_rerun(self):
        FakeWinnow.which = {"which_customer_number": "377205"}
        try:
            call = Call("t", EXTRACTOR, WinnowDecider(self.url))
            out = call.feed("caller", "This is Priya at Blue Herring Jet Charter, quote at Le Bourget please.")
        finally:
            FakeWinnow.which = {}
        self.assertEqual(call.variables["customer_number"]["value"], "377205")
        self.assertEqual(out["decision"]["result"]["customer_number"], "377205")

    def test_tail_from_the_extract_model(self):
        # The parser can't read "one two three ay bee"; Winnow says a full tail was said; the model writes it.
        model = FakeModel(tail_number="N123AB")
        call = Call("t", EXTRACTOR, WinnowDecider(self.url), model=model)
        call.feed("caller", "The tail is one two three ay bee.")
        self.assertIn("tail_said", FakeWinnow.seen[-1]["questions"])
        self.assertEqual(call.variables["tail_number"]["value"], "N123AB")

        # Model is the main path: even a tail the parser could read goes to the model, and the agent reads it back.
        model = FakeModel(tail_number="N88NW")
        call = Call("t", EXTRACTOR, WinnowDecider(self.url), model=model)
        out = call.feed("caller", "Tail number November eight eight November whiskey.")
        self.assertEqual(call.variables["tail_number"]["value"], "N88NW")
        self.assertIn("Read the tail number back", out["prompts"][-1])
        # Not asked when Winnow says no full tail was said.
        FakeWinnow.tail_said = 0.1
        try:
            Call("t", EXTRACTOR, WinnowDecider(self.url), model=model).feed("caller", "The tail is one two three.")
        finally:
            FakeWinnow.tail_said = 0.9
        self.assertEqual(model.asked.count("tail_number"), 1)
        # EXTRACT_MODE=parser: the parser reads it and the model isn't asked.
        call = Call("t", EXTRACTOR, WinnowDecider(self.url), model=model, mode="parser")
        call.feed("caller", "Tail number November eight eight November whiskey.")
        self.assertNotIn("tail_said", FakeWinnow.seen[-1]["questions"])
        self.assertEqual(model.asked.count("tail_number"), 1)

        # Its answer is validated: not a registration, or not on the customer's fleet -> not used.
        call = Call("t", EXTRACTOR, WinnowDecider(self.url), model=FakeModel(tail_number="HELLO1"))
        call.feed("caller", "The tail is one two three ay bee.")
        self.assertNotIn("tail_number", call.variables)
        call = Call("t", EXTRACTOR, WinnowDecider(self.url), model=FakeModel(tail_number="N123AB"))
        call.feed("caller", "Account 204511.")
        out = call.feed("caller", "The tail is one two three ay bee.")
        self.assertNotIn("tail_number", call.variables)
        self.assertIn("isn't on file", out["prompts"][-1])

    def test_customer_name_from_the_extract_model(self):
        model = FakeModel(customer="Blue Herring Jet Charter")
        call = Call("t", EXTRACTOR, WinnowDecider(self.url), model=model)
        out = call.feed("caller", "Hi, it's Priya from Blue Herring Jet Charter.")
        self.assertIn("customer_said", FakeWinnow.seen[-1]["questions"])
        self.assertEqual(call.variables["customer_number"]["value"], "318842")  # matched against customers.csv
        self.assertIn("Confirm the account", out["prompts"][-1])
        self.assertEqual(model.asked.count("customer"), 1)
        # Known customer, no new name: not asked again.
        call.feed("caller", "We need fuel on Friday.")
        self.assertNotIn("customer_said", FakeWinnow.seen[-1]["questions"])

        # An account number said as digits is still matched directly, with no model call.
        model = FakeModel(customer="should not be asked")
        call = Call("t", EXTRACTOR, WinnowDecider(self.url), model=model)
        call.feed("caller", "Account 204511.")
        self.assertEqual(call.variables["customer_number"]["value"], "204511")
        self.assertEqual(model.asked, [])

        # A name that isn't on file is not committed; the agent asks for the account number.
        call = Call("t", EXTRACTOR, WinnowDecider(self.url), model=FakeModel(customer="Acme Rockets"))
        out = call.feed("caller", "This is Sam from Acme Rockets.")
        self.assertNotIn("customer_number", call.variables)
        self.assertIn("isn't a customer on file", out["prompts"][-1])

        # Winnow says no name was said: the model isn't asked.
        FakeWinnow.customer_said = 0.1
        try:
            model = FakeModel(customer="Northwind Aviation")
            Call("t", EXTRACTOR, WinnowDecider(self.url), model=model).feed("caller", "Hi, I need some fuel.")
        finally:
            FakeWinnow.customer_said = 0.9
        self.assertEqual(model.asked, [])

    def test_unreachable_winnow_falls_back_to_rules(self):
        call = Call("t", EXTRACTOR, WinnowDecider("http://127.0.0.1:9", timeout=0.5))
        out = call.feed("caller", "Northwind at Van Nuys, what's the price per gallon?")
        self.assertEqual(out["decision"]["function"], "fuel_quote")
        self.assertEqual(out["decision"]["by"], "rules (winnow unavailable)")


class Api(unittest.TestCase):
    def test_app_events(self):
        app = App({"DECIDER": "rules"})
        out = app.events("c1", [{"role": "USER", "content": "Northwind account 204511.", "generationStage": "FINAL"},
                                {"role": "USER", "content": "Price per gallon at Teterboro?", "generationStage": "FINAL"}])
        self.assertEqual(out["variables"], {"customer_number": "204511", "icao": "KTEB"})
        self.assertEqual(out["decision"]["function"], "fuel_quote")


    def test_simulated_nova_stream_gives_the_same_call(self):
        import itertools
        from .simulate import nova_events, timed
        app = App({"DECIDER": "rules"})
        ids, steps = itertools.count(1), []
        for turn in timed(json.loads((ROOT / "demos" / "demo_quote_call.json").read_text())):
            for _, ev in nova_events(turn, ids):
                steps += app.events("sim", ev)["steps"]
        self.assertTrue(any(not s["final"] for s in steps))  # partials were streamed
        _, call = play("demo_quote_call")
        self.assertEqual(app.call("sim")[0].variables.keys(), call.variables.keys())
        self.assertEqual([s["line"]["text"] for s in steps if s["final"]], [l["text"] for l in call.transcript])


class FakeSSM(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeSSM.seen.append((self.headers["X-Amz-Target"], self.headers["Authorization"], body))
        page = {"Parameters": [{"Name": "/v2d/EXTRACT_MODEL_API_KEY", "Value": "secret-key"},
                               {"Name": "/v2d/DECIDE_MIN_P", "Value": "0.7"}]}
        if not body.get("NextToken"):
            page["NextToken"] = "p2"
        else:
            page["Parameters"] = [{"Name": "/v2d/bedrock/region", "Value": "us-east-1"}]
        data = json.dumps(page).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class ParameterStore(unittest.TestCase):
    def test_secrets_fill_missing_variables_only(self):
        from . import ssm
        server = HTTPServer(("127.0.0.1", 0), FakeSSM)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        env = {"SSM_PARAMETER_PATH": "/v2d/", "SSM_ENDPOINT": f"http://127.0.0.1:{server.server_port}",
               "AWS_ACCESS_KEY_ID": "AKIDEXAMPLE", "AWS_SECRET_ACCESS_KEY": "x", "AWS_REGION": "us-east-1",
               "EXTRACT_MODEL_API_KEY": "", "DECIDE_MIN_P": "0.5"}
        try:
            filled = ssm.load_into(env)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(sorted(filled), ["BEDROCK_REGION", "EXTRACT_MODEL_API_KEY"])
        self.assertEqual(env["EXTRACT_MODEL_API_KEY"], "secret-key")
        self.assertEqual(env["DECIDE_MIN_P"], "0.5")  # set already: the environment wins
        target, auth, body = FakeSSM.seen[0]
        self.assertEqual(target, "AmazonSSM.GetParametersByPath")
        self.assertTrue(auth.startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"))
        self.assertEqual(body, {"Path": "/v2d", "Recursive": True, "WithDecryption": True})
        self.assertEqual(FakeSSM.seen[1][2]["NextToken"], "p2")

    def test_nothing_without_a_path(self):
        from . import ssm
        self.assertEqual(ssm.load_into({}), [])

if __name__ == "__main__":
    unittest.main()
