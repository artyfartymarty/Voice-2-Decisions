from copilot.winnow import Winnower, judge, split_sentences, summarize, winnow


def kinds(text):
    return [m.kind for m in winnow(text)]


def test_decision():
    m = judge("We decided to go with the annual plan.")
    assert m.kind == "decision"
    assert m.tool_call["name"] == "log_decision"


def test_action_item_with_owner_and_due():
    m = judge("Sarah will send the contract by Friday.")
    assert m.kind == "action_item"
    assert m.owner == "Sarah"
    assert m.due.lower() == "friday"
    assert m.tool_call == {"name": "create_action_item", "args": {"text": m.text, "owner": "Sarah", "due": "Friday"}}


def test_pronoun_commitments_any_case():
    assert judge("I will send you the quote by end of day.").kind == "action_item"
    assert judge("i'll book the slot.").kind == "action_item"


def test_risk_question_and_key_moment():
    assert judge("My concern is the onboarding timeline.").kind == "risk"
    assert judge("When does the pilot start?").kind == "question"
    m = judge("The budget is $40,000 for the pilot.")
    assert m.kind == "key_moment"


def test_small_talk_is_dropped():
    assert kinds("Hi, how is the weather. Great to see you all.") == []


def test_agreed_alone_is_not_a_decision():
    # Seen on real ASR output: a speech line, not a decision.
    assert kinds("Whether we've seen eye to eye or rarely agreed at all, thank you.") == []
    assert kinds("We learned a great deal.") == []
    assert kinds("We agreed on the rollout plan.") == ["decision"]


REAL_ASR = (  # Nemotron output for tests/fixtures/sample_call.wav: note the dropped periods
    "Thanks for joining the renewal call today.  The budget for the pilot is forty thousand dollars after looking at "
    "both options, we decided to go with the annual plan Sarah will send a contract by Friday.  My main concern is "
    "the onboarding timeline.  Can you confirm the start date?"
)


def test_run_on_asr_sentence_is_split_into_clauses():
    moments = winnow(REAL_ASR)
    assert [m.kind for m in moments] == ["key_moment", "decision", "action_item", "risk", "question"]
    decision, action = moments[1], moments[2]
    assert decision.text == "we decided to go with the annual plan"
    assert (action.owner, action.due) == ("Sarah", "Friday")


def test_short_fragments_stay_attached():
    assert len(winnow("So Sarah will send it.")) == 1
    assert winnow("So Sarah will send it.")[0].text == "So Sarah will send it."


def test_split_sentences_keeps_unfinished_tail():
    done, tail = split_sentences("We agreed on the price. Sarah will send")
    assert done == ["We agreed on the price."]
    assert tail == "Sarah will send"


def test_partial_transcripts_decide_when_sentence_ends():
    w = Winnower()
    assert w.feed("We decided to go") == []
    assert w.pending == "We decided to go"
    out = w.feed("with the annual plan. Can you")
    assert [m.kind for m in out] == ["decision"]
    assert out[0].text == "We decided to go with the annual plan."
    assert w.pending == "Can you"
    out = w.feed("confirm the start date?")
    assert [m.kind for m in out] == ["question"]


def test_no_duplicates_across_partials():
    w = Winnower()
    first = w.feed("We decided to go with the annual plan.")
    again = w.feed("We decided to go with the annual plan.")
    assert len(first) == 1 and again == []


def test_flush_judges_the_tail():
    w = Winnower()
    w.feed("Let's go with vendor B")
    assert [m.kind for m in w.flush()] == ["decision"]
    assert w.pending == ""


def test_summary():
    moments = [m.as_dict() for m in winnow(
        "The budget is $40,000. We decided to go with the annual plan. "
        "Sarah will send the contract by Friday. My concern is the timeline. Can you confirm the start date?")]
    s = summarize(moments)
    assert s["headline"] == "We decided to go with the annual plan."
    assert s["counts"] == {"decision": 1, "action_item": 1, "risk": 1, "question": 1, "key_moment": 1,
                           "quote_request": 0, "dispute_risk": 0, "sentiment_shift": 0,
                           "tail_number": 0, "agent_prompt": 0}
    assert s["action_items"][0]["owner"] == "Sarah"


def test_summary_empty():
    assert summarize([])["headline"] == "No decisions or action items detected."
