"""The verdict line is what the eval scores."""

from src.assistant import extract_verdict


def test_explicit_line():
    assert extract_verdict("Verdict: worth it\nCritics liked it.") == "worth it"
    assert extract_verdict("Verdict: mixed\nSome did, some didn't.") == "mixed"
    assert extract_verdict("Verdict: skip\nNot enough here.") == "skip"


def test_missing():
    assert extract_verdict("I don't know what to tell you.") is None
    assert extract_verdict("") is None
