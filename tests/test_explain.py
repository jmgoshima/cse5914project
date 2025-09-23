import os
import importlib
import pytest


@pytest.mark.parametrize("prompt, expected_start", [
    ("Hello world", "stub-explain:"),
    ("Some long prompt\nwith lines", "stub-explain:"),
])
def test_explain_stub(monkeypatch, prompt, expected_start):
    # Ensure OPENAI_API_KEY is not set so explain uses stub path
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mod = importlib.import_module("backend.langchain.explain")
    out = mod.explain(prompt)
    assert isinstance(out, str)
    assert out.startswith(expected_start)


def test_explain_empty_prompt():
    mod = importlib.import_module("backend.langchain.explain")
    with pytest.raises(ValueError):
        mod.explain("   \n  \t")
