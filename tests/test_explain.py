import os
import sys
import importlib
import pytest


@pytest.mark.parametrize("prompt, expected_start", [
    ("Hello world", "stub-explain:"),
    ("Some long prompt\nwith lines", "stub-explain:"),
])
def test_explain_stub(monkeypatch, prompt, expected_start):
    # Ensure provider keys are absent so explain uses stub path
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    module_name = "backend.langchain.explain"
    if module_name in sys.modules:
        mod = importlib.reload(sys.modules[module_name])
    else:
        mod = importlib.import_module(module_name)
    def fake_getenv(key, default=None):
        if key in {"OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"}:
            return None
        return os.environ.get(key, default)
    monkeypatch.setattr(mod.os, "getenv", fake_getenv, raising=False)
    out = mod.explain(prompt)
    assert isinstance(out, str)
    assert out.startswith(expected_start)


def test_explain_empty_prompt():
    mod = importlib.import_module("backend.langchain.explain")
    with pytest.raises(ValueError):
        mod.explain("   \n  \t")
