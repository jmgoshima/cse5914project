from __future__ import annotations

"""
CLI demo to run the conversation agent locally without Elasticsearch.

Usage examples:

  # Interactive session
  python -m backend.langchain.cli_demo

  # Provide messages upfront (multi-turn)
  python -m backend.langchain.cli_demo -m "I work in tech and can be remote." -m "Budget around $2500" -m "Prefer warm weather and walkable areas"

  # Start from an existing profile JSON file
  python -m backend.langchain.cli_demo --start-profile ./profile.json

Environment:
  - If OPENAI_API_KEY (or another provider via LangChain) is configured and
    corresponding packages are installed, the agent will use the LLM path.
  - Otherwise it falls back to a deterministic heuristic updater.
"""

import argparse
import json
import sys
from typing import Any, Dict, List

from backend.langchain.schemas import Profile
from backend.langchain.conversation import (
    stepAgent,
    stepAgent_with_callback,
    is_profile_ready,
)
from backend.langchain.explain import explain

# Best-effort check to report whether LLM is wired up (optional)
try:  # type: ignore[attr-defined]
    from backend.langchain.conversation import _LLM as _AGENT_LLM  # pragma: no cover
except Exception:  # pragma: no cover
    _AGENT_LLM = None


def _pretty(obj: Any) -> str:
    try:
        if hasattr(obj, "model_dump_json"):
            return json.dumps(json.loads(obj.model_dump_json()), indent=2)
        return json.dumps(obj, indent=2)
    except Exception:
        return str(obj)


USE_LLM_REASONS: bool = False


def mock_on_ready(profile: Profile) -> Dict[str, Any]:
    """Return a fake search result; avoids any ES dependency."""
    # You can tailor the mock using simple heuristics from the profile
    warm = "warm" in (profile.preferred_climates or [])
    remote = bool(profile.wants_remote_friendly)
    budget = profile.budget_monthly_usd or 0

    cities: List[Dict[str, Any]] = []
    if warm:
        cities.append({
            "id": "1",
            "name": "Austin, TX",
            "score": 1.0,
            "reason": None,
        })
        cities.append({
            "id": "2",
            "name": "San Diego, CA",
            "score": 0.9,
            "reason": None,
        })
    else:
        cities.append({
            "id": "3",
            "name": "Seattle, WA",
            "score": 0.85,
            "reason": None,
        })

    # Filter mock by budget very loosely (purely illustrative)
    if budget and budget < 2000:
        cities = [c for c in cities if "Austin" in c["name"]]

    # Optionally generate LLM reasons
    if USE_LLM_REASONS:
        for c in cities:
            try:
                prompt = (
                    "You are a relocation advisor. In 2–4 sentences, explain why this city fits the user's preferences. "
                    "Reference cost of living, climate, safety, job market, and lifestyle if available. "
                    "No markdown, no lists.\n\n"
                    f"USER_PROFILE_JSON:\n{profile.model_dump_json()}\n\nCITY_NAME:\n{c['name']}\n\nREASON:"
                )
                c["reason"] = explain(prompt).strip()
            except Exception:
                c["reason"] = c.get("reason") or "(reason unavailable)"

    return {
        "count": len(cities),
        "cities": cities,
        "raw_query": {"mock": True},
        "notes": {
            "llm_mode": bool(_AGENT_LLM),
            "remote": remote,
            "budget": budget,
        }
    }


def run_messages(messages: List[str], start_profile_path: str | None, keep_going: bool) -> int:
    profile = Profile()
    if start_profile_path:
        try:
            with open(start_profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            profile = Profile(**data)
        except Exception as exc:
            print(f"Failed to load start profile from {start_profile_path}: {exc}")
            return 2

    print(f"Agent mode: {'LLM' if _AGENT_LLM else 'Heuristic'}\n")

    for i, msg in enumerate(messages, 1):
        out = stepAgent_with_callback(profile, msg, on_ready=mock_on_ready)
        profile = out["profile"]

        print(f"Turn {i} - User: {msg}")
        print("Updated Profile:")
        print(_pretty(profile))

        # If the agent suggested a follow-up question, show it to the user
        try:
            q = None
            if getattr(profile, "notes", None):
                q = profile.notes.get("next_question")
            if q:
                print(f"\nAgent: {q}")
        except Exception:
            pass

        if profile.notes.get("ready"):
            print("\nProfile ready. Mock recommendations:")
            print(_pretty(out.get("on_ready_result", {})))
            if not keep_going:
                return 0

        print("\n---\n")

    return 0


def run_interactive(start_profile_path: str | None) -> int:
    profile = Profile()
    if start_profile_path:
        try:
            with open(start_profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            profile = Profile(**data)
        except Exception as exc:
            print(f"Failed to load start profile from {start_profile_path}: {exc}")
            return 2

    print("Conversation agent demo (type 'exit' to quit).")
    print(f"Agent mode: {'LLM' if _AGENT_LLM else 'Heuristic'}\n")

    turn = 0
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()  # newline
            return 0

        if not msg or msg.lower() in {"exit", "quit"}:
            return 0

        turn += 1
        out = stepAgent_with_callback(profile, msg, on_ready=mock_on_ready)
        profile = out["profile"]

        print("Updated Profile:")
        print(_pretty(profile))

        # If the agent suggested a follow-up question, show it to the user
        try:
            q = None
            if getattr(profile, "notes", None):
                q = profile.notes.get("next_question")
            if q:
                print(f"\nAgent: {q}")
        except Exception:
            pass

        if profile.notes.get("ready"):
            print("\nProfile ready. Mock recommendations:")
            print(_pretty(out.get("on_ready_result", {})))
            print("\nType more to refine further, or 'exit' to quit.")
        print("---")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the conversation agent locally (no ES), with mock recommendations.")
    parser.add_argument(
        "-m", "--message", dest="messages", action="append", default=[],
        help="Provide a user message (repeatable for multi-turn). If omitted, runs interactive mode.",
    )
    parser.add_argument(
        "--start-profile", dest="start_profile", default=None,
        help="Path to a JSON file containing an initial Profile.",
    )
    parser.add_argument(
        "--keep-going", action="store_true",
        help="Do not exit when the profile becomes ready; continue processing messages.",
    )
    parser.add_argument(
        "--llm-reasons", action="store_true",
        help="Use the LLM to generate reasons for mock results (requires API key).",
    )

    args = parser.parse_args(argv)

    global USE_LLM_REASONS
    USE_LLM_REASONS = bool(args.llm_reasons)

    if args.messages:
        return run_messages(args.messages, args.start_profile, keep_going=args.keep_going)
    return run_interactive(args.start_profile)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
