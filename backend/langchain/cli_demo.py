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
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.langchain.schemas import Profile, RecommendationResults
from backend.langchain.conversation import (
    stepAgent,
    stepAgent_with_callback,
    is_profile_ready,
)
from backend.langchain.explain import explain
from backend.search.query import _profile_to_vector, DATA_FIELDS

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


USE_LLM_REASONS: bool = True


QUERY_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "search" / "query.py"

def _profile_to_query_payload(profile: Profile) -> Dict[str, Any]:
    """Return a dict aligned with DATA_FIELDS using the same scaling as query.py."""
    vector = _profile_to_vector(profile)
    payload: Dict[str, Any] = {}
    for field, value in zip(DATA_FIELDS, vector):
        payload[field] = float(value)
    notes = profile.notes if isinstance(profile.notes, dict) else {}
    city_label = None
    if isinstance(notes.get("qual_answers"), dict):
        city_label = notes["qual_answers"].get("city")
    if not city_label:
        direct_city = notes.get("city")
        if isinstance(direct_city, str):
            city_label = direct_city
    if city_label:
        payload["City"] = city_label
    return payload

def _parse_query_output(raw_output: str) -> Dict[str, Any]:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    header = None
    reasoning: List[Dict[str, Any]] = []
    rank = 1
    for line in lines:
        if header is None and line.lower().startswith("nearest neighbors"):
            header = line
            continue
        match = re.match(r"(.+?)\s+\(score=([0-9.]+)\)", line)
        if match:
            name = match.group(1).strip()
            score_str = match.group(2)
            try:
                score = float(score_str)
            except ValueError:
                score = score_str
            reasoning.append({
                "city": name,
                "score": score,
                "reason": None,
                "rank": rank,
            })
            rank += 1
    result: Dict[str, Any] = {
        "raw_output": raw_output.strip(),
        "reasoning": reasoning,
    }
    if header:
        result["header"] = header
    return result

def query_on_ready(profile: Profile) -> Dict[str, Any]:
    payload = _profile_to_query_payload(profile)
    profile_json = json.dumps(payload)
    script_path = QUERY_SCRIPT_PATH

    if not script_path.exists():
        return {
            "reasoning": [],
            "error": f"Query script not found at {script_path}",
            "profile_payload": payload,
        }

    cmd = [sys.executable, str(script_path), profile_json]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        return {
            "reasoning": [],
            "error": f"Query script failed with exit code {exc.returncode}",
            "stderr": (exc.stderr or "").strip(),
            "stdout": (exc.stdout or "").strip(),
            "profile_payload": payload,
            "command": cmd,
        }

    raw_result = _parse_query_output(completed.stdout)
    if completed.stderr and completed.stderr.strip():
        raw_result["stderr"] = completed.stderr.strip()
    raw_result["profile_payload"] = payload
    raw_result["command"] = cmd

    rec = RecommendationResults(**raw_result)

    if USE_LLM_REASONS and rec.reasoning:
        profile_json = profile.model_dump_json()
        for entry in rec.reasoning:
            if not (entry and entry.city):
                continue
            try:
                prompt = (
                    "You are a relocation advisor. In 2-4 sentences, explain why this city fits the user's preferences. "
                    "Reference cost of living, climate, safety, job market, and lifestyle if available. "
                    "Do not use markdown or bullet points.\n\n"
                    f"USER_PROFILE_JSON:\n{profile_json}\n\n"
                    f"CITY_NAME:\n{entry.city}\n"
                )
                if entry.score is not None:
                    prompt += f"\nMATCH_CONFIDENCE:\n{entry.score}\n"
                prompt += "\nREASON:"
                reason_text = explain(prompt).strip()
            except Exception:
                reason_text = entry.reason or "(reason unavailable)"
            entry.reason = reason_text

    return rec.model_dump(
        exclude={"reasoning": {"__all__": {"score", "id", "rank"}}},
        exclude_none=True,
    )

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
        out = stepAgent_with_callback(profile, msg, on_ready=query_on_ready)
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
            print("\nProfile ready. Query results:")
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

    seeded = stepAgent_with_callback(profile, "", on_ready=query_on_ready)
    profile = seeded["profile"]
    initial_question = None
    try:
        if isinstance(profile.notes, dict):
            turns = profile.notes.get("turns")
            if isinstance(turns, list) and turns and not turns[0]:
                profile.notes["turns"] = [turn for turn in turns if turn]
            initial_question = profile.notes.get("next_question")
    except Exception:
        initial_question = None
    if initial_question:
        print(f"Agent: {initial_question}")

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
        out = stepAgent_with_callback(profile, msg, on_ready=query_on_ready)
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
            print("\nProfile ready. Query results:")
            print(_pretty(out.get("on_ready_result", {})))
            print("\nType more to refine further, or 'exit' to quit.")
        print("---")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the conversation agent locally (no ES), using the search query integration.")
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
        "--no-llm-reasons",
        dest="llm_reasons",
        action="store_false",
        help="Skip LLM-generated explanations for nearest-neighbor results.",
    )
    parser.add_argument(
        "--llm-reasons",
        dest="llm_reasons",
        action="store_true",
        help="Generate LLM explanations for nearest-neighbor results (default).",
    )
    parser.set_defaults(llm_reasons=True)

    args = parser.parse_args(argv)

    global USE_LLM_REASONS
    USE_LLM_REASONS = bool(args.llm_reasons)

    if args.messages:
        return run_messages(args.messages, args.start_profile, keep_going=args.keep_going)
    return run_interactive(args.start_profile)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
