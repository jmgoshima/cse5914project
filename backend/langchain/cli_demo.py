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


QUERY_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "search" / "query.py"

def _profile_to_query_payload(profile: Profile) -> Dict[str, Any]:
    notes = profile.notes if isinstance(profile.notes, dict) else {}
    payload: Dict[str, Any] = {
        "Climate": ", ".join(profile.preferred_climates) if profile.preferred_climates else None,
        "HousingCost": profile.housing_cost_target_max or profile.budget_monthly_usd,
        "HlthCare": profile.healthcare_min_score,
        "Crime": profile.safety_min_score,
        "Transp": profile.transit_min_score,
        "Educ": profile.education_min_score,
        "Arts": notes.get("arts"),
        "Recreat": notes.get("recreation"),
        "Econ": profile.economy_score_min,
        "Pop": profile.population_min if profile.population_min is not None else notes.get("population"),
    }
    return payload

def _parse_query_output(raw_output: str) -> Dict[str, Any]:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    header = None
    cities: List[Dict[str, Any]] = []
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
            cities.append({
                "id": None,
                "name": name,
                "score": score,
                "reason": None,
            })
    result: Dict[str, Any] = {
        "count": len(cities),
        "cities": cities,
        "raw_output": raw_output.strip(),
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
            "count": 0,
            "cities": [],
            "error": f"Query script not found at {script_path}",
            "profile_payload": payload,
        }

    cmd = [sys.executable, str(script_path), profile_json]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        return {
            "count": 0,
            "cities": [],
            "error": f"Query script failed with exit code {exc.returncode}",
            "stderr": (exc.stderr or "").strip(),
            "stdout": (exc.stdout or "").strip(),
            "profile_payload": payload,
            "command": cmd,
        }

    result = _parse_query_output(completed.stdout)
    if completed.stderr and completed.stderr.strip():
        result["stderr"] = completed.stderr.strip()
    result["profile_payload"] = payload
    result["command"] = cmd

    if USE_LLM_REASONS and result.get("cities"):
        for city in result["cities"]:
            if not city.get("name"):
                continue
            try:
                prompt = (
                    "You are a relocation advisor. In 2-4 sentences, explain why this city fits the user's preferences. "
                    "Reference cost of living, climate, safety, job market, and lifestyle if available. "
                    "No markdown, no lists.\n\n"
                    f"USER_PROFILE_JSON:\n{profile.model_dump_json()}\n\nCITY_NAME:\n{city['name']}\n\nREASON:"
                )
                city["reason"] = explain(prompt).strip()
            except Exception:
                city["reason"] = city.get("reason") or "(reason unavailable)"

    return result

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
        "--llm-reasons", action="store_true",
        help="Use the LLM to generate reasons for query results (requires API key).",
    )

    args = parser.parse_args(argv)

    global USE_LLM_REASONS
    USE_LLM_REASONS = bool(args.llm_reasons)

    if args.messages:
        return run_messages(args.messages, args.start_profile, keep_going=args.keep_going)
    return run_interactive(args.start_profile)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
