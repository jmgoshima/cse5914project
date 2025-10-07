from __future__ import annotations

"""
CLI demo to run the conversation agent locally and fetch search results from Elasticsearch.

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
from typing import Any, Dict, List, Optional

from backend.langchain.schemas import Profile
from backend.langchain.conversation import stepAgent_with_callback
from backend.search.query import search_for_profile, PROFILE_METRICS

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


SEARCH_TOP_K: int = 5
SEARCH_NUM_CANDIDATES: Optional[int] = None


def _profile_to_vector(profile: Profile) -> List[float]:
    vector: List[float] = []
    for metric in PROFILE_METRICS:
        value = getattr(profile, metric, None)
        if value is None:
            raise ValueError(f"Profile is missing a value for '{metric}'.")
        vector.append(float(value))
    return vector


def _search_on_ready(profile: Profile) -> Dict[str, Any]:
    try:
        vector = _profile_to_vector(profile)
    except ValueError as exc:
        return {"error": str(exc)}

    try:
        return search_for_profile(
            vector,
            k=SEARCH_TOP_K,
            num_candidates=SEARCH_NUM_CANDIDATES,
        )
    except Exception as exc:
        return {"error": f"Search failed: {exc}"}


def _require_llm() -> bool:
    if _AGENT_LLM:
        return True
    print(
        "LLM not configured. Set GOOGLE_API_KEY (or GEMINI credentials) before running the CLI."
    )
    return False


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

    if not _require_llm():
        return 2

    print("Agent mode: LLM\n")

    for i, msg in enumerate(messages, 1):
        try:
            out = stepAgent_with_callback(profile, msg, on_ready=_search_on_ready)
        except RuntimeError as err:
            print(f"Agent error: {err}")
            return 2
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
            result = out.get("on_ready_result", {})
            if isinstance(result, dict) and result.get("error"):
                print("\nSearch error:")
                print(result["error"])
            else:
                print("\nProfile ready. Search results:")
                print(_pretty(result))
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

    if not _require_llm():
        return 2

    print("Conversation agent demo (type 'exit' to quit).")
    print("Agent mode: LLM\n")

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
        try:
            out = stepAgent_with_callback(profile, msg, on_ready=_search_on_ready)
        except RuntimeError as err:
            print(f"Agent error: {err}")
            return 2
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
            result = out.get("on_ready_result", {})
            if isinstance(result, dict) and result.get("error"):
                print("\nSearch error:")
                print(result["error"])
            else:
                print("\nProfile ready. Search results:")
                print(_pretty(result))
            print("\nType more to refine further, or 'exit' to quit.")
        print("---")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the conversation agent locally and fetch Elasticsearch recommendations.")
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
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest neighbours to request from Elasticsearch (default: 5).",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=None,
        help="Optional Elasticsearch num_candidates override for kNN search.",
    )

    args = parser.parse_args(argv)

    global SEARCH_TOP_K, SEARCH_NUM_CANDIDATES
    SEARCH_TOP_K = max(1, args.top_k)
    SEARCH_NUM_CANDIDATES = args.num_candidates if args.num_candidates is None else max(args.num_candidates, args.top_k)

    if args.messages:
        return run_messages(args.messages, args.start_profile, keep_going=args.keep_going)
    return run_interactive(args.start_profile)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
