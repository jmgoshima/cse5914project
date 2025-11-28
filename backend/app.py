from __future__ import annotations
import os, json, uuid, threading, queue
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Tuple, List
from flask import Flask, jsonify, request, g, Blueprint, Response, stream_with_context
from flask_cors import CORS

try:  # ensure Gemini creds load when app boots
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

# ---- your modules (heavy lifting lives here) ----
from backend.langchain.conversation import stepAgent             # agent: fills Profile
from backend.langchain.explain import explain                    # generic LLM call used for "reasons"
from backend.langchain.schemas import (
    Profile,
    Weights,
    HardFilters,
    RecommendationResults,
    ReasoningEntry,
)
from backend.search.query import buildQuery, _profile_to_vector, DATA_FIELDS
from backend.search.es_client import get_client                  # ES client factory
from backend.utils.cache import get_cache                        # cache (redis or in-memory)
# --------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__)
    app.config.update(
        ES_INDEX_DEFAULT=os.getenv("ES_INDEX_DEFAULT", "cities"),
        MAX_RESULTS_DEFAULT=int(os.getenv("MAX_RESULTS_DEFAULT", "20")),
        MAX_RESULTS_LIMIT=int(os.getenv("MAX_RESULTS_LIMIT", "100")),
        TAKE_DEFAULT=int(os.getenv("TAKE_DEFAULT", "5")),
        CORS_ORIGINS=os.getenv("CORS_ORIGINS", "*"),
        CACHE_TTL_SECONDS=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
        # CONVERSATION_TTL_SECONDS=int(os.getenv("CONVERSATION_TTL_SECONDS", "900")),
    )

    api = Blueprint("api", __name__, url_prefix="/api")

    CORS(app, resources={r"/*": {"origins": app.config["CORS_ORIGINS"]}})

    _es, _cache = None, None
    _conversation_state: Dict[str, Dict[str, Any]] = {}
    _event_streams: Dict[str, queue.Queue] = {}
    _conversation_lock = threading.Lock()
    def es():
        nonlocal _es
        if _es is None:
            _es = get_client()
        return _es
    def cache():
        nonlocal _cache
        if _cache is None:
            _cache = get_cache()
        return _cache

    def _get_stream_queue(conversation_id: str) -> queue.Queue:
        with _conversation_lock:
            q = _event_streams.get(conversation_id)
            if q is None:
                q = queue.Queue()
                _event_streams[conversation_id] = q
            return q

    def _close_stream(conversation_id: str) -> None:
        with _conversation_lock:
            q = _event_streams.pop(conversation_id, None)
        if q:
            q.put(None)

    def _publish_event(conversation_id: Optional[str], event: str, data: Dict[str, Any]) -> None:
        if not conversation_id:
            return
        try:
            q = _get_stream_queue(conversation_id)
            q.put({"event": event, "data": data})
        except Exception:
            pass

    def _format_event(event: str, data: Dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _load_conversation_state(conversation_id: str) -> Optional[Tuple[Profile, Optional[Weights], Optional[HardFilters]]]:
        if not conversation_id:
            return None
        with _conversation_lock:
            state = _conversation_state.get(conversation_id)
            if not state:
                return None
            # expires_at = state.get("expires_at")
            # if isinstance(expires_at, datetime) and expires_at <= datetime.utcnow():
            #     _conversation_state.pop(conversation_id, None)
            #     _close_stream(conversation_id)
            #     return None
            state = dict(state)
        try:
            profile_json = state.get("profile")
            weights_json = state.get("weights")
            hard_filters_json = state.get("hard_filters")
            if isinstance(profile_json, str):
                profile = Profile.model_validate_json(profile_json)
            elif isinstance(profile_json, dict):
                profile = Profile(**profile_json)
            else:
                profile = Profile()
            if isinstance(weights_json, str):
                weights = Weights.model_validate_json(weights_json)
            elif isinstance(weights_json, dict):
                weights = Weights(**weights_json)
            else:
                weights = None
            if isinstance(hard_filters_json, str):
                hard_filters = HardFilters.model_validate_json(hard_filters_json)
            elif isinstance(hard_filters_json, dict):
                hard_filters = HardFilters(**hard_filters_json)
            else:
                hard_filters = None
        except Exception:
            return None
        return profile, weights, hard_filters

    def _store_conversation_state(conversation_id: str, profile: Profile, weights: Optional[Weights], hard_filters: Optional[HardFilters]) -> None:
        # ttl_seconds = int(app.config.get("CONVERSATION_TTL_SECONDS", 900))
        # expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        state = {
            "profile": profile.model_dump_json(),
            "weights": weights.model_dump_json() if weights else None,
            "hard_filters": hard_filters.model_dump_json() if hard_filters else None,
            # "expires_at": expires_at,
        }
        with _conversation_lock:
            _conversation_state[conversation_id] = state

    def _touch_conversation_state(conversation_id: str) -> None:
        # ttl_seconds = int(app.config.get("CONVERSATION_TTL_SECONDS", 900))
        with _conversation_lock:
            state = _conversation_state.get(conversation_id)
            # if state:
            #     state["expires_at"] = datetime.utcnow() + timedelta(seconds=ttl_seconds)


    # ---------- small helpers ----------
    def ok(data: Any, status: int = 200):
        return jsonify({"ok": True, "data": data, "request_id": g.request_id}), status
    def err(msg: str, status: int = 400, details: Optional[Dict] = None):
        return jsonify({"ok": False, "error": {"message": msg, "details": details or {}}, "request_id": g.request_id}), status
    def parse_envelope(obj: Dict[str, Any]) -> Tuple[Profile, Optional[Weights], Optional[HardFilters]]:
        if "profile" in obj:
            p = Profile(**obj["profile"])
            w = Weights(**obj["weights"]) if obj.get("weights") else None
            h = HardFilters(**obj["hard_filters"]) if obj.get("hard_filters") else None
        else:
            p, w, h = Profile(**obj), None, None
        return p, w, h

    # --- internal helper used by /recommend and agent auto-flow ---
    def _recommend_for_profile(profile: Profile, top_k: int, take: int, index: str, progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        query = buildQuery(profile=profile, topN=top_k)
        if progress_cb:
            progress_cb("search_started", {"topK": top_k, "index": index})
        try:
            resp = es().search(index=index, body=query, from_=0)
        except Exception as e:
            raise RuntimeError(f"Elasticsearch query failed: {e}")

        all_hits = resp.get("hits", {}).get("hits", []) or []
        hits = all_hits[:take]
        if progress_cb:
            progress_cb("search_complete", {"found": len(all_hits), "returning": len(hits)})

        def make_reason(src: Dict[str, Any]) -> str:
            prompt = (
                "You are a relocation advisor. In 2-4 sentences, explain why this city fits the user's preferences. "
                "Reference cost of living, climate, safety, job market, and lifestyle if available. "
                "No markdown, no lists.\n\n"
                f"USER_PROFILE_JSON:\n{profile.model_dump_json()}\n\nCITY_FACTS_JSON:\n{json.dumps(src, ensure_ascii=False)}\n\nREASON:"
            )
            return explain(prompt).strip()

        try:
            profile_vector = _profile_to_vector(profile)
            profile_payload = {
                field: float(value) for field, value in zip(DATA_FIELDS, profile_vector)
            }
        except Exception:
            profile_payload = None

        header = "Nearest neighbors to input vector:"
        raw_lines: List[str] = []
        reasoning_entries: List[ReasoningEntry] = []
        if progress_cb and hits:
            progress_cb("reasoning_started", {"count": len(hits)})
        profile_hash = hash(profile.model_dump_json())
        for idx, h in enumerate(hits, start=1):
            src = h.get("_source", {})
            cache_key = f"reason:{h.get('_id')}:{profile_hash}"
            reason = cache().get(cache_key) or make_reason(src)
            try:
                cache().set(cache_key, reason, ex=app.config["CACHE_TTL_SECONDS"])
            except Exception:
                pass
            city_name = src.get("name") or src.get("city") or "Unknown"
            score_raw = h.get("_score")
            score_value = float(score_raw) if isinstance(score_raw, (int, float)) else None
            if isinstance(score_value, float):
                raw_lines.append(f"  {idx}. {city_name} (score={score_value:.4f})")
            else:
                raw_lines.append(f"  {idx}. {city_name}")
            reasoning_entries.append(
                ReasoningEntry(
                    city=city_name,
                    reason=reason,
                    score=score_value,
                    id=str(h.get("_id")) if h.get("_id") is not None else None,
                    rank=idx,
                )
            )
            if progress_cb:
                progress_cb("reasoning_progress", {"current": idx, "total": len(hits)})
        if progress_cb:
            progress_cb("reasoning_complete", {"count": len(reasoning_entries)})

        raw_output_lines = [header] + raw_lines if raw_lines else [header]
        raw_output = "\n".join(raw_output_lines)
        results = RecommendationResults(
            header=header,
            raw_output=raw_output,
            profile_payload=profile_payload,
            reasoning=reasoning_entries,
        )

        return results.model_dump(
            exclude={"reasoning": {"__all__": {"score", "id", "rank"}}},
            exclude_none=True,
        )

    @app.before_request
    def _before():
        g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        # g.started_at = datetime.utcnow()

    # @app.get("/health")
    # def health():
    #     try:
    #         es_ok = bool(es().ping())
    #     except Exception:
    #         es_ok = False
    #     return ok({"status": "up", "time": datetime.utcnow().isoformat() + "Z", "elasticsearch": "up" if es_ok else "down"})
    

    # # 1) conversation: agent fills profile
    # @api.post("/userStep")
    # def user_step():
    #     body = request.get_json(silent=True) or {}
    #     msg = body.get("message")
    #     if not msg:
    #         return err("Missing 'message'.", 422)
    #     conversation_id = body.get("conversationId") or body.get("conversation_id")
    #     conversation_id = str(conversation_id) if conversation_id else None

    #     profile: Profile
    #     weights: Optional[Weights]
    #     hard_filters: Optional[HardFilters]

    #     if "profile" in body and isinstance(body["profile"], dict):
    #         profile, weights, hard_filters = parse_envelope(body)
    #     elif conversation_id:
    #         loaded = _load_conversation_state(conversation_id)
    #         if not loaded:
    #             return err("Conversation expired or unknown. Send a full profile to start a new conversation.", 404)
    #         profile, weights, hard_filters = loaded
    #         _touch_conversation_state(conversation_id)
    #     else:
    #         profile, weights, hard_filters = Profile(), None, None
    #         conversation_id = str(uuid.uuid4())

    #     if isinstance(body.get("weights"), dict):
    #         try:
    #             weights = Weights(**body["weights"])
    #         except Exception as exc:
    #             return err("Invalid 'weights' payload.", 422, {"reason": str(exc)})

    #     if isinstance(body.get("hard_filters"), dict):
    #         try:
    #             hard_filters = HardFilters(**body["hard_filters"])
    #         except Exception as exc:
    #             return err("Invalid 'hard_filters' payload.", 422, {"reason": str(exc)})

    #     updated = stepAgent(profile=profile, message=msg)
    #     if conversation_id is None:
    #         conversation_id = str(uuid.uuid4())
    #     _store_conversation_state(conversation_id, updated, weights, hard_filters)
    #     response_payload = {
    #         "conversationId": conversation_id,
    #         "profile": json.loads(updated.model_dump_json()),
    #         "weights": json.loads(weights.model_dump_json()) if weights else None,
    #         "hard_filters": json.loads(hard_filters.model_dump_json()) if hard_filters else None,
    #     }
    #     _publish_event(conversation_id, "status", {"stage": "profile_updated", "ready": bool(updated.notes.get("ready"))})
    #     return ok(response_payload)

    # initialize the api 
    @api.post("/initialize")
    def initialize():
        conversation_id = str(uuid.uuid4())
        with _conversation_lock:
            _conversation_state[conversation_id] = {
                "profile": Profile().model_dump_json(),
                "weights": None,
                "hard_filters": None,
                "history": [],
                # "expires_at": datetime.utcnow() + timedelta(seconds=app.config["CONVERSATION_TTL_SECONDS"])
            }
        _publish_event(conversation_id, "status", {"stage": "initialized"})
        return ok({
            "conversationId": conversation_id,
            "history": [],
            "message": "Conversation initialized successfully."
        })
    

    # 1b) conversation turn with auto-handoff to search+reasons when ready
    @api.post("/chat")
    def chat_turn():
        body = request.get_json(silent=True) or {}
        message = body.get("message")
        conversation_id = body.get("conversationId")

        if not message:
            return err("Missing 'message'.", 422)
        if not conversation_id:
            return err("Missing 'conversationId'.", 422)

        loaded = _load_conversation_state(conversation_id)
        print(loaded)
        if not loaded:
            return err("Conversation not found or expired.", 404)

        profile, weights, hard_filters = loaded
        _touch_conversation_state(conversation_id)

        # Retrieve conversation history if present
        with _conversation_lock:
            state = _conversation_state.get(conversation_id, {})
            history = state.get("history", [])

        # Append user message
        history.append({"role": "user", "content": message})

        # Use LLM agent to get next step
        updated_profile = stepAgent(profile=profile, message=message)

        # Add system/agent response
        response_text = updated_profile.notes.get("response", "Profile updated.")
        history.append({"role": "assistant", "content": response_text})

        # Save back to conversation state
        with _conversation_lock:
            _conversation_state[conversation_id]["profile"] = updated_profile.model_dump_json()
            _conversation_state[conversation_id]["history"] = history
            # _conversation_state[conversation_id]["expires_at"] = datetime.utcnow() + timedelta(seconds=app.config["CONVERSATION_TTL_SECONDS"])

        payload = {
            "conversationId": conversation_id,
            "response": response_text,
            "profile": json.loads(updated_profile.model_dump_json()),
            "history": history,
            "ready": bool(updated_profile.notes.get("ready")),
        }

        # Optionally trigger recommendation when ready
        if payload["ready"]:
            _publish_event(conversation_id, "status", {"stage": "searching"})
            try:
                rec = _recommend_for_profile(updated_profile, top_k=10, take=5, index=app.config["ES_INDEX_DEFAULT"])
                payload["recommendations"] = rec
                _publish_event(conversation_id, "status", {"stage": "results_ready"})
            except RuntimeError as e:
                return err("Elasticsearch query failed.", 502, {"reason": str(e)})

        return ok(payload)

    # @api.post("/chat")
    # def chat_turn():
    #     body = request.get_json(silent=True) or {}
    #     msg = body.get("message")
    #     #conversation_id = body.get("conversationId")
    #     if not msg:
    #         return err("Missing 'message'.", 422)
    #     conversation_id = body.get("conversationId") or body.get("conversation_id")
    #     conversation_id = str(conversation_id) if conversation_id else None
        
    #     loaded = _load_conversation_state(conversation_id)
    #     if not loaded:
    #         return err("Conversation not found or expired.", 404)
        
    #     # optional overrides for the next steps
    #     top_k = max(1, min(int(body.get("topK", app.config["MAX_RESULTS_DEFAULT"])), app.config["MAX_RESULTS_LIMIT"]))
    #     take = max(1, min(int(body.get("take", app.config["TAKE_DEFAULT"])), top_k))
    #     index = body.get("index", app.config["ES_INDEX_DEFAULT"])

    #     profile: Profile
    #     weights: Optional[Weights]
    #     hard_filters: Optional[HardFilters]

    #     if "profile" in body and isinstance(body["profile"], dict):
    #         profile, weights, hard_filters = parse_envelope(body)
    #     elif conversation_id:
    #         loaded = _load_conversation_state(conversation_id)
    #         if not loaded:
    #             return err("Conversation expired or unknown. Send a full profile to continue.", 404)
    #         profile, weights, hard_filters = loaded
    #         _touch_conversation_state(conversation_id)
    #     else:
    #         return err("Missing 'profile'. Include the profile from the previous response or provide a conversationId.", 422)

    #     if isinstance(body.get("weights"), dict):
    #         try:
    #             weights = Weights(**body["weights"])
    #         except Exception as exc:
    #             return err("Invalid 'weights' payload.", 422, {"reason": str(exc)})

    #     if isinstance(body.get("hard_filters"), dict):
    #         try:
    #             hard_filters = HardFilters(**body["hard_filters"])
    #         except Exception as exc:
    #             return err("Invalid 'hard_filters' payload.", 422, {"reason": str(exc)})

    #     updated = stepAgent(profile=profile, message=msg)
    #     if conversation_id is None:
    #         conversation_id = str(uuid.uuid4())
    #     _store_conversation_state(conversation_id, updated, weights, hard_filters)
    #     _publish_event(conversation_id, "status", {"stage": "agent_updated", "ready": bool(updated.notes.get("ready"))})
    #     payload: Dict[str, Any] = {
    #         #"conversationId": conversation_id,
    #         "profile": json.loads(updated.model_dump_json()),
    #         "ready": bool(updated.notes.get("ready")),
    #         "action": "recommend" if updated.notes.get("ready") else "continue"
    #     }
    #     payload["conversationId"] = conversation_id

    #     if payload["ready"]:
    #         def _progress(stage: str, info: Dict[str, Any]) -> None:
    #             _publish_event(conversation_id, "status", {"stage": stage, **info})

    #         _publish_event(conversation_id, "status", {"stage": "searching", "topK": top_k, "take": take, "index": index})
    #         try:
    #             rec = _recommend_for_profile(updated, top_k=top_k, take=take, index=index, progress_cb=_progress)
    #             payload.update(rec)
    #             _publish_event(conversation_id, "status", {"stage": "results_ready", "count": rec.get("count", 0)})
    #         except RuntimeError as e:
    #             _publish_event(conversation_id, "status", {"stage": "error", "message": str(e)})
    #             return err("Elasticsearch query failed.", 502, {"reason": str(e)})
    #     else:
    #         _publish_event(conversation_id, "status", {"stage": "collecting_requirements"})

    #     return ok(payload)

    # @api.get("/stream")
    # def stream_events():
    #     conversation_id = request.args.get("session_id") or request.args.get("conversationId") or request.args.get("conversation_id")
    #     if not conversation_id:
    #         return err("Missing 'session_id'.", 422)
    #     event_queue = _get_stream_queue(conversation_id)

    #     def _event_source():
    #         yield _format_event("status", {"stage": "connected"})
    #         while True:
    #             try:
    #                 item = event_queue.get(timeout=15)
    #             except queue.Empty:
    #                 yield "event: ping\ndata: {}\n\n"
    #                 continue
    #             if item is None:
    #                 break
    #             event_name = item.get("event") or "message"
    #             data = item.get("data") or {}
    #             yield _format_event(event_name, data)

    #     response = Response(stream_with_context(_event_source()), mimetype="text/event-stream")
    #     response.headers["Cache-Control"] = "no-cache"
    #     response.headers["X-Accel-Buffering"] = "no"
    #     return response

    # # 2) search: ES top-k cities from profile
    # @api.post("/search/places")
    # def search_places():
    #     body = request.get_json(silent=True) or {}
    #     if "profile" not in body:
    #         return err("Missing 'profile'.", 422)
    #     profile, _, _ = parse_envelope(body)
    #     top_k = max(1, min(int(body.get("topK", app.config["MAX_RESULTS_DEFAULT"])), app.config["MAX_RESULTS_LIMIT"]))
    #     index = body.get("index", app.config["ES_INDEX_DEFAULT"])
    #     offset = int(body.get("from", 0))
    #     query = buildQuery(profile=profile, topN=top_k)
    #     try:
    #         resp = es().search(index=index, body=query, from_=offset, size=top_k)
    #     except Exception as e:
    #         return err("Elasticsearch query failed.", 502, {"reason": str(e)})
    #     hits = resp.get("hits", {}).get("hits", [])
    #     total = resp.get("hits", {}).get("total", {}).get("value", len(hits))
    #     results = [{"id": h.get("_id"), "score": h.get("_score"), "source": h.get("_source", {}), "highlight": h.get("highlight")} for h in hits]
    #     return ok({"total": total, "count": len(results), "from": offset, "results": results, "raw_query": query})

    # # 3) recommend: ES → take top N → LLM adds reasons
    # @api.post("/recommend")
    # def recommend():
    #     body = request.get_json(silent=True) or {}
    #     if "profile" not in body:
    #         return err("Missing 'profile'.", 422)
    #     profile, _, _ = parse_envelope(body)
    #     top_k = max(1, min(int(body.get("topK", app.config["MAX_RESULTS_DEFAULT"])), app.config["MAX_RESULTS_LIMIT"]))
    #     take = max(1, min(int(body.get("take", app.config["TAKE_DEFAULT"])), top_k))
    #     index = body.get("index", app.config["ES_INDEX_DEFAULT"])

    #     try:
    #         rec = _recommend_for_profile(profile, top_k=top_k, take=take, index=index)
    #     except RuntimeError as e:
    #         return err("Elasticsearch query failed.", 502, {"reason": str(e)})

    #     return ok({
    #         "profile": json.loads(profile.model_dump_json()),
    #         **rec
    #     })

    app.register_blueprint(api)

    return app


if __name__ == "__main__":
    app = create_app()
    debug_mode = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
        debug=debug_mode,
    )
