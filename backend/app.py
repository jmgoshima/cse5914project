from __future__ import annotations
import os, json, uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from flask import Flask, jsonify, request, g
from flask_cors import CORS

# ---- your modules (heavy lifting lives here) ----
from langchain.conversation import stepAgent             # agent: fills Profile
from langchain.explain import explain                    # generic LLM call used for "reasons"
from langchain.schemas import Profile, Weights, HardFilters
from search.query import buildQuery                      # build ES query from Profile
from search.es_client import get_client                  # ES client factory
from utils.cache import get_cache                        # cache (redis or in-memory)
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
    )

    CORS(app, resources={r"/*": {"origins": app.config["CORS_ORIGINS"]}})

    _es, _cache = None, None
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
    def _recommend_for_profile(profile: Profile, top_k: int, take: int, index: str):
        query = buildQuery(profile=profile, topN=top_k)
        try:
            resp = es().search(index=index, body=query, from_=0, size=top_k)
        except Exception as e:
            raise RuntimeError(f"Elasticsearch query failed: {e}")

        hits = (resp.get("hits", {}).get("hits", []) or [])[:take]

        def make_reason(src: Dict[str, Any]) -> str:
            prompt = (
                "You are a relocation advisor. In 2–4 sentences, explain why this city fits the user's preferences. "
                "Reference cost of living, climate, safety, job market, and lifestyle if available. "
                "No markdown, no lists.\n\n"
                f"USER_PROFILE_JSON:\n{profile.model_dump_json()}\n\nCITY_FACTS_JSON:\n{json.dumps(src, ensure_ascii=False)}\n\nREASON:"
            )
            return explain(prompt).strip()

        cities = []
        for h in hits:
            src = h.get("_source", {})
            cache_key = f"reason:{h.get('_id')}:{hash(profile.model_dump_json())}"
            reason = cache().get(cache_key) or make_reason(src)
            try:
                cache().set(cache_key, reason, ex=app.config["CACHE_TTL_SECONDS"])
            except Exception:
                pass
            cities.append({
                "id": h.get("_id"),
                "name": src.get("name") or src.get("city") or "Unknown",
                "score": h.get("_score"),
                "source": src,
                "reason": reason
            })

        return {
            "count": len(cities),
            "cities": cities,
            "raw_query": query
        }

    @app.before_request
    def _before():
        g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        g.started_at = datetime.utcnow()

    @app.get("/health")
    def health():
        try:
            es_ok = bool(es().ping())
        except Exception:
            es_ok = False
        return ok({"status": "up", "time": datetime.utcnow().isoformat() + "Z", "elasticsearch": "up" if es_ok else "down"})

    # 1) conversation: agent fills profile
    @app.post("/conversation/step")
    def conversation_step():
        body = request.get_json(silent=True) or {}
        msg = body.get("message")
        if not msg:
            return err("Missing 'message'.", 422)
        profile, weights, hard_filters = parse_envelope(body)
        updated = stepAgent(profile=profile, message=msg)
        return ok({"profile": json.loads(updated.model_dump_json()),
                   "weights": json.loads(weights.model_dump_json()) if weights else None,
                   "hard_filters": json.loads(hard_filters.model_dump_json()) if hard_filters else None})

    # 1b) conversation turn with auto-handoff to search+reasons when ready
    @app.post("/agent/turn")
    def agent_turn():
        body = request.get_json(silent=True) or {}
        msg = body.get("message")
        if not msg:
            return err("Missing 'message'.", 422)
        if "profile" not in body:
            return err("Missing 'profile'.", 422)

        # optional overrides for the next steps
        top_k = max(1, min(int(body.get("topK", app.config["MAX_RESULTS_DEFAULT"])), app.config["MAX_RESULTS_LIMIT"]))
        take = max(1, min(int(body.get("take", app.config["TAKE_DEFAULT"])), top_k))
        index = body.get("index", app.config["ES_INDEX_DEFAULT"])

        profile, _, _ = parse_envelope(body)
        updated = stepAgent(profile=profile, message=msg)
        payload: Dict[str, Any] = {
            "profile": json.loads(updated.model_dump_json()),
            "ready": bool(updated.notes.get("ready")),
            "action": "recommend" if updated.notes.get("ready") else "continue"
        }

        if payload["ready"]:
            try:
                rec = _recommend_for_profile(updated, top_k=top_k, take=take, index=index)
                payload.update(rec)
            except RuntimeError as e:
                return err("Elasticsearch query failed.", 502, {"reason": str(e)})

        return ok(payload)

    # 2) search: ES top-k cities from profile
    @app.post("/search/places")
    def search_places():
        body = request.get_json(silent=True) or {}
        if "profile" not in body:
            return err("Missing 'profile'.", 422)
        profile, _, _ = parse_envelope(body)
        top_k = max(1, min(int(body.get("topK", app.config["MAX_RESULTS_DEFAULT"])), app.config["MAX_RESULTS_LIMIT"]))
        index = body.get("index", app.config["ES_INDEX_DEFAULT"])
        offset = int(body.get("from", 0))
        query = buildQuery(profile=profile, topN=top_k)
        try:
            resp = es().search(index=index, body=query, from_=offset, size=top_k)
        except Exception as e:
            return err("Elasticsearch query failed.", 502, {"reason": str(e)})
        hits = resp.get("hits", {}).get("hits", [])
        total = resp.get("hits", {}).get("total", {}).get("value", len(hits))
        results = [{"id": h.get("_id"), "score": h.get("_score"), "source": h.get("_source", {}), "highlight": h.get("highlight")} for h in hits]
        return ok({"total": total, "count": len(results), "from": offset, "results": results, "raw_query": query})

    # 3) recommend: ES → take top N → LLM adds reasons
    @app.post("/recommend")
    def recommend():
        body = request.get_json(silent=True) or {}
        if "profile" not in body:
            return err("Missing 'profile'.", 422)
        profile, _, _ = parse_envelope(body)
        top_k = max(1, min(int(body.get("topK", app.config["MAX_RESULTS_DEFAULT"])), app.config["MAX_RESULTS_LIMIT"]))
        take = max(1, min(int(body.get("take", app.config["TAKE_DEFAULT"])), top_k))
        index = body.get("index", app.config["ES_INDEX_DEFAULT"])

        try:
            rec = _recommend_for_profile(profile, top_k=top_k, take=take, index=index)
        except RuntimeError as e:
            return err("Elasticsearch query failed.", 502, {"reason": str(e)})

        return ok({
            "profile": json.loads(profile.model_dump_json()),
            **rec
        })

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8080")), debug=os.getenv("FLASK_DEBUG", "false").lower()=="true")