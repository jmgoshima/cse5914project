from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# Load environment for local Elastic credentials if present
ENV_PATH = Path(__file__).parent.parent.parent / "elastic-start-local" / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

INDEX_NAME = os.getenv("ES_INDEX_NAME", "us_cities")
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", os.getenv("ES_LOCAL_PASSWORD"))

# Keep the metric order aligned with the LLM profile schema
PROFILE_METRICS: List[str] = [
    "Climate",
    "HousingCost",
    "HlthCare",
    "Crime",
    "Transp",
    "Educ",
    "Arts",
    "Recreat",
    "Econ",
    "Pop",
]

_es_client: Optional[Elasticsearch] = None


def _get_client() -> Elasticsearch:
    global _es_client
    if _es_client is None:
        auth = None
        if ES_USERNAME and ES_PASSWORD:
            auth = (ES_USERNAME, ES_PASSWORD)
        _es_client = Elasticsearch(
            ES_URL,
            basic_auth=auth,
            verify_certs=False,
            request_timeout=30,
        )
    return _es_client


def search_for_profile(
    vector: List[float],
    *,
    k: int = 5,
    num_candidates: Optional[int] = None,
    index: str = INDEX_NAME,
) -> Dict[str, Any]:
    """Run a kNN search for the provided profile vector."""
    if not vector:
        raise ValueError("Profile vector must contain at least one value.")

    client = _get_client()
    knn_body: Dict[str, Any] = {
        "field": "review_vector",
        "query_vector": vector,
        "k": k,
    }
    if num_candidates is not None:
        knn_body["num_candidates"] = num_candidates

    response = client.search(index=index, knn=knn_body)
    return response


def _load_vector_from_json(path: Path) -> List[float]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        vector = [float(v) for v in payload]
    elif isinstance(payload, dict):
        vector = []
        for key in PROFILE_METRICS:
            if key not in payload:
                raise ValueError(f"Missing '{key}' in profile JSON.")
            vector.append(float(payload[key]))
    else:
        raise ValueError("Profile JSON must be either a list of floats or an object with metric keys.")

    return vector


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run an Elasticsearch kNN search using a profile vector.")
    parser.add_argument("vector", help="Path to a JSON file containing the profile vector.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest neighbours to return.")
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=None,
        help="Optional num_candidates override for Elasticsearch kNN search.",
    )
    parser.add_argument(
        "--index",
        default=INDEX_NAME,
        help=f"Elasticsearch index name (default: {INDEX_NAME}).",
    )
    args = parser.parse_args()

    vector = _load_vector_from_json(Path(args.vector))
    response = search_for_profile(
        vector,
        k=args.top_k,
        num_candidates=args.num_candidates,
        index=args.index,
    )
    json.dump(response, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    _cli()
