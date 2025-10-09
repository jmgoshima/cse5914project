from elasticsearch import Elasticsearch
import os
import json
import sys
from pathlib import Path
from typing import List, Sequence, Union
from dotenv import load_dotenv

env_path_relative = Path(__file__).parent.parent.parent / "elastic-start-local" / ".env"
load_dotenv(dotenv_path=env_path_relative)

# Build Elasticsearch client without auth if no local password is provided.
es_password = os.getenv("ES_LOCAL_PASSWORD")
es_client_kwargs = {"verify_certs": False}
if es_password:
    es_client_kwargs["basic_auth"] = ("elastic", es_password)
else:
    print(
        "Warning: ES_LOCAL_PASSWORD not set; attempting unauthenticated connection to Elasticsearch."
    )

es = Elasticsearch("http://localhost:9200", **es_client_kwargs)
index_name = "us_cities"

city_to_search = "Random City"

# # Fetch the city’s vector from ES
# city_doc = es.search(
#     index=index_name,
#     query={"term": {"city": city_to_search}},
#     size=1
# )

# if not city_doc["hits"]["hits"]:
#     raise ValueError(f"City '{city_to_search}' not found in index '{index_name}'.")

# query_vector = city_doc["hits"]["hits"][0]["_source"]["review_vector"]

EXPECTED_VECTOR_KEYS: Sequence[str] = (
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
)


def _normalize_query_values(values: Sequence[Union[int, float]]) -> List[float]:
    return [float(v) for v in values]


def _dict_to_vector(data: dict) -> List[float]:
    vector: List[float] = []
    for key in EXPECTED_VECTOR_KEYS:
        value = data.get(key)
        if value is None:
            vector.append(0.0)
            continue
        try:
            vector.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Value for '{key}' must be numeric or null, got {value!r}."
            ) from exc
    return vector


def get_query_vector_from_payload(payload: str) -> List[float]:
    data = json.loads(payload)

    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(
                "JSON payload must contain exactly one record when provided as a list."
            )
        data = data[0]

    if isinstance(data, dict):
        return _dict_to_vector(data)

    if isinstance(data, (list, tuple)):
        return _normalize_query_values(data)

    raise TypeError(
        "Query payload must be a JSON object, single-element list, or list of numbers."
    )


def get_query_vector_from_file(json_path: Path) -> List[float]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read()
    return get_query_vector_from_payload(raw)


def resolve_query_vector(argv: Sequence[str]) -> List[float]:
    if len(argv) >= 2:
        payload = argv[1]
        return get_query_vector_from_payload(payload)

    default_path = Path(__file__).parent / "data" / "test_city.json"
    if not default_path.exists():
        raise FileNotFoundError(
            "No query payload provided and default vector file is missing."
        )
    print(f"No query payload supplied; falling back to {default_path}.")
    return get_query_vector_from_file(default_path)


def main(argv: Sequence[str]) -> None:
    query_vector = resolve_query_vector(argv)

    # Run kNN to find similar cities
    response = es.search(
        index=index_name,
        knn={
            "field": "review_vector",
            "query_vector": query_vector,
            "k": 5,  # how many nearest neighbors
            "num_candidates": 50,  # search depth
        },
    )

    # Print results (excluding the city itself)
    print(f"Nearest neighbors to {city_to_search}:")
    for hit in response["hits"]["hits"]:
        city = hit["_source"]["city"]
        score = hit["_score"]
        if city != city_to_search:  # skip the same city
            print(f"  {city:20}  (score={score:.4f})")


if __name__ == "__main__":
    main(sys.argv)

# # Example 1: Cities with HousingCost <= 6000, sorted by HlthCare descending
# query = {
#     "query": {
#         "range": {
#             "HousingCost": {"lte": 6000}
#         }
#     },
#     "sort": [{"HlthCare": "desc"}]
# }

# results = es.search(index=index_name, body=query)

# print("Cities with HousingCost <= 6000 sorted by HlthCare:")
# for hit in results['hits']['hits']:
#     src = hit['_source']
#     print(f"{src['City']}: HousingCost={src['HousingCost']} HlthCare={src['HlthCare']}")

# # Example 2: Average HlthCare by Climate
# agg_query = {
#     "size": 0,
#     "aggs": {
#         "avg_hlthcare_by_climate": {
#             "terms": {"field": "Climate"},
#             "aggs": {"avg_hlthcare": {"avg": {"field": "HlthCare"}}}
#         }
#     }
# }

# agg_results = es.search(index=index_name, body=agg_query)

# print("\nAverage HlthCare by Climate:")
# for bucket in agg_results['aggregations']['avg_hlthcare_by_climate']['buckets']:
#     print(f"Climate {bucket['key']}: Avg HlthCare={bucket['avg_hlthcare']['value']:.2f}")
