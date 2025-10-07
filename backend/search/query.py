from elasticsearch import Elasticsearch, helpers
import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv

env_path_relative = Path(__file__).parent.parent.parent / "elastic-start-local" / ".env"
load_dotenv(dotenv_path=env_path_relative)

es = Elasticsearch('http://localhost:9200', basic_auth=('elastic', os.getenv("ES_LOCAL_PASSWORD")), verify_certs=False)
index_name = "us_cities"

city_to_search = "Columbus,OH"

# Fetch the city’s vector from ES
city_doc = es.search(
    index=index_name,
    query={"term": {"city": city_to_search}},
    size=1
)

if not city_doc["hits"]["hits"]:
    raise ValueError(f"City '{city_to_search}' not found in index '{index_name}'.")

query_vector = city_doc["hits"]["hits"][0]["_source"]["review_vector"]

# Run kNN to find similar cities
response = es.search(
    index=index_name,
    knn={
        "field": "review_vector",
        "query_vector": query_vector,
        "k": 5,               # how many nearest neighbors
        "num_candidates": 50  # search depth
    }
)

# Print results (excluding the city itself)
print(f"Nearest neighbors to {city_to_search}:")
for hit in response["hits"]["hits"]:
    city = hit["_source"]["city"]
    score = hit["_score"]
    if city != city_to_search:   # skip the same city
        print(f"  {city:20}  (score={score:.4f})")

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