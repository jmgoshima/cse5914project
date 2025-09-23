from elasticsearch import Elasticsearch, helpers
import os
import json
import re

es = Elasticsearch("http://localhost:9200")
index_name = "us_cities"

# Example 1: Cities with HousingCost <= 6000, sorted by HlthCare descending
query = {
    "query": {
        "range": {
            "HousingCost": {"lte": 6000}
        }
    },
    "sort": [{"HlthCare": "desc"}]
}

results = es.search(index=index_name, body=query)

print("Cities with HousingCost <= 6000 sorted by HlthCare:")
for hit in results['hits']['hits']:
    src = hit['_source']
    print(f"{src['City']}: HousingCost={src['HousingCost']} HlthCare={src['HlthCare']}")

# Example 2: Average HlthCare by Climate
agg_query = {
    "size": 0,
    "aggs": {
        "avg_hlthcare_by_climate": {
            "terms": {"field": "Climate"},
            "aggs": {"avg_hlthcare": {"avg": {"field": "HlthCare"}}}
        }
    }
}

agg_results = es.search(index=index_name, body=agg_query)

print("\nAverage HlthCare by Climate:")
for bucket in agg_results['aggregations']['avg_hlthcare_by_climate']['buckets']:
    print(f"Climate {bucket['key']}: Avg HlthCare={bucket['avg_hlthcare']['value']:.2f}")