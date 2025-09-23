from elasticsearch import Elasticsearch
client = Elasticsearch(
  "http://localhost:9200",
  api_key="YOUR_API_KEY"
)
client.indices.create(
  index="search-qlr2",
  mappings={
        "properties": {
            "text": {"type": "semantic_text"}
        }
    }
)