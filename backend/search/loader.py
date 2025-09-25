from elasticsearch import Elasticsearch, helpers
from pathlib import Path
import pandas as pd
import os


# 1. Clean and Normalize Data

# Load data as a dataframe
df = pd.read_csv(Path(__file__).parent / "data" / "places.csv")

# 2. Loading data into elastic search



# # Path to CSV relative to this file (robust across CWDs)
# DATA_PATH = Path(__file__).parent / "data" / "places.csv"

# # Canonical index name
# INDEX_NAME = os.getenv("ES_INDEX", "us_cities")


# def _normalize_row(row):
#     # Map CSV columns to canonical field names expected by the backend
#     def _safe_float(x):
#         try:
#             return float(x)
#         except Exception:
#             return None

#     def _safe_int(x):
#         try:
#             return int(x)
#         except Exception:
#             return None

#     name = row.get("City")
#     # try split state if present
#     state = None
#     if isinstance(name, str) and "," in name:
#         parts = [p.strip() for p in name.split(",")]
#         if len(parts) >= 2:
#             state = parts[-1]

#     doc = {
#         "name": name,
#         "case_num": _safe_int(row.get("CaseNum")),
#         "housing_cost": _safe_float(row.get("HousingCost")),
#         "climate_score": _safe_int(row.get("Climate")),
#         "healthcare_score": _safe_float(row.get("HlthCare")),
#         "crime_score": _safe_float(row.get("Crime")),
#         "transit_score": _safe_float(row.get("Transp")),
#         "education_score": _safe_float(row.get("Educ")),
#         "arts_score": _safe_float(row.get("Arts")),
#         "recreation_score": _safe_float(row.get("Recreat")),
#         "economy_score": _safe_float(row.get("Econ")),
#         "population": _safe_int(row.get("Pop")),
#         "state": state,
#     }

#     # Geo point
#     lon = row.get("Long")
#     lat = row.get("Lat")
#     try:
#         lonf = float(lon)
#         latf = float(lat)
#         doc["location"] = {"lon": lonf, "lat": latf}
#     except Exception:
#         # skip geo if invalid
#         pass

#     # Clean None values (Elasticsearch can accept nulls but we'll drop them for cleanliness)
#     doc = {k: v for k, v in doc.items() if v is not None}
#     return doc


# def load_into_es(csv_path: Path = DATA_PATH, index_name: str = INDEX_NAME, es_url: str = None):
#     """Load the CSV into Elasticsearch using canonical field names.

#     This function is idempotent for the same index (it will index documents
#     and overwrite by `_id` if CaseNum is present and used). It does not create
#     index mappings automatically.
#     """
#     if not csv_path.exists():
#         raise FileNotFoundError(f"CSV not found at {csv_path}")

#     df = pd.read_csv(csv_path)

#     es_url = es_url or os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
#     es = Elasticsearch(es_url)

#     actions = []
#     for _, row in df.iterrows():
#         doc = _normalize_row(row)
#         if not doc:
#             continue
#         op = {
#             "_op_type": "index",
#             "_index": index_name,
#             "_source": doc,
#         }
#         # Use case_num as _id if present
#         if doc.get("case_num"):
#             op["_id"] = str(doc["case_num"])
#         actions.append(op)

#     if not actions:
#         print("No documents to index.")
#         return

#     helpers.bulk(es, actions)
#     print(f"Indexed {len(actions)} documents into index '{index_name}'")


# if __name__ == "__main__":
#     # Simple CLI to load the CSV
#     try:
#         load_into_es()
#     except Exception as e:
#         print("Failed to load CSV:", e)