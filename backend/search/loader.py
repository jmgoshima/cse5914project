from elasticsearch import helpers
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from dotenv import load_dotenv
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.search.es_client import get_client  # type: ignore

# 0. load .env variables from elastic-start-local

env_path_relative = Path(__file__).parent.parent.parent / "elastic-start-local" / ".env"
load_dotenv(dotenv_path=env_path_relative)

# 1. Clean and Normalize Data

# Load data as a dataframe
df = pd.read_csv(Path(__file__).parent / "data" / "places.csv")

# Remove CaseNum, Long, Lat, StNum
df = df.drop(['CaseNum', 
              'Long',
              'Lat',
              'StNum'],
              axis = 1)

df.hist()

# Rescale values to a range from low_num to high_num
low_num = 0
high_num = 10

# Select only numerical columns for scaling
numerical_cols = df.select_dtypes(include=['number']).columns

# Initialize Scalers
min_max_scaler = MinMaxScaler((low_num, high_num))
# z_score_scaler = StandardScaler()

# Apply Min-Max Scaling to numerical columns
min_max_df = df.copy()
min_max_df[numerical_cols] = min_max_scaler.fit_transform(df[numerical_cols])

# Apply z-score normalization to numerical columns
# z_score_df = df.copy()
# z_score_df[numerical_cols] = z_score_scaler.fit_transform(df[numerical_cols])


print(min_max_df.head(10))
# print(z_score_df.head(10))

min_max_df.to_csv(Path(__file__).parent / "data" / "places_min_max.csv", index=False)
# z_score_df.to_csv(Path(__file__).parent / "data" / "places_z_score.csv", index=False)


# 2. Loading data into elastic search
index_name = "cities"
es = get_client()

# Define mapping for the index
dims = len(min_max_df.columns) - 1  # exclude "City" column
mapping = {
    "mappings": {
        "properties": {
            "city": {"type": "keyword"},
            "review_vector": {"type": "dense_vector", "dims": dims}
        }
    }
}

# delete existing index if it exists
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

es.indices.create(index=index_name, body=mapping)

# Prepare bulk actions
actions = []
for _, row in min_max_df.iterrows():
    row_dict = row.to_dict()
    city_name = row_dict.pop("City")  # remove "City" from vector fields
    vector_values = list(row_dict.values())  # remaining numeric values

    doc = {
        "_index": index_name,
        "_source": {
            "city": city_name,
            "review_vector": vector_values
        }
    }
    actions.append(doc)

# bulk insert
helpers.bulk(es, actions)

print(f"Inserted {len(actions)} records into index '{index_name}'.")

