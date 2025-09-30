from elasticsearch import Elasticsearch, helpers
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from dotenv import load_dotenv
import os

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
z_score_scaler = StandardScaler()

# Apply Min-Max Scaling to numerical columns
min_max_df = df.copy()
min_max_df[numerical_cols] = min_max_scaler.fit_transform(df[numerical_cols])

# Apply z-score normalization to numerical columns
z_score_df = df.copy()
z_score_df[numerical_cols] = z_score_scaler.fit_transform(df[numerical_cols])


print(min_max_df.head(10))
print(z_score_df.head(10))

min_max_df.to_csv(Path(__file__).parent / "data" / "places_min_max.csv", index=False)
z_score_df.to_csv(Path(__file__).parent / "data" / "places_z_score.csv", index=False)


# 2. Loading data into elastic search
#df = pd.read_csv("data/places.csv")
index_name = "us_cities"
es = Elasticsearch('http://localhost:9200', basic_auth=('elastic', os.getenv("ES_LOCAL_PASSWORD")), verify_certs=False)

# prepare bulk actions
actions = [
    {
        "_index": index_name,
        "_source": row.to_dict()
    }
    for _, row in z_score_df.iterrows()
]

# bulk insert
helpers.bulk(es, actions)

print(f"Inserted {len(actions)} records into index '{index_name}'.")

