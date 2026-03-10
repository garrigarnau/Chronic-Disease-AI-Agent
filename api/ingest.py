import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
import os
import numpy as np
from dotenv import load_dotenv


load_dotenv(path="../config/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Database Configuration
DB_CONFIG = {
    "dbname": os.getenv("MY_ENV_DB"),
    "user": os.getenv("MY_ENV_NAME"),
    "password": os.getenv("MY_ENV_PASSWORD"),
    "host": "localhost",
    "port": "5432"
}

# 2. Load and Clean the CDC Data
csv_path = "data/U.S._Chronic_Disease_Indicators.csv"
df = pd.read_csv(csv_path, low_memory=False)

# Data Cleaning: Handle non-numeric strings in numeric columns (e.g., footnotes)
numeric_cols = ['DataValue', 'LowConfidenceLimit', 'HighConfidenceLimit']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Enhanced Embedding String (The "Context" for the AI)
def create_text_for_embedding(row):
    return (
        f"Year: {row['YearStart']}, Topic: {row['Topic']}, "
        f"Indicator: {row['Question']}, Location: {row['LocationDesc']}, "
        f"Demographic: {row['Stratification1']} ({row['StratificationCategory1']}), "
        f"Result: {row['DataValue']} {row['DataValueUnit']} ({row['DataValueType']})"
    )

# 4. Database Setup
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)

cur.execute("DROP TABLE IF EXISTS chronic_disease_indicators")
cur.execute("""
    CREATE TABLE chronic_disease_indicators (
        id SERIAL PRIMARY KEY,
        year_start INT,
        location_desc TEXT,
        topic TEXT,
        question TEXT,
        data_value FLOAT,
        data_value_unit TEXT,
        data_value_type TEXT,
        stratification_category1 TEXT,
        stratification1 TEXT,
        low_confidence_limit FLOAT,
        high_confidence_limit FLOAT,
        combined_text TEXT,
        embedding vector(1536)
    )
""")
conn.commit()

# 5. Batch Ingestion Process
BATCH_SIZE = 100  # Adjust based on speed/rate limits

# Processing only a sample or the full set (remove .head() for full ingestion)
df_to_process = df.head(1000).copy() 

for i in range(0, len(df_to_process), BATCH_SIZE):
    batch = df_to_process.iloc[i:i+BATCH_SIZE]
    
    # Create the text strings for this batch
    batch_texts = batch.apply(create_text_for_embedding, axis=1).tolist()
    
    # Batch generate embeddings (More efficient than row-by-row)
    response = client.embeddings.create(
        input=batch_texts,
        model="text-embedding-3-small"
    )
    embeddings = [e.embedding for e in response.data]
    
    # Batch Insert into PostgreSQL
    for j, (_, row) in enumerate(batch.iterrows()):
        cur.execute("""
            INSERT INTO chronic_disease_indicators 
            (year_start, location_desc, topic, question, data_value, 
             data_value_unit, data_value_type, stratification_category1, 
             stratification1, low_confidence_limit, high_confidence_limit, 
             combined_text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            row['YearStart'], row['LocationDesc'], row['Topic'], row['Question'], 
            row['DataValue'], row['DataValueUnit'], row['DataValueType'],
            row['StratificationCategory1'], row['Stratification1'],
            row['LowConfidenceLimit'], row['HighConfidenceLimit'],
            batch_texts[j], embeddings[j]
        ))
    
    conn.commit()
    print(f"Processed through row {i + BATCH_SIZE}")

# 6. Final Indexing for Speed
print("Creating vector index...")
cur.execute("CREATE INDEX ON chronic_disease_indicators USING hnsw (embedding vector_cosine_ops)")
conn.commit()

cur.close()
conn.close()
print("Full Ingestion Complete!")