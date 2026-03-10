import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 1. Setup Clients
client = OpenAI()
conn = psycopg2.connect(
    dbname=os.getenv("MY_ENV_DB"),
    user=os.getenv("MY_ENV_NAME"),
    password=os.getenv("MY_ENV_PASSWORD"),
    host="localhost"
)

def search_docs(query, limit=5):
    # Only embed the QUERY, not the whole database again!
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding

    with conn.cursor() as cur:
        # The <=> operator calculates cosine distance (lower is more similar)
        select_query = """
            SELECT combined_text, 1 - (embedding <=> %s::vector) AS similarity
            FROM chronic_disease_indicators
            ORDER BY similarity DESC
            LIMIT %s;
        """
        cur.execute(select_query, (query_embedding, limit))
        results = cur.fetchall()
        
        return results

# --- Run a Test ---
user_input = "Alcohol consumption?"
matches = search_docs(user_input)

print(f"\nTop results for: '{user_input}'")
for text, score in matches:
    print(f"[{score:.4f}] {text[:100]}...")

conn.close()