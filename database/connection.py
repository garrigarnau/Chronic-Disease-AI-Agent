# database.py
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("database")

load_dotenv(Path(__file__).resolve().parent.parent / "config" / ".env")

# Initialize the embedding model once
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
logger.info("Embedding model initialised (text-embedding-3-small)")

# Initialize the connection once
_db_params = {
    "dbname": os.getenv("MY_ENV_DB"),
    "user": os.getenv("MY_ENV_NAME"),
    "password": os.getenv("MY_ENV_PASSWORD"),
    "host": os.getenv("MY_ENV_HOST", "localhost"),
}
logger.info(
    "Connecting to PostgreSQL — host=%s  db=%s  user=%s",
    _db_params["host"],
    _db_params["dbname"],
    _db_params["user"],
)
conn = psycopg2.connect(**_db_params)
cur = conn.cursor(cursor_factory=RealDictCursor)
logger.info("Database connection established successfully")