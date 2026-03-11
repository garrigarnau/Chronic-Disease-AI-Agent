# Chronic Disease AI Agent

A conversational AI assistant that lets you query the **U.S. Chronic Disease Indicators** dataset using natural language. It combines a PostgreSQL + pgvector database with an OpenAI-powered LangGraph agent to answer both precise statistical questions and open-ended semantic searches.

---

## How It Works

```
User prompt
     │
     ▼
┌─────────────┐     tool call?     ┌──────────────────────────┐
│  GPT-4o     │ ──────────────────▶│  query_db_metadata        │  SQL → PostgreSQL
│  (LangGraph │                    │  vector_search_chronic_   │  pgvector similarity
│   agent)    │ ◀──────────────────│  diseases                 │
└─────────────┘     tool result    └──────────────────────────┘
     │
     ▼
 Final answer
```

1. The user types a question in the CLI.
2. The **LangGraph orchestrator** (`agents/orchestrator.py`) passes the conversation to GPT-4o.
3. The model decides which tool to call (or answers directly):
   - **`query_db_metadata`** — runs a SQL query for precise stats, counts, or averages.
   - **`vector_search_chronic_diseases`** — embeds the query and finds semantically similar records via pgvector.
4. Tool results are fed back to the model, which synthesises a final response.
5. The conversation history is preserved for multi-turn dialogue.

---

## Project Structure

```
.
├── api/
│   ├── app.py          # CLI entry point — runs the conversational agent loop
│   ├── ingest.py       # One-time script: loads CSV, generates embeddings, inserts into DB
│   └── search.py       # Standalone vector search utility / smoke-test
├── agents/
│   ├── orchestrator.py # LangGraph StateGraph — the agent brain
│   └── tools/
│       ├── sql_search.py     # query_db_metadata tool
│       └── vector_search.py  # vector_search_chronic_diseases tool
├── database/
│   └── connection.py   # Shared DB connection, cursor, and embedding model
├── data/
│   └── U.S._Chronic_Disease_Indicators.csv  # Source dataset (CDC)
├── config/
│   └── .env            # Secret credentials (not committed)
├── docker-compose.yml  # PostgreSQL + pgvector container
└── requirements.txt
```

---

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- An OpenAI API key

---

## Setup

### 1. Configure environment variables

Create `config/.env`:

```env
OPENAI_API_KEY=sk-...
MY_ENV_DB=chronic_db
MY_ENV_NAME=postgres
MY_ENV_PASSWORD=yourpassword
MY_ENV_HOST=localhost
```

### 2. Start the database

```bash
docker compose up -d
```

This starts a PostgreSQL 17 container with the pgvector extension pre-installed.

### 3. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Ingest the dataset

Run once to create the table, generate OpenAI embeddings for every row, insert them, and build the vector index:

```bash
python api/ingest.py
```

> This may take a few minutes depending on dataset size, as it calls the OpenAI Embeddings API for each record.

The ingestion pipeline does the following in order:

1. **Creates the table** — `chronic_disease_indicators` with a `vector(1536)` column for embeddings.
2. **Generates embeddings** — rows are processed in batches of 100 using `text-embedding-3-small` (1 536-dimension vectors).
3. **Inserts rows** — each batch is committed to PostgreSQL incrementally.
4. **Builds the HNSW index** — after all rows are inserted, an approximate nearest-neighbour index is created:

```sql
CREATE INDEX ON chronic_disease_indicators
USING hnsw (embedding vector_cosine_ops);
```

**Why HNSW?** The Hierarchical Navigable Small World index trades a tiny amount of recall for dramatically faster similarity searches at query time. Without it, every vector search would do a full table scan (`O(n)`). With it, queries are sub-linear (`O(log n)`), which matters as the dataset grows.

**`vector_cosine_ops`** tells pgvector to optimise for cosine distance (`<=>`), which is what the `vector_search_chronic_diseases` tool uses.

### 5. Run the agent

```bash
python api/app.py
```

---

## Example Queries

```
User: What is the average diabetes prevalence in California?
User: Which states have the highest cardiovascular disease rates?
User: Tell me about alcohol-related chronic disease indicators.
User: Compare obesity rates between men and women in 2021.
```

---

## Key Technologies

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o |
| Agent framework | LangGraph |
| Vector search | pgvector (PostgreSQL extension) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Database driver | psycopg2 |
| Infrastructure | Docker + pgvector/pgvector:pg17 |


## 6. Vector space 2d image
<img width="1440" height="739" alt="vectospace" src="https://github.com/user-attachments/assets/280cc9bc-89d3-4b81-8921-036733927358" />

