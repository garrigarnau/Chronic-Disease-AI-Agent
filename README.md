# Chronic Disease AI Agent

A conversational AI assistant that lets you query the **U.S. Chronic Disease Indicators** dataset using natural language. It combines a PostgreSQL + pgvector database with an OpenAI-powered LangGraph agent to answer both precise statistical questions and open-ended semantic searches.

---

## How It Works

The agent follows a **two-step pattern** that separates semantic concept discovery from precise structured retrieval:

```
User prompt
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  GPT-4o  (LangGraph orchestrator)                            │
│  Decides which tool(s) to call based on the question         │
└───────────────┬──────────────────────────────────────────────┘
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
  Step 1 (optional)   Step 2 (always when filters exist)
  vector_search       query_db_metadata
  ─────────────       ─────────────────
  Embeds query        Runs exact SQL with
  → top-50 similar    WHERE filters (year,
    rows (scoped by   location, topic names
    year/location     discovered in Step 1)
    if provided)           │
        │                  │
        ▼                  ▼
  Discovered          Raw rows → pd.DataFrame
  topic & question    │
  names               ▼
        │         gpt-4o-mini decides groupby spec
        │         (JSON: groupby, agg_col, agg_funcs)
        │             │
        │             ▼
        │         pandas .groupby().agg()
        │         always groups by data_value_unit
        │         and data_value_type to avoid
        │         mixing incompatible metrics
        │             │
        └──────────┬──┘
                   ▼
     Compact aggregated table (~10–50 rows)
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  GPT-4o writes final natural language answer                 │
└──────────────────────────────────────────────────────────────┘
```

**When each tool is used:**

| Question type | Tool(s) called |
|--------------|---------------|
| Vague concept, no filters | `vector_search` only |
| Vague concept + year/location | `vector_search` → `query_db_metadata` |
| Precise topic/column + filters | `query_db_metadata` only |

**Why two steps?** The database stores exact disease names like `"Asthma"` and `"Chronic Obstructive Pulmonary Disease"`, not natural language terms like `"respiratory issues"`. Vector search bridges this gap — it finds the real column values, which SQL then uses for precise filtering.

**Raw data never enters the context window.** The pandas aggregation step collapses potentially thousands of SQL rows into a compact table before GPT-4o sees the results.

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

