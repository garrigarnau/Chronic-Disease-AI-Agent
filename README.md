# Chronic Disease AI Agent

A conversational AI assistant that lets you query the **U.S. Chronic Disease Indicators** dataset using natural language. It combines a PostgreSQL + pgvector database with an OpenAI-powered LangGraph agent to answer both precise statistical questions and open-ended semantic searches.

---

## How It Works

The agent uses a **two-step pattern** separating semantic concept discovery from precise structured retrieval. Steps are chained or skipped depending on the question.

```
User question
      │
      ▼
┌─────────────────────────────────────────┐
│  GPT-4o  (LangGraph orchestrator)       │
│                                         │
│  Is the concept vague / not a column    │
│  value?  (e.g. "respiratory issues")    │
└──────────┬──────────────────────────────┘
           │
     ┌─────┴──────┐
     YES          NO (exact topic known)
     │             │
     ▼             │
┌──────────────┐   │
│  STEP 1      │   │
│  vector_     │   │
│  search      │   │
│              │   │
│  Embeds the  │   │
│  query with  │   │
│  optional    │   │
│  year/loc    │   │
│  filters     │   │
│              │   │
│  pgvector    │   │
│  top-50 rows │   │
│  by cosine   │   │
│  similarity  │   │
│              │   │
│  → Discovers │   │
│    exact DB  │   │
│    topic &   │   │
│    question  │   │
│    names     │   │
└──────┬───────┘   │
       │           │
       │  Does the question have     
       │  structured filters?        
       │  (year / location /         
       │   demographic)              
       │                             
  ┌────┴──────┐                      
  YES         NO                     
  │           │                      
  │           ▼                      
  │    ┌────────────────┐            
  │    │  Return        │            
  │    │  discovered    │            
  │    │  topics as     │            ◀── final tool result
  │    │  answer        │
  │    └────────────────┘
  │
  ▼
┌──────────────────────────────────┐
│  STEP 2  (always when filters    │
│  exist, or question is precise)  │
│  query_db_metadata               │
│                                  │
│  SQL with no LIMIT:              │
│  WHERE topic IN (discovered)     │◀── topic names from Step 1,
│    AND year_start = ?            │    or known directly
│    AND location_desc = ?         │
│    AND stratification = ?        │
│                                  │
│  → All matching rows loaded      │
│    into pandas DataFrame         │
└────────────────┬─────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  gpt-4o-mini    │
        │                 │
        │  Sees:          │
        │  question +     │
        │  column names + │
        │  3 sample rows  │
        │                 │
        │  → JSON spec:   │
        │  { groupby,     │
        │    agg_col,     │
        │    agg_funcs }  │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  pandas         │
        │  .groupby()     │
        │  .agg()         │
        │                 │
        │  Always splits  │
        │  by:            │
        │  data_value_    │
        │  unit +         │
        │  data_value_    │
        │  type           │
        │                 │
        │  → compact      │
        │  table          │
        │  (10–50 rows)   │
        └────────┬────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  GPT-4o reads compact table             │
│  → writes final natural language answer │
└─────────────────────────────────────────┘
```

**When each tool is used:**

| Question type | Tools called | Why |
|--------------|--------------|-----|
| Vague concept, no filters | `vector_search` only | Concept discovery is enough |
| Vague concept + year/location | `vector_search` → `query_db_metadata` | Discover names, then filter precisely |
| Precise topic + filters | `query_db_metadata` only | Topic already known, SQL is sufficient |

**Why two steps?** The database stores exact names like `"Asthma"` and `"Chronic Obstructive Pulmonary Disease"`, not natural language terms like `"respiratory issues"`. Vector search bridges this gap — it translates the user's language into the real column values that SQL can filter on.

**Raw data never enters the context window.** The pandas aggregation step collapses potentially thousands of SQL rows into a compact table before GPT-4o sees the results.

---

## Project Structure

```
.
├── api/
│   ├── app.py              # CLI entry point — runs the conversational agent loop
│   ├── ingest.py           # One-time script: loads CSV, generates embeddings, inserts into DB
│   ├── plot_embeddings.py  # Generates a 2D UMAP scatter plot of the embedding space
│   └── search.py           # Standalone vector search utility / smoke-test
├── agents/
│   ├── __init__.py         # Re-exports `app` from orchestrator
│   ├── orchestrator.py     # LangGraph StateGraph — the agent brain
│   └── tools/
│       ├── __init__.py          # Re-exports both tool functions
│       ├── sql_search.py        # query_db_metadata tool (SQL + pandas aggregation)
│       └── vector_search.py     # vector_search_chronic_diseases tool (pgvector + pandas)
├── database/
│   ├── __init__.py         # Re-exports conn, cur, embedding_model
│   └── connection.py       # Shared DB connection, cursor, and embedding model
├── deployment/
│   └── docker-compose.yml  # PostgreSQL + pgvector container
├── data/
│   └── U.S._Chronic_Disease_Indicators.csv  # Source dataset (CDC)
├── config/
│   └── .env                # Secret credentials (not committed)
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
User: What data do we have on respiratory health issues in Colorado 2021?
User: What is the average diabetes prevalence in California?
User: Which states have the highest cardiovascular disease rates?
User: Tell me about alcohol-related chronic disease indicators.
User: Compare obesity rates between men and women in 2021.
```

---

## Visualising the Embedding Space

Generate a 2D UMAP projection of the database embeddings coloured by disease topic:

```bash
pip install umap-learn matplotlib
python api/plot_embeddings.py              # default: 1 000 rows
python api/plot_embeddings.py --limit 3000 --output my_plot.png
```

---

## Key Technologies

| Component | Technology |
|-----------|-----------|
| LLM (reasoning) | OpenAI GPT-4o |
| LLM (aggregation spec) | OpenAI GPT-4o-mini |
| Agent framework | LangGraph |
| Vector search | pgvector (PostgreSQL extension) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Data aggregation | pandas |
| Dimensionality reduction | UMAP |
| Database driver | psycopg2 |
| Infrastructure | Docker + pgvector/pgvector:pg17 |


## 6. Vector space 2d image
<img width="1440" height="739" alt="vectospace" src="https://github.com/user-attachments/assets/280cc9bc-89d3-4b81-8921-036733927358" />

