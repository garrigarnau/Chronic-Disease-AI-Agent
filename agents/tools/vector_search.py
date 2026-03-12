import json
import logging
import pandas as pd
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from database import cur, embedding_model

logger = logging.getLogger("tools.vector_search")

_spec_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_SPEC_PROMPT = """
You are a data analyst. Given a user question and a sample of vector-search results,
decide how to best aggregate the rows into a compact summary table.

Return ONLY valid JSON in this exact format (no extra text):
{{
  "groupby": [<list of column names to group by, or [] for no grouping>],
  "agg_col": "<numeric column to aggregate>",
  "agg_funcs": [<list of: "mean", "min", "max", "count", "sum">]
}}

CRITICAL RULES:
- ALWAYS include 'data_value_unit' in groupby if it is present in the columns.
  Mixing units like '%', 'Number', 'cases per 100,000' in the same aggregation
  produces meaningless results.
- ALWAYS include 'data_value_type' in groupby if it is present in the columns.
  Mixing 'Crude Prevalence', 'Age-adjusted Rate', 'Number' is invalid.

User question: {query}
Available columns: {columns}
Sample rows (first 3): {sample}
"""


def _get_agg_spec(query: str, df: pd.DataFrame) -> dict:
    prompt = _SPEC_PROMPT.format(
        query=query,
        columns=list(df.columns),
        sample=df.head(3).to_dict(orient="records"),
    )
    response = _spec_llm.invoke([HumanMessage(content=prompt)])
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse aggregation spec, returning raw sample")
        return None


def _aggregate(df: pd.DataFrame, spec: dict) -> str:
    agg_col = spec.get("agg_col")
    agg_funcs = spec.get("agg_funcs", ["mean", "count"])
    groupby_cols = spec.get("groupby", [])

    if not agg_col or agg_col not in df.columns:
        return df.head(50).to_string(index=False)

    valid_funcs = [f for f in agg_funcs if f in ("mean", "min", "max", "count", "sum")]

    # Always group by unit/type columns if present to avoid mixing incompatible values
    for mandatory in ("data_value_unit", "data_value_type"):
        if mandatory in df.columns and mandatory not in groupby_cols:
            groupby_cols.append(mandatory)

    if groupby_cols and all(c in df.columns for c in groupby_cols):
        result = df.groupby(groupby_cols)[agg_col].agg(valid_funcs).reset_index()
    else:
        result = df[agg_col].agg(valid_funcs).to_frame().T

    return result.to_string(index=False)


@tool
def vector_search_chronic_diseases(query: str, year: int = None, location: str = None):
    """Translate a vague concept into exact topic/question names that exist in the
    database. Always call this first when the user mentions a disease category or
    symptom (e.g. 'respiratory', 'heart disease').

    Optional filters:
    - year: restrict results to a specific year (e.g. 2021)
    - location: restrict results to a state or region (e.g. 'Colorado')

    The returned topic and question names can then be used in a follow-up
    query_db_metadata SQL call to retrieve precise aggregated data."""
    logger.info("Vector search query: '%s'  year=%s  location=%s", query, year, location)

    query_embedding = embedding_model.embed_query(query)
    logger.info("Embedding generated (dimensions: %d)", len(query_embedding))

    cur.execute("""
        SELECT topic, question, location_desc, year_start,
               data_value, data_value_unit, data_value_type,
               stratification_category1, stratification1,
               combined_text
        FROM chronic_disease_indicators
        WHERE (%(year)s IS NULL OR year_start = %(year)s)
          AND (%(location)s IS NULL OR location_desc ILIKE %(location)s)
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT 50
    """, {"year": year, "location": location, "vec": query_embedding})

    rows = cur.fetchall()
    logger.info("Vector search returned %d row(s)", len(rows))

    if not rows:
        return "No results found."

    df = pd.DataFrame(rows)
    logger.debug("Columns: %s", list(df.columns))

    # Always surface the distinct concept names so the agent can build SQL with them
    discovered_topics = sorted(df["topic"].dropna().unique().tolist()) if "topic" in df.columns else []
    discovered_questions = sorted(df["question"].dropna().unique().tolist()) if "question" in df.columns else []
    logger.info("Discovered topics: %s", discovered_topics)
    logger.info("Discovered questions: %s", discovered_questions)

    spec = _get_agg_spec(query, df)
    logger.info("Aggregation spec: %s", spec)
    aggregated = _aggregate(df, spec) if spec else df.drop(columns=["combined_text"], errors="ignore").head(20).to_string(index=False)
    logger.info("Aggregated result:\n%s", aggregated)

    return (
        f"DISCOVERED TOPICS (use these exact strings in SQL):\n{discovered_topics}\n\n"
        f"DISCOVERED QUESTIONS (use these exact strings in SQL):\n{discovered_questions}\n\n"
        f"AGGREGATED PREVIEW:\n{aggregated}"
    )