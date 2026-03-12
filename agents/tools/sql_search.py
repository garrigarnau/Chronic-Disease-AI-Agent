import json
import logging
import pandas as pd
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from database import cur

logger = logging.getLogger("tools.sql_search")

_spec_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_SPEC_PROMPT = """
You are a data analyst. Given a SQL query and a sample of its results, decide
how to best aggregate the rows into a compact summary table.

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

SQL query:
{sql_query}

Available columns: {columns}
Sample rows (first 3): {sample}
"""


def _get_agg_spec(sql_query: str, df: pd.DataFrame) -> dict:
    prompt = _SPEC_PROMPT.format(
        sql_query=sql_query,
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
        return df.head(20).to_string(index=False)

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
def query_db_metadata(sql_query: str):
    """Execute a standard SQL query to get precise statistics or filtered data."""
    logger.info("Executing SQL query:\n%s", sql_query)
    cur.execute(sql_query)
    rows = cur.fetchall()
    logger.info("SQL query returned %d row(s)", len(rows))

    if not rows:
        return "No results found."

    df = pd.DataFrame(rows)
    logger.debug("Columns: %s", list(df.columns))

    spec = _get_agg_spec(sql_query, df)
    logger.info("Aggregation spec: %s", spec)

    if spec is None:
        return df.head(20).to_string(index=False)

    summary = _aggregate(df, spec)
    logger.info("Aggregated result:\n%s", summary)
    return summary