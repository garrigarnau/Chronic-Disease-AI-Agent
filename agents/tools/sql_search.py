import logging
from langchain_core.tools import tool
from database import cur

logger = logging.getLogger("tools.sql_search")


@tool
def query_db_metadata(sql_query: str):
    """Execute a standard SQL query to get precise statistics or filtered data."""
    logger.info("Executing SQL query:\n%s", sql_query)
    cur.execute(sql_query)
    rows = cur.fetchall()
    logger.info("SQL query returned %d row(s)", len(rows))
    logger.debug("SQL results: %s", rows)
    return rows