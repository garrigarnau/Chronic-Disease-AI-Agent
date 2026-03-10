import logging
from langchain_core.tools import tool
from database import cur, embedding_model

logger = logging.getLogger("tools.vector_search")


@tool
def vector_search_chronic_diseases(query: str):
    """Search for chronic disease information using semantic similarity."""
    logger.info("Vector search query: '%s'", query)

    # 1. Generate embedding for the query
    query_embedding = embedding_model.embed_query(query)
    logger.info("Embedding generated (dimensions: %d)", len(query_embedding))

    # 2. Execute SQL with pgvector similarity
    cur.execute("""
        SELECT combined_text, data_value, location_desc, year_start
        FROM chronic_disease_indicators
        ORDER BY embedding <=> %s::vector
        LIMIT 5
    """, (query_embedding,))

    rows = cur.fetchall()
    logger.info("Vector search returned %d row(s)", len(rows))
    for i, row in enumerate(rows, 1):
        logger.debug("  [%d] location=%s  year=%s  value=%s", i, row.get("location_desc"), row.get("year_start"), row.get("data_value"))
    return rows