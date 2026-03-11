"""
plot_embeddings.py
------------------
Fetches embeddings from PostgreSQL, reduces them to 2D with UMAP,
and produces a scatter plot coloured by disease topic.

Run from the project root:
    python api/plot_embeddings.py

Optional CLI args:
    --limit   Number of rows to fetch (default: 1000)
    --output  Path to save the PNG  (default: embeddings_plot.png)
"""

import argparse
import os
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from umap import UMAP

load_dotenv(Path(__file__).resolve().parent.parent / "config" / ".env")

DB_CONFIG = {
    "dbname": os.getenv("MY_ENV_DB"),
    "user": os.getenv("MY_ENV_NAME"),
    "password": os.getenv("MY_ENV_PASSWORD"),
    "host": os.getenv("MY_ENV_HOST", "localhost"),
    "port": "5432",
}


def fetch_embeddings(limit: int):
    print(f"Connecting to database and fetching {limit} rows...")
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT topic, embedding
            FROM chronic_disease_indicators
            ORDER BY id
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    conn.close()
    print(f"Fetched {len(rows)} rows.")

    topics = [r[0] for r in rows]
    embeddings = np.array([r[1] for r in rows], dtype=np.float32)
    return topics, embeddings


def reduce_to_2d(embeddings: np.ndarray) -> np.ndarray:
    print("Reducing dimensions with UMAP (this may take ~30s for 1 000 rows)...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(embeddings)


def plot(topics, coords_2d, output_path: str):
    unique_topics = sorted(set(topics))
    palette = cm.get_cmap("tab20", len(unique_topics))
    colour_map = {t: palette(i) for i, t in enumerate(unique_topics)}

    fig, ax = plt.subplots(figsize=(14, 10))
    for topic in unique_topics:
        mask = [t == topic for t in topics]
        xs = coords_2d[mask, 0]
        ys = coords_2d[mask, 1]
        ax.scatter(xs, ys, s=12, alpha=0.6, label=topic, color=colour_map[topic])

    ax.set_title("Chronic Disease Indicator Embeddings (UMAP 2D projection)", fontsize=14)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(
        title="Topic",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=7,
        markerscale=2,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000, help="Rows to fetch")
    parser.add_argument("--output", type=str, default="embeddings_plot.png")
    args = parser.parse_args()

    topics, embeddings = fetch_embeddings(args.limit)
    coords_2d = reduce_to_2d(embeddings)
    plot(topics, coords_2d, args.output)


if __name__ == "__main__":
    main()
