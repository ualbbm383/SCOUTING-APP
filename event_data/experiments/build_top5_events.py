import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "event_data" / "scraper" / "datasets"
PROCESSED_DIR = BASE_DIR / "event_data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

db_path = PROCESSED_DIR / "events.duckdb"
out_parquet = PROCESSED_DIR / "top5_events_current.parquet"

con = duckdb.connect(str(db_path))

con.execute(f"""
    CREATE OR REPLACE TABLE top5_events AS
    SELECT *
    FROM read_parquet('{RAW_DIR.as_posix()}/*.parquet', union_by_name = true)
""")

summary = con.execute("""
    SELECT league, COUNT(*) AS n_events, COUNT(DISTINCT matchId) AS n_matches
    FROM top5_events
    GROUP BY league
    ORDER BY league
""").df()

print(summary)

con.execute(f"""
    COPY top5_events
    TO '{out_parquet.as_posix()}'
    (FORMAT PARQUET)
""")

print(f"\nGuardado: {out_parquet}")