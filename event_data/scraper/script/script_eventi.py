"""
script_eventi.py
----------------
Parses WhoScored match HTML files and appends event data to a Parquet file.

Now parameterised by:
- output parquet path
- league
- season

Usage (standalone example):
    python script/script_eventi.py
"""

import json
import re
import glob
import os
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PARTITE_FOLDER = BASE_DIR / "partite"

# ── Regex for embedded WhoScored JSON ─────────────────────────────────────────
_ARGS_REGEX = r'(?<=require\.config\.params\["args"\].=.)[\s\S]*?;'

_JSON_KEYS = [
    "matchId",
    "matchCentreData",
    "matchCentreEventTypeJson",
    "formationIdNameMappings",
]


# ── HTML / JSON extraction ────────────────────────────────────────────────────

def extract_json_from_html(html_path: str | Path) -> str:
    with open(html_path, "r", encoding="utf-8") as fh:
        html = fh.read()

    matches = re.findall(_ARGS_REGEX, html)
    if not matches:
        raise ValueError(f"matchCentreData pattern not found in {html_path}")

    data_txt = matches[0]
    for key in _JSON_KEYS:
        data_txt = data_txt.replace(key, f'"{key}"')
    data_txt = data_txt.replace("};", "}")

    return data_txt


def extract_data_from_dict(data: dict) -> tuple:
    match_id = data["matchId"]
    match_date = data["matchCentreData"]["startDate"][:10]

    events_list = data["matchCentreData"]["events"]
    teams_dict = {
        data["matchCentreData"]["home"]["teamId"]: data["matchCentreData"]["home"]["name"],
        data["matchCentreData"]["away"]["teamId"]: data["matchCentreData"]["away"]["name"],
    }
    players_ids = data["matchCentreData"]["playerIdNameDictionary"]

    return events_list, match_id, match_date, players_ids, teams_dict


# ── Event parsing ─────────────────────────────────────────────────────────────

def extract_event_data(event: dict) -> dict:
    event_data = {
        "Player ID": event.get("playerId"),
        "Event Type": event["type"]["displayName"],
        "Event Value": event["type"]["value"],
        "Outcome": event["outcomeType"]["displayName"],
        "Minuto": event["minute"],
        "Secondo": event.get("second"),
        "Team ID": event["teamId"],
        "Start X": event["x"],
        "Start Y": event["y"],
        "End X": event.get("endX"),
        "End Y": event.get("endY"),
    }

    qualifier_columns = {}
    for qualifier in event.get("qualifiers", []):
        name = qualifier["type"]["displayName"]
        value = qualifier.get("value")

        if value is None:
            value = "Yes"

        if name in qualifier_columns:
            i = 1
            while f"{name}_{i}" in qualifier_columns:
                i += 1
            name = f"{name}_{i}"

        qualifier_columns[name] = value

    event_data.update(qualifier_columns)
    return event_data


def create_events_dataframe(path: str | Path, league: str, season: str) -> pd.DataFrame:
    json_str = extract_json_from_html(path)
    data = json.loads(json_str)
    events_list, match_id, match_date, players_ids, teams_dict = extract_data_from_dict(data)

    rows = [extract_event_data(event) for event in events_list]
    df = pd.DataFrame(rows)

    # match-level columns
    df.insert(0, "matchId", match_id)
    df.insert(1, "match_date", pd.to_datetime(match_date))
    df.insert(2, "league", league)
    df.insert(3, "season", season)

    # player names
    def _resolve_player(pid):
        if pd.isna(pid):
            return None
        return players_ids.get(str(int(pid)))

    pid_pos = df.columns.get_loc("Player ID")
    df.insert(pid_pos + 1, "player_name", df["Player ID"].map(_resolve_player))

    # team names
    tid_pos = df.columns.get_loc("Team ID")
    df.insert(tid_pos + 1, "team_name", df["Team ID"].map(teams_dict))

    # ensure all qualifiers from match exist as columns
    all_qualifiers = set()
    for event in events_list:
        for q in event.get("qualifiers", []):
            all_qualifiers.add(q["type"]["displayName"])

    for qualifier in all_qualifiers:
        if qualifier not in df.columns:
            df[qualifier] = None

    return df


# ── Parquet helpers ───────────────────────────────────────────────────────────

def load_processed_ids(parquet_path: str | Path) -> set:
    path = Path(parquet_path)
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["matchId"])
    return set(df["matchId"].unique())


# ── Main processing function ──────────────────────────────────────────────────

def process_and_save(folder_path: str | Path, output_parquet: str | Path, league: str, season: str) -> None:
    folder_path = Path(folder_path)
    output_parquet = Path(output_parquet)

    html_files = glob.glob(str(folder_path / "**" / "*.html"), recursive=True)

    # deduplicate by basename
    seen_names = set()
    unique_files = []
    for fp in html_files:
        basename = os.path.basename(fp)
        if basename not in seen_names:
            seen_names.add(basename)
            unique_files.append(fp)

    if not unique_files:
        print(f"Nessun file HTML trovato in {folder_path}")
        return

    processed_ids = load_processed_ids(output_parquet)
    print(f"Match già nel Parquet: {len(processed_ids)}")

    new_dataframes = []
    skipped = 0
    errors = 0

    for file_path in unique_files:
        try:
            json_str = extract_json_from_html(file_path)
            data = json.loads(json_str)
            match_id = data["matchId"]

            if match_id in processed_ids:
                skipped += 1
                continue

            print(f"Elaborazione: {os.path.basename(file_path)}")
            df = create_events_dataframe(file_path, league=league, season=season)
            new_dataframes.append(df)
            print(f"  ✓ {len(df):,} eventi trovati (matchId={match_id})")

        except Exception as exc:
            errors += 1
            print(f"  ✗ Errore in {os.path.basename(file_path)}: {exc}")

    print(f"\n⏭️  File già presenti, saltati : {skipped}")
    if errors:
        print(f"⚠️  File con errori           : {errors}")

    if not new_dataframes:
        print("Nessun nuovo match da aggiungere.")
        return

    new_df = pd.concat(new_dataframes, ignore_index=True)

    if output_parquet.exists():
        existing_df = pd.read_parquet(output_parquet)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        final_df = new_df

    final_df.to_parquet(output_parquet, index=False)

    print(f"\n✅ Parquet aggiornato: {len(final_df):,} eventi totali ({len(new_df):,} nuovi)")
    print(f"\n=== Riepilogo ===")
    print(f"Nuovi match aggiunti : {len(new_dataframes)}")
    print(f"Nuovi eventi         : {len(new_df):,}")
    print(f"Totale eventi        : {len(final_df):,}")


if __name__ == "__main__":
    # ejemplo standalone
    default_out = BASE_DIR / "datasets" / "serie_a_2025_2026.parquet"
    process_and_save(
        PARTITE_FOLDER,
        output_parquet=default_out,
        league="Serie A",
        season="2025_2026"
    )