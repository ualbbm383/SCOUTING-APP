"""
whoscored_downloader.py
-----------------------
Playwright-based scraper for WhoScored match pages.

Modo manual recomendado:
- abre el calendario de la liga
- te deja unos segundos para moverte manualmente a la semana deseada
- scrapea los partidos visibles en esa semana
- actualiza el parquet de esa liga

Usage:
    python script/whoscored_downloader.py --league serie_a
"""

import re
import json
import time
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright, Page

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from script_eventi import load_processed_ids, process_and_save

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PARTITE_FOLDER = BASE_DIR / "partite"
DATASETS_FOLDER = BASE_DIR / "datasets"
DATASETS_FOLDER.mkdir(parents=True, exist_ok=True)

# ── League config ─────────────────────────────────────────────────────────────
LEAGUES = {
    "serie_a": {
        "calendar_url": "https://it.whoscored.com/regions/108/tournaments/5/italia-serie-a",
        "league_name": "Serie A",
        "season": "2025_2026",
        "output_parquet": DATASETS_FOLDER / "serie_a_2025_2026.parquet",
    },
    "premier_league": {
        "calendar_url": "https://www.whoscored.com/regions/252/tournaments/2/england-premier-league",
        "league_name": "Premier League",
        "season": "2025_2026",
        "output_parquet": DATASETS_FOLDER / "premier_league_2025_2026.parquet",
    },
    "laliga": {
        "calendar_url": "https://www.whoscored.com/regions/206/tournaments/4/spain-la-liga",
        "league_name": "LaLiga",
        "season": "2025_2026",
        "output_parquet": DATASETS_FOLDER / "laliga_2025_2026.parquet",
    },
    "bundesliga": {
        "calendar_url": "https://www.whoscored.com/regions/81/tournaments/3/germany-bundesliga",
        "league_name": "Bundesliga",
        "season": "2025_2026",
        "output_parquet": DATASETS_FOLDER / "bundesliga_2025_2026.parquet",
    },
    "ligue1": {
        "calendar_url": "https://www.whoscored.com/regions/74/tournaments/22/france-ligue-1",
        "league_name": "Ligue 1",
        "season": "2025_2026",
        "output_parquet": DATASETS_FOLDER / "ligue1_2025_2026.parquet",
    },
}

# ── Regex for embedded JSON blob ──────────────────────────────────────────────
_ARGS_REGEX = r'(?<=require\.config\.params\["args"\].=.)[\s\S]*?;'
_JSON_KEYS = ["matchId", "matchCentreData", "matchCentreEventTypeJson", "formationIdNameMappings"]

# Delay between requests to match pages
REQUEST_DELAY = 1.5


def safe_filename(name: str, max_len: int = 180) -> str:
    name = re.sub(r'[\\/:"*?<>|]+', " - ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:max_len]


def try_accept_cookies(page: Page) -> None:
    selectors = [
        "button:has-text('Accetta tutto')",
        "button:has-text('Accetta')",
        "button:has-text('Accept all')",
        "button:has-text('Accept')",
        "button[id*='accept']",
        "button[class*='accept']",
        "button[id*='cookie']",
        "button[class*='cookie']",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible():
                loc.click(timeout=1500)
                page.wait_for_timeout(800)
                return
        except Exception:
            pass


def get_fixtures_json(page: Page) -> dict | None:
    el = page.query_selector(
        "script[data-hypernova-key='tournamentfixtures'][type='application/json']"
    )
    if not el:
        return None

    txt = (el.text_content() or "").strip()
    if txt.startswith("<!--"):
        txt = txt[4:]
    if txt.endswith("-->"):
        txt = txt[:-3]
    txt = txt.strip()

    try:
        return json.loads(txt)
    except Exception:
        return None


def extract_match_links(page: Page) -> list[str]:
    anchors = page.query_selector_all("a[id^='scoresBtn-'][href*='/matches/']")
    urls = []
    for a in anchors:
        href = a.get_attribute("href") or ""
        if not href:
            continue
        url = f"https://it.whoscored.com{href}" if href.startswith("/") else href
        if url not in urls:
            urls.append(url)
    return urls


def parse_args_from_html(html: str) -> dict | None:
    found = re.findall(_ARGS_REGEX, html)
    if not found:
        return None

    data_txt = found[0]
    for key in _JSON_KEYS:
        data_txt = data_txt.replace(key, f'"{key}"')
    data_txt = data_txt.replace("};", "}")

    try:
        return json.loads(data_txt)
    except Exception:
        return None


def extract_match_info(html: str) -> tuple[str, str, str] | None:
    data = parse_args_from_html(html)
    if not data:
        return None
    try:
        home = data["matchCentreData"]["home"]["name"]
        away = data["matchCentreData"]["away"]["name"]
        match_id = str(data["matchId"])
        return home, away, match_id
    except Exception:
        return None


def save_html_to_inbox(html: str, match_id: str, page_title: str, inbox_folder: Path) -> int:
    filename = f"{match_id} - {safe_filename(page_title)}.html"
    dest = inbox_folder / filename

    if dest.exists():
        print(f"   ⏭️  Già presente: {dest.name}")
        return 0

    dest.write_text(html, encoding="utf-8")
    print(f"   ✅ Salvato: {dest}")
    return 1


def run(league_key: str, manual_wait_seconds: int = 10) -> None:
    config = LEAGUES[league_key]
    calendar_url = config["calendar_url"]
    league_name = config["league_name"]
    season = config["season"]
    output_parquet = config["output_parquet"]

    league_folder = PARTITE_FOLDER / league_key
    inbox_folder = league_folder / "_inbox"
    inbox_folder.mkdir(parents=True, exist_ok=True)

    processed_ids = load_processed_ids(output_parquet)
    print(f"📋 Match già nel Parquet ({league_name}): {len(processed_ids)}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        calendar_page = context.new_page()
        match_page = context.new_page()

        print(f"\n🌐 Apertura calendario: {calendar_url}")
        calendar_page.goto(calendar_url, wait_until="domcontentloaded", timeout=60_000)
        calendar_page.wait_for_timeout(2500)
        try_accept_cookies(calendar_page)
        calendar_page.wait_for_timeout(1200)

        calendar_page.wait_for_selector(
            "script[data-hypernova-key='tournamentfixtures'][type='application/json']",
            state="attached",
            timeout=60_000,
        )

        print(f"\n⏸️ Tienes {manual_wait_seconds} segundos para colocarte manualmente en la semana que quieras...")
        print("   Usa el navegador abierto para ir a la semana deseada.")
        calendar_page.wait_for_timeout(manual_wait_seconds * 1000)

        fixtures = get_fixtures_json(calendar_page)
        if fixtures:
            fixture_date = fixtures.get("fixtureDate")
            print(f"📅 fixtureDate visible: {fixture_date}")

        match_links = extract_match_links(calendar_page)
        print(f"🔗 Partidos encontrados en la semana visible: {len(match_links)}")

        total_saved = 0
        total_opened = 0

        for url in match_links:
            try:
                print(f"\n📥 Apertura: {url}")
                match_page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                total_opened += 1

                for _ in range(20):
                    if "matchCentreData" in match_page.content():
                        break
                    time.sleep(0.8)
                else:
                    print("   ⚠️ Timeout: matchCentreData no encontrado, salto.")
                    continue

                html = match_page.content()
                info = extract_match_info(html)
                if not info:
                    print("   ⚠️ No se pudo extraer info del partido, salto.")
                    continue

                home, away, match_id = info

                if int(match_id) in processed_ids:
                    print(f"   ⏭️ matchId {match_id} ya en parquet, salto.")
                    continue

                print(f"   → {home} vs {away} | ID: {match_id}")
                saved = save_html_to_inbox(html, match_id, match_page.title(), inbox_folder)
                total_saved += saved

                if saved:
                    processed_ids.add(int(match_id))

                time.sleep(REQUEST_DELAY)

            except Exception as exc:
                print(f"   ❌ Error: {exc}")

        browser.close()

    print(f"\n✅ Páginas abiertas: {total_opened}")
    print(f"✅ HTML guardados (nuevos): {total_saved}")

    if total_saved > 0:
        print("\n📊 Actualizando Parquet...")
        process_and_save(
            inbox_folder,
            output_parquet=output_parquet,
            league=league_name,
            season=season
        )

        deleted = 0
        for f in inbox_folder.glob("*.html"):
            f.unlink()
            deleted += 1
        print(f"🧹 Inbox vaciado ({deleted} archivos borrados).")
    else:
        print("\nℹ️ No se guardó ningún HTML nuevo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--league",
        required=True,
        choices=LEAGUES.keys(),
        help="League to scrape"
    )
    parser.add_argument(
        "--manual-wait-seconds",
        type=int,
        default=10,
        help="Seconds to manually move the calendar before scraping the visible week"
    )
    args = parser.parse_args()
    run(args.league, manual_wait_seconds=args.manual_wait_seconds)

#python event_data/scraper/script/whoscored_downloader.py --league serie_a