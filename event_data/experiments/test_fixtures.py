from playwright.sync_api import sync_playwright
import json

URL = "https://it.whoscored.com/regions/108/tournaments/5/italia-serie-a"

def get_fixtures_json(page):
    el = page.query_selector(
        "script[data-hypernova-key='tournamentfixtures'][type='application/json']"
    )
    txt = el.text_content().strip()

    if txt.startswith("<!--"):
        txt = txt[4:]
    if txt.endswith("-->"):
        txt = txt[:-3]

    return json.loads(txt)

with sync_playwright() as p:

    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    page.goto(URL)
    page.wait_for_timeout(3000)

    fixtures = get_fixtures_json(page)

    print(fixtures.keys())
    print(fixtures["tournaments"][0].keys())

    match = fixtures["tournaments"][0]["matches"][0]
    print(match)

    browser.close()