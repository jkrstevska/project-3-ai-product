"""Download the Balkans travel guides from Wikivoyage into data/raw/.

Source
------
English Wikivoyage (https://en.wikivoyage.org), fetched through its public
MediaWiki API. Text is licensed CC BY-SA 3.0/4.0: credit Wikivoyage and its
contributors in your README and slides, and do not commit the data.

What it downloads
-----------------
Every destination article (countries, regions, cities, rural areas, national
parks, itineraries) whose breadcrumb leads to one of the Balkan countries in
COUNTRIES, plus the itineraries and parks filed directly under the "Balkans"
region itself (cross-border routes, which belong to no single country).
Articles are found four ways, so a missing category does not silently drop
a region:

1. Walking down the category tree from each country (Wikivoyage's IsPartOf
   template puts every article in a category named after its parent region),
   and reading the articles directly in the "Balkans" category.
2. Sweeping all park and itinerary articles worldwide (status categories such
   as "Usable parks" or "Usable itineraries") and keeping the Balkan ones.
3. Following the destination links ("Cities", "Other destinations", ...) of
   the kept region and country articles, which reaches articles whose parent
   region has no category.
4. Any titles passed with --extra, kept as they are.

Every candidate is then checked by walking *up* its IsPartOf chain, which also
gives each article its country and breadcrumb.

The raw wikitext is kept as it is. Cleaning, splitting into sections and
choosing the nature-relevant articles happen in notebook 01, where the rules
can be read and justified.

Output
------
data/raw/wikivoyage_balkans.jsonl           one article per line
data/raw/wikivoyage_balkans_manifest.json   download date, licence, counts

Usage
-----
    python download_data.py                      # the 9 default countries
    python download_data.py --extra "Some Itinerary"   # add an article by title
    python download_data.py --countries Greece   # add a country
    python download_data.py --no-sweep           # faster, category tree only
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from src.functions import RAW
import sys

import requests

API_URL = "https://en.wikivoyage.org/w/api.php"
# Wikimedia asks every API client for a descriptive User-Agent.
# Replace the URL with your own GitHub repo so they can reach you if needed.
USER_AGENT = "TrailScout-student-project/0.1 (educational; https://github.com/jkrstevska/project-3-ai-product)"

OUT_FILE = RAW / "wikivoyage_balkans.jsonl"
MANIFEST_FILE = RAW / "wikivoyage_balkans_manifest.json"


# Wikivoyage article titles of the countries kept. Slovenia is excluded:
# Wikivoyage counts it as Central Europe, not the Balkans.
COUNTRIES = [
    "Albania",
    "Bosnia and Herzegovina",
    "Bulgaria",
    "Croatia",
    "Kosovo",
    "Montenegro",
    "North Macedonia",
    "Romania",
    "Serbia",
]

# The region article above the countries. Articles filed directly under it
# (not under a country) are cross-border: kept only if they are of a type in
# ROOT_LEVEL_TYPES, so a non-kept country filed under "Balkans" (and everything
# below it) is never pulled in.
ROOT_REGION = "Balkans"
ROOT_LEVEL_TYPES = {"itinerary", "park"}

# Status categories swept as a safety net. The names follow Wikivoyage's
# "<status> <type>s" pattern (as in "Usable stations"); they are inferred, not
# confirmed, so check the counts printed when the script runs: 0 for all four
# of a type means the names are wrong, not that there are no such articles.
SWEEP_CATEGORIES = {
    "park": ["Outline parks", "Usable parks", "Guide parks", "Star parks"],
    "itinerary": ["Outline itineraries", "Usable itineraries", "Guide itineraries", "Star itineraries"],
}

# Sections of region and country articles that list destinations, whose links
# are followed to find articles the category walk missed.
DESTINATION_SECTIONS = {"cities", "other destinations", "towns", "towns and villages", "regions"}
LINK_SOURCE_TYPES = {"country", "region", "ruralarea", None}   # None: no status template
MAX_LINK_ROUNDS = 5

BATCH_SIZE = 50        # the API's limit on pages per content request
MAX_DEPTH = 15         # longest breadcrumb we follow before giving up
PAUSE_SECONDS = 0.2    # be polite: one request at a time, with a short pause

ISPARTOF_RE = re.compile(r"\{\{\s*(?:is\s*part\s*of|is\s*in)\s*\|\s*([^|}\n]+)", re.IGNORECASE)
HEADING_RE = re.compile(r"^==\s*([^=].*?)\s*==\s*$", re.MULTILINE)   # level-2 headings only
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")
STATUS_RE = re.compile(
    r"\{\{\s*(stub|outline|usable|guide|star)"
    r"(city|park|region|country|ruralarea|district|itinerary|topic|airport|continent|dive)\s*[|}]",
    re.IGNORECASE,
)


def normalise_title(title: str) -> str:
    """Return a title as MediaWiki stores it: spaces, not underscores, first letter upper case."""
    title = title.replace("_", " ").strip()
    return title[:1].upper() + title[1:] if title else title


def parse_parent(wikitext: str) -> str | None:
    """Return the region named in the article's {{IsPartOf|...}} template, or None."""
    match = ISPARTOF_RE.search(wikitext or "")
    return normalise_title(match.group(1)) if match else None


def parse_status(wikitext: str) -> tuple[str | None, str | None]:
    """Return (status, article type) from a template such as {{usablepark}}, e.g. ("usable", "park")."""
    match = STATUS_RE.search(wikitext or "")
    if not match:
        return None, None
    return match.group(1).lower(), match.group(2).lower()


class WikivoyageClient:
    """A small, polite wrapper around the MediaWiki API, with retries."""

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.n_requests = 0

    def get(self, params: dict) -> dict:
        """Send one API request and return its JSON, retrying on rate limits and server errors."""
        params = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
        for attempt in range(5):
            time.sleep(PAUSE_SECONDS)
            self.n_requests += 1
            try:
                response = self.session.get(API_URL, params=params, timeout=60)
            except requests.RequestException as error:
                print(f"  network error ({error}); retrying")
                time.sleep(2 ** attempt)
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            data = response.json()
            if data.get("error", {}).get("code") == "maxlag":  # servers busy: wait and retry
                time.sleep(5)
                continue
            if "error" in data:
                raise RuntimeError(f"API error: {data['error']}")
            return data
        raise RuntimeError(f"API request kept failing: {params}")

    def category_members(self, category: str) -> tuple[list[str], list[str]]:
        """Return (article titles, subcategory names) directly in a category."""
        pages, subcats = [], []
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmnamespace": "0|14",
            "cmlimit": "max",
        }
        while True:
            data = self.get(params)
            for member in data.get("query", {}).get("categorymembers", []):
                if member["ns"] == 14:
                    subcats.append(member["title"].removeprefix("Category:"))
                else:
                    pages.append(member["title"])
            if "continue" not in data:
                return pages, subcats
            params = {**params, **data["continue"]}

    def fetch_pages(self, titles: list[str]) -> tuple[dict[str, dict], dict[str, str]]:
        """Fetch the current wikitext of up to BATCH_SIZE pages.

        Returns the pages found, keyed by their canonical title, and an alias map
        from each requested title (after normalising or following a redirect) to
        the canonical title.
        """
        data = self.get({
            "action": "query",
            "prop": "revisions",
            "rvprop": "content|ids|timestamp",
            "rvslots": "main",
            "redirects": 1,
            "titles": "|".join(titles),
        })
        query = data.get("query", {})
        aliases = {}
        for step in query.get("normalized", []) + query.get("redirects", []):
            aliases[step["from"]] = step["to"]
        pages = {}
        for page in query.get("pages", []):
            if page.get("missing") or page.get("invalid") or page.get("ns", 0) != 0:
                continue
            revision = page["revisions"][0]
            pages[page["title"]] = {
                "pageid": page["pageid"],
                "title": page["title"],
                "revid": revision["revid"],
                "last_edited": revision["timestamp"],
                "wikitext": revision["slots"]["main"]["content"],
            }
        return pages, aliases


class BalkansCrawler:
    """Collects articles and works out which Balkan country each belongs to."""

    def __init__(self, client: WikivoyageClient, countries: list[str]):
        self.client = client
        self.countries = set(countries)
        self.pages: dict[str, dict] = {}     # canonical title -> page record
        self.aliases: dict[str, str] = {}    # requested title -> canonical title
        self.missing: set[str] = set()       # titles that do not exist

    def canonical(self, title: str) -> str:
        """Follow normalisations and redirects (possibly chained) to the stored title."""
        seen = set()
        while title in self.aliases and title not in seen:
            seen.add(title)
            title = self.aliases[title]
        return title

    def ensure_fetched(self, titles) -> None:
        """Fetch every title not already downloaded, in batches."""
        todo = sorted({normalise_title(t) for t in titles}
                      - set(self.pages) - self.missing - set(self.aliases))
        for start in range(0, len(todo), BATCH_SIZE):
            batch = todo[start:start + BATCH_SIZE]
            pages, aliases = self.client.fetch_pages(batch)
            self.pages.update(pages)
            self.aliases.update(aliases)
            for title in batch:
                if self.canonical(title) not in self.pages:
                    self.missing.add(title)
            if len(todo) > BATCH_SIZE:
                print(f"  fetched {min(start + BATCH_SIZE, len(todo))}/{len(todo)}")

    def crawl_categories(self) -> set[str]:
        """Walk down the category tree from each country; return the article titles found."""
        found, seen_cats = set(self.countries), set()
        queue = list(self.countries)
        while queue:
            category = queue.pop()
            if category in seen_cats:
                continue
            seen_cats.add(category)
            pages, subcats = self.client.category_members(category)
            found.update(pages)
            queue.extend(c for c in subcats if c not in seen_cats)
        # Articles directly in the region's category: the cross-border ones.
        # Its subcategories are not followed, as they include non-kept countries.
        root_pages, _ = self.client.category_members(ROOT_REGION)
        found.update(root_pages)
        print(f"  {len(seen_cats)} country categories walked, {len(root_pages)} articles "
              f"directly under {ROOT_REGION}, {len(found)} articles found")
        return found

    def breadcrumb(self, title: str) -> list[str] | None:
        """Return the chain from the country down to the article, or None if it is not kept.

        An article is kept if its chain reaches a kept country, or if it sits
        directly under ROOT_REGION and is of a type in ROOT_LEVEL_TYPES (a
        cross-border itinerary or park); then the chain starts at ROOT_REGION.
        Parents are fetched level by level as needed.
        """
        chain, current = [], self.canonical(normalise_title(title))
        for _ in range(MAX_DEPTH):
            if current in chain:            # a loop in the breadcrumbs
                return None
            chain.append(current)
            if current in self.countries:
                return list(reversed(chain))
            if current == ROOT_REGION:
                article = self.pages.get(chain[0])
                _, article_type = parse_status(article["wikitext"]) if article else (None, None)
                if len(chain) == 2 and article_type in ROOT_LEVEL_TYPES:
                    return list(reversed(chain))
                return None                 # e.g. Greece, or something deeper under it
            self.ensure_fetched([current])
            page = self.pages.get(self.canonical(current))
            parent = parse_parent(page["wikitext"]) if page else None
            if parent is None:
                return None
            current = self.canonical(parent)
        return None

    def resolve_level_by_level(self, titles: set[str]) -> None:
        """Pre-fetch all ancestors in batches, so breadcrumb() rarely fetches one page at a time."""
        frontier = set(titles)
        for _ in range(MAX_DEPTH):
            self.ensure_fetched(frontier)
            parents = set()
            for title in frontier:
                page = self.pages.get(self.canonical(normalise_title(title)))
                parent = parse_parent(page["wikitext"]) if page else None
                if parent and parent not in self.countries:
                    parents.add(parent)
            frontier = {p for p in parents if self.canonical(p) not in self.pages and p not in self.missing}
            if not frontier:
                return


def destination_links(wikitext: str) -> set[str]:
    """Return the article titles linked from an article's destination-list sections.

    Region and country articles list their towns and parks under headings such
    as "Cities" or "Other destinations". Following these links finds articles
    whose parent region has no category, which the category walk cannot reach.
    """
    parts = HEADING_RE.split(wikitext or "")
    titles = set()
    for heading, body in zip(parts[1::2], parts[2::2]):
        if heading.strip().lower() in DESTINATION_SECTIONS:
            for link in WIKILINK_RE.findall(body):
                if ":" not in link:          # skip File:, Category:, other wikis
                    titles.add(normalise_title(link))
    return titles


def expand_via_links(crawler: BalkansCrawler, candidates: set[str]) -> set[str]:
    """Add the destinations linked from kept region-level articles, until no new article is kept."""
    candidates = set(candidates)
    for round_number in range(1, MAX_LINK_ROUNDS + 1):
        kept = [crawler.pages[c[-1]] for t in candidates
                if (c := crawler.breadcrumb(t)) and c[-1] in crawler.pages]
        linked = set()
        for page in kept:
            if parse_status(page["wikitext"])[1] in LINK_SOURCE_TYPES:
                linked |= destination_links(page["wikitext"])
        new = {t for t in linked if crawler.canonical(t) not in candidates and t not in candidates}
        if not new:
            return candidates
        crawler.resolve_level_by_level(new)
        new_kept = {t for t in new if crawler.breadcrumb(t)}
        print(f"  round {round_number}: {len(new)} linked titles checked, {len(new_kept)} new articles kept")
        if not new_kept:
            return candidates
        candidates |= new_kept
    return candidates


def build_records(crawler: BalkansCrawler, candidates: set[str]) -> list[dict]:
    """Keep the candidates inside a kept country and turn them into output records."""
    records, kept = [], set()
    for title in sorted(candidates):
        chain = crawler.breadcrumb(title)
        if chain is None:
            continue
        page = crawler.pages.get(chain[-1])
        if page is None:                    # e.g. a country title with a typo: nothing to save
            continue
        if page["title"] in kept:           # two candidates redirecting to one article
            continue
        kept.add(page["title"])
        status, article_type = parse_status(page["wikitext"])
        records.append({
            "pageid": page["pageid"],
            "title": page["title"],
            "url": "https://en.wikivoyage.org/wiki/" + page["title"].replace(" ", "_"),
            "country": chain[0] if chain[0] in crawler.countries else None,  # None: cross-border
            "breadcrumb": chain,
            "article_type": article_type,   # park, city, region, ruralarea, itinerary, ...
            "status": status,               # stub/outline/usable/guide/star: how complete it is
            "revid": page["revid"],
            "last_edited": page["last_edited"],
            "wikitext": page["wikitext"],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--countries", nargs="+", default=[],
                        help="extra countries to add to the default nine (Wikivoyage titles)")
    parser.add_argument("--extra", nargs="+", default=[],
                        help="extra article titles to download as they are, e.g. itineraries "
                             "(an itinerary title, say) (kept even without a Balkan breadcrumb)")
    parser.add_argument("--no-sweep", action="store_true",
                        help="skip the worldwide sweep of park and itinerary articles "
                             "(faster, may miss some)")
    args = parser.parse_args()

    countries = COUNTRIES + [normalise_title(c) for c in args.countries]
    client = WikivoyageClient()
    crawler = BalkansCrawler(client, countries)

    print(f"1/5 Walking the category tree of {len(countries)} countries")
    candidates = crawler.crawl_categories()

    if not args.no_sweep:
        print("2/5 Sweeping park and itinerary articles worldwide")
        for categories in SWEEP_CATEGORIES.values():
            for category in categories:
                pages, _ = client.category_members(category)
                print(f"  {category}: {len(pages)}")
                candidates.update(pages)
    else:
        print("2/5 Sweep skipped")

    print(f"3/5 Downloading {len(candidates)} candidate articles and their parent regions")
    crawler.resolve_level_by_level(candidates)

    print("4/5 Following destination links from region and country articles")
    candidates = expand_via_links(crawler, candidates)

    print("5/5 Keeping the articles inside the Balkan countries")
    records = build_records(crawler, candidates)

    if args.extra:                           # itineraries and topics cross borders: keep as they are
        crawler.ensure_fetched(args.extra)
        kept = {r["title"] for r in records}
        for title in args.extra:
            page = crawler.pages.get(crawler.canonical(normalise_title(title)))
            if page is None:
                print(f"  extra article not found: {title}")
                continue
            if page["title"] in kept:
                continue
            status, article_type = parse_status(page["wikitext"])
            records.append({
                "pageid": page["pageid"], "title": page["title"],
                "url": "https://en.wikivoyage.org/wiki/" + page["title"].replace(" ", "_"),
                "country": None, "breadcrumb": [page["title"]],
                "article_type": article_type, "status": status,
                "revid": page["revid"], "last_edited": page["last_edited"],
                "wikitext": page["wikitext"],
            })

    RAW.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_country = Counter(r["country"] or "(cross-border)" for r in records)
    by_type = Counter(r["article_type"] or "unknown" for r in records)
    manifest = {
        "source": "English Wikivoyage, https://en.wikivoyage.org",
        "licence": "CC BY-SA 3.0/4.0 - credit 'Wikivoyage contributors' and link to each article",
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "countries": countries,
        "n_articles": len(records),
        "articles_by_country": dict(by_country.most_common()),
        "articles_by_type": dict(by_type.most_common()),
        "n_api_requests": client.n_requests,
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved {len(records)} articles to {OUT_FILE.relative_to(Path.cwd()) if OUT_FILE.is_relative_to(Path.cwd()) else OUT_FILE}")
    print("By country:", dict(by_country.most_common()))
    print("By type:   ", dict(by_type.most_common()))
    print(f"Parks:      {by_type.get('park', 0)}")


if __name__ == "__main__":
    main()