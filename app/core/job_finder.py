import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://jobs.citi.com"
SEARCH_URL = f"{BASE_URL}/search-jobs"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (personal resume tool; contact: you@example.com)",
    "Accept-Language": "en-US,en;q=0.9",
}

# Words that appear in almost every job title and say nothing about the domain.
ROLE_NOUNS = {"engineer", "developer", "scientist", "researcher", "architect", "analyst"}

GENERIC_WORDS = {
    "engineer", "developer", "analyst", "associate", "senior", "junior", "lead",
    "principal", "staff", "specialist", "consultant", "manager", "intern",
    "engineering", "software", "and", "of", "the", "i", "ii", "iii", "avp",
    "vice", "president", "assistant", "officer",
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tokens(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9\+\#\.]+", text.lower()))


def _location_from_url(url: str) -> str:
    """Citi job URLs look like /job/{city}/{title}/287/{id}; use the city as a fallback."""
    m = re.search(r"/job/([^/]+)/", url)
    return m.group(1).replace("-", " ").title() if m else "Unknown"


# --------------------------------------------------------------------------
# Domain relevance
# --------------------------------------------------------------------------
def _prepare_keywords(keywords: List[str]) -> List[Dict]:
    """Normalizes keywords and splits each into its 'domain' words."""
    prepared, seen = [], set()
    for kw in keywords:
        kw_clean = _clean(kw)
        if not kw_clean or kw_clean.lower() in seen:
            continue
        seen.add(kw_clean.lower())
        domain_tokens = _tokens(kw_clean) - GENERIC_WORDS
        role_words = _tokens(kw_clean) & ROLE_NOUNS
        prepared.append({"raw": kw_clean, "lower": kw_clean.lower(), "domain": domain_tokens, "roles": role_words})
    return prepared


def _matching_keyword(title: str, prepared: List[Dict]) -> str:
    """
    Returns the first keyword that this job title is relevant to, or "".
    A title is relevant if it contains the keyword phrase, or contains ALL of the
    keyword's domain words AND the keyword's role noun (Engineer, Scientist, ...).
    Requiring the role noun stops short words like "AI" from matching unrelated
    titles such as "AI Audio Data Trainer" or "AEO Specialist (AI Search)".
    """
    title_lower = title.lower()
    title_tokens = _tokens(title)
    if "engineering" in title_tokens:
        title_tokens.add("engineer")
    if "scientists" in title_tokens:
        title_tokens.add("scientist")
    for kw in prepared:
        if kw["lower"] in title_lower:
            return kw["raw"]
        if kw["domain"] and kw["domain"].issubset(title_tokens):
            if not kw["roles"] or (kw["roles"] & title_tokens):
                return kw["raw"]
    return ""


# --------------------------------------------------------------------------
# Citi scraping
# --------------------------------------------------------------------------
def _search_citi(keyword: str, location: str, max_pages: int) -> List[Dict]:
    """Runs one keyword search on jobs.citi.com and returns raw listings."""
    results = []
    for page in range(1, max_pages + 1):
        params = {"k": keyword, "l": location, "p": page}
        try:
            resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"[Citi] search failed for '{keyword}' page {page}: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.select("a[href*='/job/']")
        if not links:
            break

        for a in links:
            href = a.get("href")
            if not isinstance(href, str):
                continue
            url = urljoin(BASE_URL, href)
            title_tag = a.find(["h2", "h3"])
            title = _clean(title_tag.get_text()) if title_tag else _clean(a.get_text())
            loc_tag = a.select_one("[class*='location']")
            loc = _clean(loc_tag.get_text()) if loc_tag else _location_from_url(url)
            results.append({"title": title, "url": url, "location": loc})
        time.sleep(0.5)
    return results


def _fetch_details(job: Dict) -> Dict:
    """Opens one job page and fills in the description and exact location."""
    try:
        resp = requests.get(job["url"], headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"[Citi] detail fetch failed for {job['url']}: {e}")
        return job

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    container = (
        soup.select_one(".ats-description")
        or soup.select_one("[class*='description']")
        or soup.find("main")
        or soup.body
    )
    job["description"] = _clean(container.get_text(" ")) if container else ""

    m = re.search(
        r"Location\(s\):\s*(.+?)\s*(?:Job Type:|Posted:|Discover)",
        _clean(soup.get_text(" ")),
    )
    if m:
        job["location"] = m.group(1).strip()
    return job


def fetch_citi_jobs(
    keywords: List[str],
    location: str = "India",
    max_results: int = 50,
    max_pages: int = 1,
    with_descriptions: bool = True,
    workers: int = 4,
) -> List[Dict]:
    """
    Searches Citi for each detected keyword, then keeps ONLY the jobs whose
    title is relevant to the candidate's domain. Descriptions are fetched only
    for the jobs that survive the filter.
    """
    prepared = _prepare_keywords(keywords)
    if not prepared:
        return []

    # 1) Search Citi for every keyword (in parallel, a few at a time)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        batches = list(pool.map(
            lambda kw: _search_citi(kw["raw"], location, max_pages), prepared
        ))

    # 2) Dedupe by URL and keep only domain-relevant titles
    seen_urls: Set[str] = set()
    relevant: List[Dict] = []
    for batch in batches:
        for item in batch:
            if item["url"] in seen_urls:
                continue
            seen_urls.add(item["url"])
            matched = _matching_keyword(item["title"], prepared)
            if not matched:
                continue
            relevant.append({
                "title": item["title"] or "Unknown role",
                "company": "Citi",
                "url": item["url"],
                "description": "",
                "location": item["location"],
                "source": "Citi Careers",
                "matched_keyword": matched,
            })

    relevant = relevant[:max_results]

    # 3) Fetch full descriptions only for the relevant jobs
    if with_descriptions and relevant:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            relevant = list(pool.map(_fetch_details, relevant))

    return relevant


# --------------------------------------------------------------------------
# Wellfound (best effort - the site blocks many automated requests)
# --------------------------------------------------------------------------
WELLFOUND_BASE = "https://wellfound.com"


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def fetch_wellfound_jobs(
    keywords: List[str],
    location: str = "India",
    max_results: int = 50,
    max_keywords: int = 8,
    **_unused,
) -> List[Dict]:
    """
    Reads Wellfound's public role pages (wellfound.com/role/l/{role}/{location}).
    Wellfound has no public API and often answers automated requests with a block
    page. If that happens this stops immediately and returns whatever it already
    collected, so it never breaks the other sources. It does not try to get around
    the block.
    """
    prepared = _prepare_keywords(keywords)[:max_keywords]
    loc_slug = _slugify(location)

    jobs: List[Dict] = []
    seen_urls: Set[str] = set()
    for kw in prepared:
        url = f"{WELLFOUND_BASE}/role/l/{_slugify(kw['raw'])}/{loc_slug}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
        except Exception as e:
            print(f"[Wellfound] request failed for '{kw['raw']}': {e}")
            continue

        blocked = resp.status_code in (401, 403, 429) or any(
            s in resp.text.lower() for s in ("captcha-delivery", "datadome", "access denied")
        )
        if blocked:
            print("[Wellfound] blocked the request (bot protection). Skipping Wellfound.")
            break
        if resp.status_code != 200:
            print(f"[Wellfound] HTTP {resp.status_code} for '{kw['raw']}'")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.select("a[href*='/jobs/']"):
            href = a.get("href")
            if not isinstance(href, str):
                continue
            job_url = urljoin(WELLFOUND_BASE, href).split("?")[0]
            m = re.search(r"/jobs/\d+-([a-z0-9\-]+)", job_url)
            if not m or job_url in seen_urls:
                continue
            seen_urls.add(job_url)

            title = _clean(a.get_text()) or m.group(1).replace("-", " ").title()
            company = "Wellfound listing"
            node = a
            for _ in range(6):  # look at nearby parent blocks for a /company/ link
                node = node.parent
                if node is None:
                    break
                c = node.select_one("a[href*='/company/']")
                if c and _clean(c.get_text()):
                    company = _clean(c.get_text())
                    break
            matched = _matching_keyword(title, prepared)
            if not matched:
                continue
            jobs.append({
                "title": title,
                "company": company,
                "url": job_url,
                "description": "",
                "location": location,
                "source": "Wellfound",
                "matched_keyword": matched,
            })
            if len(jobs) >= max_results:
                return jobs
        time.sleep(2)  # be gentle
    return jobs


# --------------------------------------------------------------------------
# Combined entry point
# --------------------------------------------------------------------------
def fetch_all_jobs(keywords: List[str], max_results: int = 100, location: str = "India") -> List[Dict]:
    """
    Same name/signature as before so the rest of your project keeps working.
    Pulls from Citi and Wellfound. If one source fails, the other still returns results.
    """
    all_jobs: List[Dict] = []
    sources = (
        ("Citi", fetch_citi_jobs),
        ("Wellfound", fetch_wellfound_jobs),
    )
    for name, fetcher in sources:
        try:
            all_jobs += fetcher(keywords, location=location, max_results=max_results)
        except Exception as e:
            print(f"[{name}] source failed: {e}")

    # Dedupe by URL, then by (title, company, location)
    seen, unique = set(), []
    for job in all_jobs:
        key = (job["url"], job["title"].lower(), job["company"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique[:max_results]


if __name__ == "__main__":
    from app.core.schema import ResumeFacts
    from app.core.domain_detector import detect_search_keywords  # <- adjust to your module path

    with open("data/resume_facts.json", "r", encoding="utf-8") as f:
        facts = ResumeFacts.model_validate_json(f.read())

    keywords = detect_search_keywords(facts)
    print(f"Detected {len(keywords)} keywords: {keywords}\n")

    jobs = fetch_all_jobs(keywords, max_results=30)
    print(f"Found {len(jobs)} relevant jobs\n")
    for job in jobs:
        print(f"[{job['company']}] {job['title']} ({job['location']})  <- matched '{job['matched_keyword']}'")
        print(f"   {job['url']}")