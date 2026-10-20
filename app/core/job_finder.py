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
        prepared.append({"raw": kw_clean, "lower": kw_clean.lower(), "domain": domain_tokens})
    return prepared


def _matching_keyword(title: str, prepared: List[Dict]) -> str:
    """
    Returns the first keyword that this job title is relevant to, or "".
    A title is relevant if it contains the keyword phrase, or contains ALL of the
    keyword's domain words (e.g. "NLP Engineer" -> {"nlp"}, "Machine Learning
    Engineer" -> {"machine", "learning"}). This stops generic words like
    "Engineer" from matching every job.
    """
    title_lower = title.lower()
    title_tokens = _tokens(title)
    for kw in prepared:
        if kw["lower"] in title_lower:
            return kw["raw"]
        if kw["domain"] and kw["domain"].issubset(title_tokens):
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
# American Express (Eightfold-powered careers site)
# --------------------------------------------------------------------------
AMEX_BASE = "https://aexp.eightfold.ai"
AMEX_API = f"{AMEX_BASE}/api/apply/v2/jobs"
AMEX_DOMAIN = "aexp.com"


def _search_amex(keyword: str, location: str, max_pages: int, page_size: int = 10) -> List[Dict]:
    """Runs one keyword search against Amex's careers API and returns raw listings."""
    results = []
    for page in range(max_pages):
        params = {
            "domain": AMEX_DOMAIN,
            "query": keyword,
            "location": location,
            "start": page * page_size,
            "num": page_size,
            "sort_by": "relevance",
        }
        try:
            resp = requests.get(AMEX_API, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            positions = resp.json().get("positions", [])
        except Exception as e:
            print(f"[Amex] search failed for '{keyword}' page {page + 1}: {e}")
            break

        if not positions:
            break

        for p in positions:
            job_id = p.get("id")
            url = p.get("canonicalPositionUrl") or f"{AMEX_BASE}/careers/job/{job_id}"
            locs = p.get("locations") or [p.get("location", "")]
            results.append({
                "id": job_id,
                "title": _clean(p.get("name", "")),
                "url": url,
                "location": "; ".join(l for l in locs if l) or "Unknown",
                "description": _clean(BeautifulSoup(p.get("job_description", "") or "", "html.parser").get_text(" ")),
            })
        time.sleep(0.5)
    return results


def _fetch_amex_description(job: Dict) -> Dict:
    """Fills in the description from the Amex job-detail endpoint if the list didn't include it."""
    if job.get("description") or not job.get("_id"):
        return job
    try:
        resp = requests.get(
            f"{AMEX_API}/{job['_id']}", params={"domain": AMEX_DOMAIN}, headers=HEADERS, timeout=15
        )
        resp.raise_for_status()
        html = resp.json().get("job_description", "") or ""
        job["description"] = _clean(BeautifulSoup(html, "html.parser").get_text(" "))
    except Exception as e:
        print(f"[Amex] detail fetch failed for {job['url']}: {e}")
    return job


def fetch_amex_jobs(
    keywords: List[str],
    location: str = "India",
    max_results: int = 50,
    max_pages: int = 1,
    with_descriptions: bool = True,
    workers: int = 4,
) -> List[Dict]:
    """Same flow as Citi: search per keyword, keep only domain-relevant titles."""
    prepared = _prepare_keywords(keywords)
    if not prepared:
        return []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        batches = list(pool.map(
            lambda kw: _search_amex(kw["raw"], location, max_pages), prepared
        ))

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
                "company": "American Express",
                "url": item["url"],
                "description": item["description"],
                "location": item["location"],
                "source": "American Express Careers",
                "matched_keyword": matched,
                "_id": item["id"],
            })

    relevant = relevant[:max_results]

    if with_descriptions and relevant:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            relevant = list(pool.map(_fetch_amex_description, relevant))

    for job in relevant:
        job.pop("_id", None)
    return relevant


# --------------------------------------------------------------------------
# Combined entry point
# --------------------------------------------------------------------------
def fetch_all_jobs(keywords: List[str], max_results: int = 100, location: str = "India") -> List[Dict]:
    """
    Same name/signature as before so the rest of your project keeps working.
    Pulls from Citi and American Express. If one site fails, the other still returns results.
    """
    all_jobs: List[Dict] = []
    for name, fetcher in (("Citi", fetch_citi_jobs), ("American Express", fetch_amex_jobs)):
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