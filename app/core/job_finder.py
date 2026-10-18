import requests
from typing import List, Dict

REMOTEOK_URL = "https://remoteok.com/api"


def fetch_remoteok_jobs(keywords: List[str], max_results: int = 200) -> List[Dict]:
    """
    Fetches RemoteOK's full public job feed and filters it locally by
    keyword match against title/tags/description - RemoteOK's API doesn't
    support reliable server-side keyword search, so we filter client-side.

    RemoteOK requires a real User-Agent header or it may reject the request.
    The first item in the response is always a legal/attribution notice,
    not a real job - it's skipped.
    """
    headers = {"User-Agent": "Mozilla/5.0 (personal resume tool; contact: you@example.com)"}
    response = requests.get(REMOTEOK_URL, headers=headers, timeout=15)
    response.raise_for_status()
    raw_jobs = response.json()

    if raw_jobs and "legal" in raw_jobs[0]:
        raw_jobs = raw_jobs[1:]  # drop the disclaimer entry

    keywords_lower = [k.lower() for k in keywords]
    matched = []

    for job in raw_jobs:
        searchable_text = " ".join([
            job.get("position", ""),
            job.get("company", ""),
            " ".join(job.get("tags", [])),
        ]).lower()

        if any(kw in searchable_text for kw in keywords_lower):
            matched.append({
                "id": job.get("id"),
                "title": job.get("position", "Unknown role"),
                "company": job.get("company", "Unknown company"),
                "url": job.get("url", ""),
                "description": job.get("description", ""),
                "tags": job.get("tags", []),
            })

        if len(matched) >= max_results:
            break

    return matched


if __name__ == "__main__":
    test_keywords = ["Machine Learning", "NLP", "AI Engineer"]
    jobs = fetch_remoteok_jobs(test_keywords, max_results=10)
    print(f"Found {len(jobs)} matching jobs")
    for job in jobs[:5]:
        print(f"- {job['title']} @ {job['company']} ({job['url']})")