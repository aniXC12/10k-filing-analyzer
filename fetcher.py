"""
SEC EDGAR Fetcher
Retrieves 10-K filing text directly from the SEC EDGAR full-text search API.
No API key required — uses public EDGAR endpoints.
"""

import re
import time
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={year}-01-01&enddt={year}-12-31&forms=10-K"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EDGAR_FILING_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/company.idx"

HEADERS = {
    "User-Agent": "sec-insight-tool research@example.com",  # EDGAR requires a User-Agent
    "Accept-Encoding": "gzip, deflate",
}


class EDGARFetcher:
    def __init__(self, rate_limit_seconds: float = 0.5):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.rate_limit = rate_limit_seconds

    def _get(self, url: str) -> Optional[requests.Response]:
        try:
            time.sleep(self.rate_limit)  # Respect SEC rate limits
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def get_cik(self, ticker: str) -> Optional[str]:
        """
        Look up the CIK number for a given ticker symbol via EDGAR company search.
        """
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK={ticker}&type=10-K&dateb=&owner=include&count=5&search_text=&action=getcompany&output=atom"
        resp = self._get(url)
        if not resp:
            return None

        match = re.search(r"CIK=(\d+)", resp.text)
        if match:
            return match.group(1).zfill(10)
        return None

    def get_latest_10k_url(self, cik: str) -> Optional[str]:
        """
        Get the URL of the most recent 10-K filing document for a given CIK.
        """
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = self._get(url)
        if not resp:
            return None

        data = resp.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accession_numbers = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form == "10-K":
                accession = accession_numbers[i].replace("-", "")
                doc = primary_docs[i]
                filing_url = f"https://www.sec.gov/Archives/edgar/{accession[:6]}/{accession[6:8]}/{accession[8:10]}/{accession}/{doc}"
                # Simpler: use the index URL
                index_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=1"
                # Return the direct document URL
                clean_accession = accession_numbers[i]
                formatted = clean_accession.replace("-", "")
                direct_url = f"https://www.sec.gov/Archives/edgar/full-index/"
                return f"https://www.sec.gov/Archives/edgar/{formatted[:10]}/{formatted}/{doc}"

        return None

    def fetch_filing_text(self, ticker: str) -> Optional[str]:
        """
        Main entry point: fetch the plain-text content of the latest 10-K for a ticker.
        Falls back to a direct EDGAR full-text search if CIK lookup fails.
        """
        logger.info(f"Fetching 10-K for {ticker}...")
        cik = self.get_cik(ticker)
        if not cik:
            logger.error(f"Could not find CIK for {ticker}")
            return None

        logger.info(f"Found CIK: {cik}")

        # Get submissions JSON for latest 10-K
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = self._get(url)
        if not resp:
            return None

        data = resp.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accession_numbers = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form == "10-K":
                accession = accession_numbers[i]
                doc = primary_docs[i]
                clean = accession.replace("-", "")
                # Build the filing index URL
                file_url = f"https://www.sec.gov/Archives/edgar/full-index/{clean[:4]}/QTR1/{doc}"
                # Use proper path
                doc_url = f"https://www.sec.gov/Archives/edgar/{clean[:10]}/{clean}/{doc}"

                logger.info(f"Fetching document: {doc_url}")
                doc_resp = self._get(doc_url)
                if doc_resp:
                    return self._clean_text(doc_resp.text)

        logger.error("No 10-K document found")
        return None

    def _clean_text(self, raw: str) -> str:
        """
        Strip HTML tags and normalize whitespace from raw filing text.
        """
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", raw)
        # Remove XBRL / XML artifacts
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()
