"""
sec-insight CLI
---------------
Usage:
    python main.py --ticker AAPL
    python main.py --ticker MSFT --file path/to/filing.txt
    python main.py --ticker TSLA --output results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.fetcher import EDGARFetcher
from src.analyzer import SECAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze SEC 10-K filings using facebook/bart-large-cnn"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, MSFT, TSLA)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a local .txt file containing the filing (skips EDGAR fetch)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON results (default: print to stdout)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device index for model (-1 = CPU, 0 = first GPU)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load filing text ---
    if args.file:
        logger.info(f"Loading filing from local file: {args.file}")
        text = Path(args.file).read_text(encoding="utf-8", errors="ignore")
    else:
        fetcher = EDGARFetcher()
        text = fetcher.fetch_filing_text(args.ticker)
        if not text:
            logger.error(f"Failed to fetch 10-K for {args.ticker}. Try --file to use a local copy.")
            sys.exit(1)

    logger.info(f"Filing text loaded ({len(text):,} chars)")

    # --- Run analysis ---
    analyzer = SECAnalyzer(device=args.device)
    results = analyzer.analyze(text)
    results["ticker"] = args.ticker.upper()

    # --- Output ---
    output_str = json.dumps(results, indent=2)

    if args.output:
        Path(args.output).write_text(output_str)
        logger.info(f"Results written to {args.output}")
    else:
        print("\n" + "=" * 60)
        print(f"  SEC INSIGHT RESULTS: {args.ticker.upper()}")
        print("=" * 60)

        risk = results["risk_factors"]
        if risk["found"]:
            print(f"\nRISK FACTORS SUMMARY ({risk['raw_length']:,} chars analyzed):")
            print("-" * 40)
            print(risk["summary"])
        else:
            print("\nRISK FACTORS: Section not found in document.")

        highlights = results["financial_highlights"]
        if highlights["found"]:
            print(f"\nFINANCIAL HIGHLIGHTS SUMMARY ({highlights['raw_length']:,} chars analyzed):")
            print("-" * 40)
            print(highlights["summary"])
        else:
            print("\nFINANCIAL HIGHLIGHTS: Section not found in document.")

        print("\n" + "=" * 60)
        print(f"Model used: {results['model']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
