"""
Unit tests for sec-insight.
Run with: pytest tests/
"""

import pytest
from src.analyzer import SECAnalyzer

# Minimal fake 10-K structure for testing without downloading the model
FAKE_FILING = """
ITEM 1A. RISK FACTORS

Our business faces significant competition from established technology companies
with greater resources. We may be unable to attract and retain key personnel.
Macroeconomic conditions including inflation and interest rate changes could
materially adversely affect our operating results. Cybersecurity threats pose
ongoing risks to our infrastructure and customer data.

ITEM 1B. UNRESOLVED STAFF COMMENTS

None.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Revenue increased 12% year-over-year to $94.5 billion driven by strong cloud
services growth. Operating income improved to $28.2 billion representing a
29.8% operating margin. Free cash flow was $21.8 billion. We returned
$15.2 billion to shareholders through dividends and share repurchases.

ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK
"""


def test_extract_risk_factors_found():
    analyzer = SECAnalyzer.__new__(SECAnalyzer)
    # Patch tokenizer and summarizer to avoid downloading model in tests
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text.split())))
        def decode(self, tokens, skip_special_tokens=True):
            return "decoded text"

    class FakeSummarizer:
        def __call__(self, text, **kwargs):
            return [{"summary_text": "Mocked summary of risk factors."}]

    analyzer.tokenizer = FakeTokenizer()
    analyzer.summarizer = FakeSummarizer()
    analyzer.model_name = "mock"

    result = analyzer.extract_risk_factors(FAKE_FILING)
    assert result["found"] is True
    assert result["raw_length"] > 0
    assert isinstance(result["summary"], str)


def test_extract_financial_highlights_found():
    analyzer = SECAnalyzer.__new__(SECAnalyzer)

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text.split())))
        def decode(self, tokens, skip_special_tokens=True):
            return "decoded text"

    class FakeSummarizer:
        def __call__(self, text, **kwargs):
            return [{"summary_text": "Mocked summary of financials."}]

    analyzer.tokenizer = FakeTokenizer()
    analyzer.summarizer = FakeSummarizer()
    analyzer.model_name = "mock"

    result = analyzer.extract_financial_highlights(FAKE_FILING)
    assert result["found"] is True
    assert result["raw_length"] > 0


def test_extract_missing_section():
    analyzer = SECAnalyzer.__new__(SECAnalyzer)

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text.split())))
        def decode(self, tokens, skip_special_tokens=True):
            return "decoded text"

    class FakeSummarizer:
        def __call__(self, text, **kwargs):
            return [{"summary_text": "Summary."}]

    analyzer.tokenizer = FakeTokenizer()
    analyzer.summarizer = FakeSummarizer()
    analyzer.model_name = "mock"

    result = analyzer.extract_risk_factors("This document has no standard sections.")
    assert result["found"] is False
    assert result["summary"] is None
