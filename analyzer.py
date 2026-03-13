"""
SEC Filing Analyzer
Uses facebook/bart-large-cnn via Hugging Face transformers to extract
key risk factors and financial highlights from 10-K filings.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import re
import logging

logger = logging.getLogger(__name__)

# Max tokens BART can handle at once
MAX_CHUNK_TOKENS = 1024
OVERLAP_TOKENS = 50


class SECAnalyzer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: int = -1):
        """
        Initialize summarizer using Hugging Face pipeline.

        Args:
            model_name: HF model ID for summarization
            device: -1 for CPU, 0+ for GPU index
        """
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=self.tokenizer,
            device=device,
        )
        self.model_name = model_name

    def chunk_text(self, text: str) -> list[str]:
        """
        Split long text into overlapping chunks that fit within BART's context window.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + MAX_CHUNK_TOKENS, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            if end == len(tokens):
                break
            start = end - OVERLAP_TOKENS  # slight overlap to preserve context
        return chunks

    def summarize_section(self, text: str, max_length: int = 200, min_length: int = 60) -> str:
        """
        Summarize a section of text, chunking if necessary.
        """
        text = text.strip()
        if not text:
            return ""

        token_count = len(self.tokenizer.encode(text))
        if token_count <= MAX_CHUNK_TOKENS:
            result = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True,
            )
            return result[0]["summary_text"]

        # Chunk and summarize each, then summarize the summaries
        chunks = self.chunk_text(text)
        partial_summaries = []
        for chunk in chunks:
            try:
                result = self.summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length // 2,
                    do_sample=False,
                    truncation=True,
                )
                partial_summaries.append(result[0]["summary_text"])
            except Exception as e:
                logger.warning(f"Chunk summarization failed: {e}")

        combined = " ".join(partial_summaries)
        # Final pass over combined summaries
        return self.summarize_section(combined, max_length=max_length, min_length=min_length)

    def extract_risk_factors(self, filing_text: str) -> dict:
        """
        Find and summarize the Risk Factors section (Item 1A) from a 10-K.
        """
        # Common patterns for Item 1A in 10-K filings
        patterns = [
            r"ITEM\s+1A[\.\s]+RISK FACTORS(.*?)(?=ITEM\s+1B|ITEM\s+2|\Z)",
            r"Item\s+1A[\.\s]+Risk Factors(.*?)(?=Item\s+1B|Item\s+2|\Z)",
            r"RISK FACTORS(.*?)(?=UNRESOLVED STAFF COMMENTS|PROPERTIES|\Z)",
        ]

        section_text = ""
        for pattern in patterns:
            match = re.search(pattern, filing_text, re.DOTALL | re.IGNORECASE)
            if match:
                section_text = match.group(1).strip()
                break

        if not section_text:
            logger.warning("Could not locate Risk Factors section")
            return {"found": False, "summary": None, "raw_length": 0}

        summary = self.summarize_section(section_text, max_length=300, min_length=100)
        return {
            "found": True,
            "summary": summary,
            "raw_length": len(section_text),
        }

    def extract_financial_highlights(self, filing_text: str) -> dict:
        """
        Find and summarize MD&A (Item 7) for financial highlights.
        """
        patterns = [
            r"ITEM\s+7[\.\s]+MANAGEMENT.S DISCUSSION(.*?)(?=ITEM\s+7A|ITEM\s+8|\Z)",
            r"Item\s+7[\.\s]+Management.s Discussion(.*?)(?=Item\s+7A|Item\s+8|\Z)",
            r"MANAGEMENT.S DISCUSSION AND ANALYSIS(.*?)(?=QUANTITATIVE AND QUALITATIVE|\Z)",
        ]

        section_text = ""
        for pattern in patterns:
            match = re.search(pattern, filing_text, re.DOTALL | re.IGNORECASE)
            if match:
                section_text = match.group(1).strip()
                break

        if not section_text:
            logger.warning("Could not locate MD&A section")
            return {"found": False, "summary": None, "raw_length": 0}

        summary = self.summarize_section(section_text, max_length=300, min_length=100)
        return {
            "found": True,
            "summary": summary,
            "raw_length": len(section_text),
        }

    def analyze(self, filing_text: str) -> dict:
        """
        Run full analysis pipeline on a 10-K filing text.

        Returns:
            dict with risk_factors and financial_highlights summaries
        """
        logger.info("Starting 10-K analysis...")

        risk = self.extract_risk_factors(filing_text)
        highlights = self.extract_financial_highlights(filing_text)

        return {
            "model": self.model_name,
            "risk_factors": risk,
            "financial_highlights": highlights,
        }
