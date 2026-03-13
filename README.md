# sec-insight

Automatically extract and summarize key information from SEC 10-K filings using `facebook/bart-large-cnn` via Hugging Face `transformers`.

Given a stock ticker, `sec-insight` fetches the latest annual report directly from SEC EDGAR and surfaces:
- **Risk Factors** (Item 1A) — condensed summary of material risks
- **Financial Highlights** (Item 7 / MD&A) — key revenue, margin, and cash flow takeaways

No API key needed. All models run locally.

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/yourusername/sec-insight.git
cd sec-insight
pip install -r requirements.txt

# Analyze Apple's latest 10-K
python main.py --ticker AAPL

# Save results to JSON
python main.py --ticker MSFT --output msft_results.json

# Use a local filing file (e.g. downloaded from EDGAR manually)
python main.py --ticker TSLA --file data/tsla_10k.txt

# Run on GPU
python main.py --ticker GOOGL --device 0
```

**Sample output:**
```
============================================================
  SEC INSIGHT RESULTS: AAPL
============================================================

RISK FACTORS SUMMARY (42,381 chars analyzed):
----------------------------------------
Apple faces intense competition across its product lines from companies with
greater resources. Supply chain disruptions and component shortages may impact
product availability. The company is subject to complex legal proceedings and
government investigations globally...

FINANCIAL HIGHLIGHTS SUMMARY (38,204 chars analyzed):
----------------------------------------
Net sales increased 8% to $394.3 billion. Services revenue reached a record
$85.2 billion. The company generated $99.6 billion in operating cash flow and
returned over $90 billion to shareholders...

============================================================
Model used: facebook/bart-large-cnn
============================================================
```

---

## Project Structure

```
sec-insight/
├── main.py                  # CLI entry point
├── src/
│   ├── analyzer.py          # Summarization pipeline (HF transformers)
│   └── fetcher.py           # SEC EDGAR data fetcher
├── notebooks/
│   └── demo.ipynb           # Interactive walkthrough
├── tests/
│   └── test_analyzer.py     # Unit tests (pytest)
├── data/                    # Drop local .txt filing files here
└── requirements.txt
```

---

## How It Works

1. **Fetch** — `EDGARFetcher` hits the public [SEC EDGAR submissions API](https://data.sec.gov/submissions/) to locate the most recent 10-K filing for a given ticker, then downloads the primary document.

2. **Parse** — Regex patterns extract standard 10-K sections (Item 1A for Risk Factors, Item 7 for MD&A) from the raw filing text.

3. **Summarize** — `SECAnalyzer` passes each section through `facebook/bart-large-cnn` via the Hugging Face `pipeline` API. Long sections are chunked to fit within BART's 1024-token context window, with partial summaries merged in a second pass.

4. **Output** — Results are printed to stdout or written to a JSON file.

---

## Model

Uses [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn) — a sequence-to-sequence model fine-tuned on CNN/DailyMail for abstractive summarization. Runs fully locally; no external API calls after the initial model download (~1.6GB).

To swap in a different summarization model:

```python
analyzer = SECAnalyzer(model_name="philschmid/bart-large-cnn-samsum")
```

---

## Running Tests

```bash
pip install pytest
pytest tests/
```

---

## Limitations

- EDGAR document structure varies across companies and years. The regex section extractors cover common patterns but may miss edge cases in older filings.
- BART's 1024-token context window means very long sections are summarized via hierarchical chunking, which can lose some detail.
- SEC EDGAR rate-limits requests. The fetcher includes a 0.5s delay between calls.

---

## License

MIT
