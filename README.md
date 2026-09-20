# UCSD Google Local government-service pipeline

Reproducible, streaming Python tools for working with the
[UCSD Google Local Data (2021)](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/)
without committing raw reviews or reviewer identifiers to Git.

The default pilot uses the complete Vermont state files. It deliberately does not
use the 10-core subset because k-core filtering removes low-review places and is
therefore unsuitable for studying public-office coverage and review sparsity.

## What the code does

1. checks remote archive sizes and downloads with resume support;
2. verifies gzip integrity, byte count, and SHA-256 provenance;
3. streams place metadata to identify government-service candidates;
4. streams reviews for selected `gmap_id` values while removing names and hashing
   reviewer IDs with a project-local HMAC key;
5. builds a deduplicated place-month descriptive panel; and
6. produces an aggregate QA report with no review text or reviewer identifiers.

The rule-retrieved candidates are not a validated government-facility universe.
Manual adjudication and official-roster recall checks are required before
substantive analysis.

## Quick start

Requires Python 3 and standard Unix tools (`bash`, `curl`, `gzip`, `shasum`). The
processing scripts themselves use only the Python standard library.

```bash
./scripts/download-ucsd-state.sh Vermont --check
./scripts/download-ucsd-state.sh Vermont --download
make pilot-all
make test
```

Raw and derived data are excluded by `.gitignore`. The downloader records local
provenance in `data/download-manifest.tsv`. Do not publish raw review text,
reviewer names, HMAC keys, or row-level restricted derivatives.

## Main files

- `scripts/download-ucsd-state.sh` — resumable, idempotent download and verification
- `scripts/01_filter_government_places.py` — candidate retrieval with match-field provenance
- `scripts/02_extract_reviews.py` — privacy-aware streaming extraction
- `scripts/03_summarize_place_month.py` — review-key deduplication and aggregation
- `scripts/04_pilot_qa.py` — aggregate data-quality report
- `scripts/tests/test_pipeline.py` — end-to-end smoke tests
- `config/government-place-taxonomy.csv` — versionable retrieval rules
- `simulation/anchor_transport.py` — small anchor-transport identification simulation

## Important scope limits

The source is a historical snapshot through September 2021. It cannot by itself
identify unposted encounters, pre-publication blocking, later deletion, or current
office performance. Review counts and ratings describe visible online traces, not
an unbiased sample of experienced service satisfaction.
