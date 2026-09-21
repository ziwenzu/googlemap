# UCSD Google Local government-service pipeline

Reproducible, streaming Python tools for working with the
[UCSD Google Local Data (2021)](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/)
without committing raw reviews or reviewer identifiers to Git.

The default pilot uses the complete Vermont state files. It deliberately does not
use the 10-core subset because k-core filtering removes low-review places and is
therefore unsuitable for studying public-office coverage and review sparsity.

## What the code does

1. checks remote archive sizes and downloads one state or all 52 U.S. partitions
   with resume support;
2. verifies archive integrity, byte count, and SHA-256 provenance;
3. streams place metadata to identify government-service candidates;
4. streams reviews for selected `gmap_id` values while removing names and hashing
   reviewer IDs with a project-local HMAC key;
5. builds a deduplicated place-month descriptive panel;
6. downloads and normalizes source-specific official government rosters; and
7. links public-library candidates to official IMLS outlets while retaining
   unresolved and many-to-one cases for review;
8. reverse-links current USGS government and public-service structures against
   complete state metadata rather than only keyword candidates; and
9. draws weighted, blinded validation samples for two independent coders.

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

For the complete U.S. collection (50 states, DC, and the publisher's `Other`
partition), point `data/raw/ucsd-google-local` to a large external volume if
needed, then run:

```bash
make national-check
make national-download
make national-filter
make national-qa
```

`national-filter` creates a high-recall candidate table; `national-qa` reports
duplicates, rule-family composition, match fields, and manual-review flags. These
outputs remain retrieval candidates, not confirmed government facilities.

## Official roster validation

There is no single national facility-level list covering every U.S. public
service. This project builds a versioned registry from sources with different
observational units:

- Census Government Units Listing: state/local government organizations;
- IMLS Public Libraries Survey: public-library service outlets; and
- GSA Federal Real Property Profile: federal assets, not necessarily publicly
  accessible offices.

Download and standardize the initial sources, then run the first
service-specific linkage:

```bash
make official-check
make official-download
make official-normalize
make official-link-libraries
```

Add the current USGS National Structures discovery frame and run the gated
Vermont pilot:

```bash
make official-usgs-check
make official-usgs-download
make official-usgs-normalize
make official-usgs-link-pilot
make official-usgs-validation-pilot
```

The current USGS source is discovery evidence, not a time-aligned 2021
denominator or proof of public ownership/access. The national target refuses to
run until the blinded pilot has been double-coded and a v1 threshold file has
been frozen.

The common schema preserves `record_grain`, `evidence_role`, and
`public_access_status`. Library linkage uses name, address, ZIP, and geographic
agreement. Multiple Google places pointing to one official outlet are sent to
`needs_review`; unmatched records remain unresolved rather than being called
non-government. See
[`docs/official-roster-linkage-design.md`](docs/official-roster-linkage-design.md).

Raw and derived data are excluded by `.gitignore`. The downloader records local
provenance in `data/download-manifest.tsv`. Do not publish raw review text,
reviewer names, HMAC keys, or row-level restricted derivatives.

## Main files

- `scripts/download-ucsd-state.sh` — resumable state download and verification
- `scripts/download-ucsd-all.sh` — complete 52-partition download/check wrapper
- `scripts/filter-ucsd-all-government-places.sh` — nationwide streaming retrieval
- `scripts/01_filter_government_places.py` — candidate retrieval with provenance
- `scripts/02_extract_reviews.py` — privacy-aware streaming extraction
- `scripts/03_summarize_place_month.py` — review-key deduplication and aggregation
- `scripts/04_pilot_qa.py` — aggregate data-quality report
- `scripts/05_audit_national_candidates.py` — nationwide candidate audit
- `scripts/download-official-rosters.sh` — official source download and verification
- `scripts/06_normalize_official_rosters.py` — grain-aware roster normalization
- `scripts/07_link_public_libraries.py` — conservative IMLS candidate linkage
- `scripts/08_download_usgs_structures.py` — paginated, checksum-logged USGS download
- `scripts/09_normalize_usgs_structures.py` — ownership-aware structure normalization
- `scripts/10_link_usgs_to_google.py` — resumable official-to-Google reverse linkage
- `scripts/11_draw_place_validation_sample.py` — weighted blinded validation sample
- `config/official-roster-sources.csv` — source roles and release constraints
- `config/usgs-structure-layers.csv` — selected layers and analytic-tier policy
- `scripts/tests/test_pipeline.py` — smoke and linkage tests

## Important scope limits

The UCSD source is a historical snapshot through September 2021. It cannot by
itself identify unposted encounters, pre-publication blocking, later deletion, or
current office performance. Review counts and ratings describe visible online
traces, not an unbiased sample of experienced service satisfaction.

An official organization or real-property record is not evidence of public
access. Even a strong automated facility link remains a candidate until the
project's adjudication policy is applied.
