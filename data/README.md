# Data

## Rules

Raw data are immutable, local-only, and ignored by Git. Each download must have a
row in `download-manifest.tsv`. Derived datasets preserve provenance fields and a
script version or Git commit.

## Layout

- `raw/ucsd-google-local/` — state-level `meta-STATE.json.gz` and
  `review-STATE.json.gz` archives from UCSD.
- `raw/review-lifecycle/` — dated full-review snapshots from future public-agency
  partners; never confuse this with the UCSD historical snapshot.
- `external/admin/` — administrative transaction counts, waiting times, staffing,
  service hours, probability surveys, and official facility frames.
- `external/official-rosters/` — source-specific government organization and
  facility lists used as the denominator and for Google-place linkage. This may be
  a symbolic link to external storage.
- `interim/` — candidate government places, adjudication sheets, and filtered
  review streams.
- `processed/` — analysis-ready office-by-month and service-by-region panels.

`raw/ucsd-google-local/` may be a symbolic link to a large external volume. Keep
the link at this exact project-relative path so all scripts continue to work; the
linked archives remain excluded from Git and should not be synchronized through
Dropbox.

After the complete metadata collection is present, run `make national-filter`.
This creates per-partition retrieval tables under
`interim/government-places-by-state/` and a combined
`interim/government-places-all.csv`. These are high-recall candidates requiring
validation, not an analysis-ready public-sector frame.

Run `make national-qa` after retrieval to create
`docs/national-candidate-screen-report.md` with duplicate, rule-family, match-field,
and manual-review diagnostics.

Run `make official-check` and `make official-download` to acquire the first
time-aligned official frames (Census 2021 government units, IMLS FY2021 public
library outlets, and GSA FY2021 federal real property). Source roles and release
constraints are recorded in `config/official-roster-sources.csv`; do not publish a
source marked `restricted`.

Run `make official-normalize` to create source-specific gzip-compressed CSV files
under `external/official-rosters/processed/`. The outputs share one schema but
retain `record_grain`, `evidence_role`, and `public_access_status`. In particular,
Census government units are organizations rather than offices, and GSA real
property assets do not establish public access. IMLS outlets are the first
facility-level frame that can validate a service-specific Google Maps subset.

Run `make official-link-libraries` for the first service-specific linkage. It
deduplicates Google candidates by `gmap_id`, links them to IMLS outlets using
name, address, ZIP, and geographic agreement, and preserves blank adjudication
and exclusion fields. An unmatched record remains unresolved rather than being
classified as non-government.

## Required provenance fields

Every processed row should be traceable to:

- source dataset and archive filename;
- `gmap_id` and review timestamp;
- extraction date and script;
- government-place classification rule and version;
- adjudication status;
- service family and government level;
- any external-frame match and match quality;
- any authenticity/coordination score with model version;
- any survival interval derived from repeated snapshots.

## Privacy

The raw UCSD files include user identifiers and names. Do not release those fields.
Use a project-specific salted hash for internal longitudinal diagnostics and drop
names from processed analytical files. Do not infer protected characteristics from
reviewer names or photographs.
