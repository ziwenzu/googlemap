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
- `interim/` — candidate government places, adjudication sheets, and filtered
  review streams.
- `processed/` — analysis-ready office-by-month and service-by-region panels.

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

