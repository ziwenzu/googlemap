# Official roster linkage design

## Why there is no single government-office list

Official lists exist, but their units and coverage differ. The Census Government
Units Listing enumerates state and local government organizations, not service
windows. The IMLS Public Libraries Survey has outlet-level records and can serve
as a facility denominator for public libraries. The GSA Federal Real Property
Profile enumerates federal property assets, many of which are not open to the
public. Agency locators for USPS, Social Security, courts, motor vehicles, and
other services must therefore be added as separate source-specific frames.

The project treats the union of those frames as a versioned registry. It does not
infer public access merely because an organization or property appears in an
official file.

## Data layers

1. `config/official-roster-sources.csv` records the source, vintage, observational
   unit, release constraint, and intended evidentiary role.
2. `scripts/download-official-rosters.sh` downloads source archives, verifies that
   each archive opens, computes its SHA-256 digest, and records the transfer in
   `data/download-manifest.tsv`.
3. `scripts/06_normalize_official_rosters.py` writes source-specific compressed
   CSV files with one common schema. It preserves `record_grain`, `evidence_role`,
   and `public_access_status`.
4. `scripts/07_link_public_libraries.py` implements the first service-specific
   linkage between Google Maps candidates and official IMLS outlets.

Raw archives remain immutable. Normalized and linked outputs belong under the
external `data/external/official-rosters/processed/` directory and stay outside
Git.

## Linkage statuses

- `confirmed_official_candidate` means the Google candidate has strong automated
  agreement with one and only one IMLS outlet on name plus ZIP/geography, or on a
  conservative combination of address, distance, and name. It is still a
  candidate, not a human-adjudicated label.
- `needs_review` covers partial agreement and any many-to-one collision in which
  several Google place IDs point to the same official outlet.
- `not_linked` means no sufficiently agreeing IMLS record was found. It does not
  imply that the Google place is private, invalid, or outside the public sector.

The linkage output retains blank `adjudicated_label` and `exclusion_reason`
fields. Manual decisions should be written there or to a versioned adjudication
table rather than overwriting the automated scores.

## Extension rule

Add the next service family only after locating a facility-level source with a
documented vintage and release policy. Organization lists may help validate names
and government level, while asset lists may help discover addresses, but neither
is a substitute for a public-facing service-office frame. Each new source should
receive its own normalizer and source-specific linkage tests before its results
enter a combined public-sector dataset.
