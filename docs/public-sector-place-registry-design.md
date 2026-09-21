# Public-sector place registry: frozen pilot design

## Objective

Build a high-recall, auditable registry of U.S. Google place listings that may
represent government organizations or public-service facilities. The registry is
a measurement instrument, not ground truth. It must preserve the difference
between a government organization, a physical structure, a public-facing service
office, and a Google listing.

The master registry is deliberately broader than the analysis sample. Downstream
models must use service-specific tiers rather than treating schools, hospitals,
parks, police, libraries, and DMV offices as exchangeable indicators.

## Units and keys

- Population unit: one unique UCSD Google Local `gmap_id`.
- Official unit: source-specific organization, outlet, asset, or structure ID.
- Linkage unit: one candidate `gmap_id`–official-record edge.
- Released lookup unit: one `gmap_id` with candidate evidence, model or rule
  score, adjudicated label, and exclusion reason.

No source is silently coerced to another record grain.

## Source hierarchy

| Source | Unit | Role | Historical limitation |
|---|---|---|---|
| Census Government Units Listing 2021 | Government organization | Organization backbone | Not an office list |
| IMLS Public Libraries Survey FY2021 | Library outlet | Time-aligned facility frame | Libraries only |
| GSA FRPP FY2021 | Federal property asset | Address discovery | Does not prove public access |
| USGS National Structures, current | Named structure point | High-recall reverse discovery | Not a 2021 denominator |
| Agency and state locators | Service office | Family-specific validation | Vintage varies |

Current sources can discover candidates and validate names. Historical coverage
claims require a time-aligned source or an explicit stability assumption.

## Candidate discovery

The candidate union has three components:

1. **Keyword/category retrieval:** the existing taxonomy scans every compressed
   state metadata file and retains interpretable rule evidence.
2. **Official reverse linkage:** each official facility searches the complete
   Google metadata frame through spatial blocking, name, address, and service
   family evidence. This can recover facilities that lack government keywords.
3. **Residual audit:** probability samples of rule/linkage negatives estimate
   missed prevalence and reveal new failure modes.

Absence from an official roster is not an exclusion. Presence in a structure or
property file is not proof that the listing is public-facing.

## USGS linkage v0.2

The current implementation:

- blocks official and Google records by state and 0.10-degree spatial cells;
- evaluates candidates within 3 km;
- uses name similarity, address agreement, distance, and family-specific text;
- requires service-family evidence for an automatic strong match;
- routes close but incomplete matches to `needs_review`;
- downgrades every many-Google-to-one-official collision;
- excludes records explicitly marked as privately owned, while retaining unknown
  ownership as review cases;
- leaves `classification_probability`, `adjudicated_label`, and
  `exclusion_reason` blank until separately estimated or coded.

The scalar `official_link_score` ranks candidates. It is not a calibrated
probability and must not be interpreted as one.

## Ownership and public access

The live USGS feature service returned no usable `ADMINTYPE` values in the
downloaded layers. Consequently:

- post offices retain official-type evidence but still require Google-listing
  validation;
- hospitals, schools, ambulance services, and other mixed-ownership types cannot
  be automatically called public;
- missing owner class is coded `needs_review_unknown_owner`;
- public access remains unknown for all structure records until place-level
  evidence is added.

This missingness is preserved rather than imputed from facility type.

## Validation design

The Vermont pilot draws 40 records from each of four strata:

1. confirmed official candidates;
2. official matches needing review;
3. keyword candidates without a USGS match;
4. general Google negatives.

The master retains stratum populations, selection probabilities, and inverse
probability weights. With one student available, reuse the existing 160 records
for one evidence-backed coding round, followed by 40 delayed blinded recodings,
10 per stratum. No primary sample is redrawn and no sampling weights change.

The student receives only the initial form, codebook, and Chinese instructions.
The investigator holds the sampling key and blank retest until the first round
is frozen. Schedule the repeat after 7-14 days without access to initial labels.
This practical interval cannot guarantee memory washout. New IDs and order reduce
copying but cannot eliminate recognition of place names.

Report within-coder repeat agreement, confusion tables, and, when defined, retest
kappa. These measure neither inter-coder reliability nor factual accuracy.
Record official evidence for ownership, access, and historical status; preserve
uncertain labels, both rounds, and separate final resolutions. An AI assistant
cannot serve as a second independent human coder.

Evaluate linkage correctness separately from public-sector status and analysis
eligibility. Report design-weighted performance against single-coder reference
labels with uncertainty. Forty ordinary negatives are a diagnostic sample, not
proof of completeness. With zero events in 40 independent binomial observations,
the one-sided 95% upper limit is 1 - 0.05^(1/40), approximately 7.2%; this is a
negative-stratum prevalence bound, not recall. Stratified finite-population
estimates require design-aware intervals. See
[NIST confidence intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).

The previous blanket kappa/precision/F1 gate is superseded. A small Vermont
development sample cannot certify national performance, and tier F1 requires a
frozen tier classifier that the entity linker does not provide. After development,
draw a fresh national evaluation sample stratified by source, service family,
geography, and negative-candidate risk. Allocate based on pilot errors/prevalence
and desired interval widths; retain a probability sample of ordinary negatives.

See [the student instructions](single-coder-student-guide.md). Historical Coder A/B
files remain preserved; they are not evidence of completed dual coding.

## Executed pilot status

As of 2026-09-20/21:

- 19 USGS layers were verified and downloaded;
- 261,399 structure records were checksum-logged and standardized;
- Vermont's 11,291 metadata records were scanned in full;
- linkage v0.2 produced 545 service-family links covering 463 unique Google
  places: 95 strong candidates and 450 review links;
- the official reverse search added 59 strong Google candidates not found by the
  prior keyword frame;
- a 160-record weighted, blinded validation sample was generated.

These are engineering and measurement results, not substantive estimates of
public-service satisfaction.

## Stage gates

1. Send the existing 160-place package to one student; retain retest and key.
2. Continue candidate discovery and official-source collection during manual work.
   All such outputs remain provisional candidates.
3. Review initial labels, delayed repeat stability, evidence, and uncertainty.
   Refine on development data and freeze versions before fresh evaluation.
4. Add DMV state adapters, then SSA, courts, IRS, VA, USPS, elections, human
   services, public health, and clerk sources.
5. Draw a fresh national evaluation sample and estimate performance with design
   weights and uncertainty before releasing an evidence-reviewed eligible lookup.
6. Only then extract reviews for adjudicated eligible places and construct
   service-specific outcome scales.
