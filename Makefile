PYTHON ?= python3
.DEFAULT_GOAL := help
PILOT_STATE ?= Vermont
RAW_DIR := data/raw/ucsd-google-local
INTERIM_DIR := data/interim
PROCESSED_DIR := data/processed
OFFICIAL_ROOT := data/external/official-rosters
USGS_RAW_DIR := $(OFFICIAL_ROOT)/raw/usgs-national-structures
USGS_PROCESSED := $(OFFICIAL_ROOT)/processed/usgs-national-structures-current.csv.gz
USGS_LINK_DIR := $(OFFICIAL_ROOT)/processed/usgs-google-linkage-by-state
USGS_VALIDATION_DIR := $(OFFICIAL_ROOT)/validation
SINGLE_CODER_MASTER ?= $(USGS_VALIDATION_DIR)/$(PILOT_STATE)-place-validation-master.csv
SINGLE_CODER_PACKAGE ?= $(USGS_VALIDATION_DIR)/single-coder-v1
SERVICE_OUTPUT ?= $(PROCESSED_DIR)/public-service-keywords-v1
SERVICE_V2_OUTPUT ?= $(PROCESSED_DIR)/public-service-keywords-v2
SERVICE_V2_REFERENCES ?= data/external/public-service-v2/2026-09-22

.PHONY: service-v2-references service-v2-national service-v2-check

service-v2-references:
	$(PYTHON) scripts/14_fetch_service_references.py \
		--output-dir "$(SERVICE_V2_REFERENCES)" \
		--manifest data/download-manifest.tsv

service-v2-national:
	$(PYTHON) scripts/15_extract_public_service_v2.py \
		--metadata-dir "$(RAW_DIR)" \
		--rules config/public-service-keywords-v2.json \
		--official "$(USGS_PROCESSED)" "$(SERVICE_V2_REFERENCES)/hud-pha-current.csv" \
		--previous "$(PROCESSED_DIR)/public-service-keywords-v1/places.csv" \
		--output-dir "$(SERVICE_V2_OUTPUT)"

service-v2-check:
	$(PYTHON) scripts/16_validate_service_v2.py \
		--output-dir "$(SERVICE_V2_OUTPUT)" \
		--previous "$(PROCESSED_DIR)/public-service-keywords-v1/places.csv"

.PHONY: service-keywords-national service-keywords-pilot

service-keywords-national:
	$(PYTHON) scripts/13_extract_public_service_places.py \
		--metadata-dir "$(RAW_DIR)" \
		--rules config/public-service-keywords-v1.json \
		--output-dir "$(SERVICE_OUTPUT)"

service-keywords-pilot:
	$(PYTHON) scripts/13_extract_public_service_places.py \
		--metadata-dir "$(RAW_DIR)" \
		--rules config/public-service-keywords-v1.json \
		--output-dir "$(SERVICE_OUTPUT)-$(PILOT_STATE)" \
		--partitions "$(PILOT_STATE)"

.PHONY: official-single-coder-package

.PHONY: help pilot-check pilot-download national-check national-download national-filter national-qa official-check official-download official-normalize official-link-libraries official-all official-usgs-check official-usgs-download official-usgs-normalize official-usgs-link-pilot official-usgs-validation-pilot official-usgs-link-national pilot-filter pilot-extract pilot-summary pilot-qa pilot-all test

help:
	@echo "Targets:"
	@echo "  service-v2-national nationwide 12-family extraction with official-location linkage"
	@echo "  service-v2-references fetch current HUD PHA / all-state DMV agency references"
	@echo "  service-v2-check validate v2 artifacts and compare with v1"
	@echo "  service-keywords-national automatic service-office extraction; no human coding"
	@echo "  service-keywords-pilot automatic extraction for $(PILOT_STATE)"
	@echo "  pilot-check     inspect remote file sizes"
	@echo "  pilot-download  download metadata and reviews for $(PILOT_STATE)"
	@echo "  national-check  inspect every complete U.S. partition"
	@echo "  national-download download every complete U.S. partition"
	@echo "  national-filter retrieve public-sector candidates from all metadata"
	@echo "  national-qa     audit the combined nationwide candidate table"
	@echo "  official-check  inspect time-aligned official roster archives"
	@echo "  official-download download and verify initial official rosters"
	@echo "  official-normalize standardize official rosters without mixing record grains"
	@echo "  official-link-libraries link Google candidates to official IMLS outlets"
	@echo "  official-all    download, verify, and normalize official rosters"
	@echo "  official-usgs-check inspect configured USGS structure layers"
	@echo "  official-usgs-download download and checksum-log USGS structure layers"
	@echo "  official-usgs-normalize standardize USGS features with owner uncertainty"
	@echo "  official-usgs-link-pilot reverse-link complete $(PILOT_STATE) metadata"
	@echo "  official-usgs-validation-pilot draw blinded weighted coder files"
	@echo "  official-single-coder-package prepare student forms and a held-back retest"
	@echo "  official-usgs-link-national generate provisional national candidates"
	@echo "  pilot-filter    identify candidate government places"
	@echo "  pilot-extract   stream reviews for candidate places"
	@echo "  pilot-summary   build place-month descriptive panel"
	@echo "  pilot-qa        generate a privacy-preserving pilot QA report"
	@echo "  pilot-all       run all pilot stages after download"
	@echo "  test            run lightweight unit tests"

pilot-check:
	./scripts/download-ucsd-state.sh "$(PILOT_STATE)" --check

pilot-download:
	./scripts/download-ucsd-state.sh "$(PILOT_STATE)" --download

national-check:
	./scripts/download-ucsd-all.sh --check

national-download:
	./scripts/download-ucsd-all.sh --download

national-filter:
	./scripts/filter-ucsd-all-government-places.sh

national-qa:
	$(PYTHON) scripts/05_audit_national_candidates.py \
		--places "$(INTERIM_DIR)/government-places-all.csv" \
		--output "docs/national-candidate-screen-report.md"

official-check:
	./scripts/download-official-rosters.sh --check

official-download:
	./scripts/download-official-rosters.sh --download

official-normalize:
	$(PYTHON) scripts/06_normalize_official_rosters.py \
		--raw-dir "data/external/official-rosters/raw" \
		--output-dir "data/external/official-rosters/processed"

official-link-libraries:
	$(PYTHON) scripts/07_link_public_libraries.py \
		--places "$(INTERIM_DIR)/government-places-all.csv" \
		--official "data/external/official-rosters/processed/imls-public-library-outlets-fy2021.csv.gz" \
		--output "data/external/official-rosters/processed/public-library-google-linkage-2021.csv.gz" \
		--summary "data/external/official-rosters/processed/public-library-google-linkage-summary-2021.json"

official-all: official-download official-normalize official-link-libraries

official-usgs-check:
	$(PYTHON) scripts/08_download_usgs_structures.py \
		--layers-config config/usgs-structure-layers.csv \
		--output-dir "$(USGS_RAW_DIR)" \
		--check

official-usgs-download:
	$(PYTHON) scripts/08_download_usgs_structures.py \
		--layers-config config/usgs-structure-layers.csv \
		--output-dir "$(USGS_RAW_DIR)" \
		--manifest data/download-manifest.tsv \
		--summary "$(USGS_RAW_DIR)/download-summary.json"

official-usgs-normalize:
	$(PYTHON) scripts/09_normalize_usgs_structures.py \
		--layers-config config/usgs-structure-layers.csv \
		--raw-dir "$(USGS_RAW_DIR)" \
		--output "$(USGS_PROCESSED)" \
		--summary "$(OFFICIAL_ROOT)/processed/usgs-national-structures-current-summary.json"

official-usgs-link-pilot:
	$(PYTHON) scripts/10_link_usgs_to_google.py \
		--metadata-dir "$(RAW_DIR)" \
		--official "$(USGS_PROCESSED)" \
		--output-dir "$(USGS_LINK_DIR)" \
		--combined-output "$(OFFICIAL_ROOT)/processed/usgs-google-linkage-$(PILOT_STATE).csv.gz" \
		--summary "$(OFFICIAL_ROOT)/processed/usgs-google-linkage-$(PILOT_STATE)-summary.json" \
		--partitions "$(PILOT_STATE)"

official-usgs-validation-pilot:
	$(PYTHON) scripts/11_draw_place_validation_sample.py \
		--linkage "$(OFFICIAL_ROOT)/processed/usgs-google-linkage-$(PILOT_STATE).csv.gz" \
		--rule-candidates "$(INTERIM_DIR)/government-places-all.csv" \
		--metadata-dir "$(RAW_DIR)" \
		--partitions "$(PILOT_STATE)" \
		--per-stratum 40 \
		--master-output "$(USGS_VALIDATION_DIR)/$(PILOT_STATE)-place-validation-master.csv" \
		--coder-a-output "$(USGS_VALIDATION_DIR)/$(PILOT_STATE)-place-validation-coder-a.csv" \
		--coder-b-output "$(USGS_VALIDATION_DIR)/$(PILOT_STATE)-place-validation-coder-b.csv" \
		--summary "$(USGS_VALIDATION_DIR)/$(PILOT_STATE)-place-validation-summary.json"

official-single-coder-package:
	$(PYTHON) scripts/12_prepare_single_coder_validation.py \
		--master "$(SINGLE_CODER_MASTER)" \
		--output-dir "$(SINGLE_CODER_PACKAGE)" \
		--codebook config/classification-codebook.md \
		--instructions docs/single-coder-student-guide.md

# Candidate discovery does not confer validated or analysis-ready status.
official-usgs-link-national:
	$(PYTHON) scripts/10_link_usgs_to_google.py \
		--metadata-dir "$(RAW_DIR)" \
		--official "$(USGS_PROCESSED)" \
		--output-dir "$(USGS_LINK_DIR)" \
		--combined-output "$(OFFICIAL_ROOT)/processed/usgs-google-linkage-national.csv.gz" \
		--summary "$(OFFICIAL_ROOT)/processed/usgs-google-linkage-national-summary.json"

pilot-filter:
	$(PYTHON) scripts/01_filter_government_places.py \
		--metadata "$(RAW_DIR)/meta-$(PILOT_STATE).json.gz" \
		--taxonomy config/government-place-taxonomy.csv \
		--output "$(INTERIM_DIR)/government-places-$(PILOT_STATE).csv"

pilot-extract:
	$(PYTHON) scripts/02_extract_reviews.py \
		--reviews "$(RAW_DIR)/review-$(PILOT_STATE).json.gz" \
		--places "$(INTERIM_DIR)/government-places-$(PILOT_STATE).csv" \
		--place-mode all-candidates \
		--output "$(INTERIM_DIR)/government-reviews-$(PILOT_STATE).jsonl.gz"

pilot-summary:
	$(PYTHON) scripts/03_summarize_place_month.py \
		--reviews "$(INTERIM_DIR)/government-reviews-$(PILOT_STATE).jsonl.gz" \
		--places "$(INTERIM_DIR)/government-places-$(PILOT_STATE).csv" \
		--deduplicate review-key \
		--output "$(PROCESSED_DIR)/place-month-$(PILOT_STATE).csv"

pilot-qa:
	$(PYTHON) scripts/04_pilot_qa.py \
		--state "$(PILOT_STATE)" \
		--reviews "$(INTERIM_DIR)/government-reviews-$(PILOT_STATE).jsonl.gz" \
		--places "$(INTERIM_DIR)/government-places-$(PILOT_STATE).csv" \
		--panel "$(PROCESSED_DIR)/place-month-$(PILOT_STATE).csv" \
		--raw-metadata "$(RAW_DIR)/meta-$(PILOT_STATE).json.gz" \
		--raw-reviews "$(RAW_DIR)/review-$(PILOT_STATE).json.gz" \
		--output "docs/$(PILOT_STATE)-pilot-report.md"

pilot-all: pilot-filter pilot-extract pilot-summary pilot-qa

test:
	$(PYTHON) -m unittest discover -s scripts/tests -p 'test_*.py'
