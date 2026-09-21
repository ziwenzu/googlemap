PYTHON ?= python3
PILOT_STATE ?= Vermont
RAW_DIR := data/raw/ucsd-google-local
INTERIM_DIR := data/interim
PROCESSED_DIR := data/processed

.PHONY: help pilot-check pilot-download national-check national-download national-filter national-qa official-check official-download official-normalize official-link-libraries official-all pilot-filter pilot-extract pilot-summary pilot-qa pilot-all test

help:
	@echo "Targets:"
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
