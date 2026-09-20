PYTHON ?= python3
PILOT_STATE ?= Vermont
RAW_DIR := data/raw/ucsd-google-local
INTERIM_DIR := data/interim
PROCESSED_DIR := data/processed

.PHONY: help pilot-check pilot-download pilot-filter pilot-extract pilot-summary pilot-qa pilot-all test

help:
	@echo "Targets:"
	@echo "  pilot-check     inspect remote file sizes"
	@echo "  pilot-download  download metadata and reviews for $(PILOT_STATE)"
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
