.PHONY: help mappings-test at-2021-energy-green-brown

PYTHON ?= $(CURDIR)/.venv/bin/python
MAP_GEO ?= AT
MAP_YEAR ?= 2020

help:
	@echo "Targets:"
	@echo "  mappings-test                Run the focused mapping test suite"
	@echo "  at-2021-energy-green-brown  Run the 3-step AT 2021 ENERGY_GREEN/BROWN pipeline"
	@echo ""
	@echo "Vars (optional): MAP_GEO=$(MAP_GEO), MAP_YEAR=$(MAP_YEAR), TEMPLATE_XLSX=..., OUT_SPLIT=..., OUT_AGG=..., EXIO_YEAR=..., EXIO_REGION=..., EUROSTAT_TIME=..., EUROSTAT_GEO=..."

mappings-test:
	$(PYTHON) -m pytest tests/test_mapping_csv.py tests/test_exio_to_eurostat_mapping.py tests/test_diagnostic_concordance.py tests/test_iot_aggregation.py tests/test_split_eurostat_energy_by_exiobase.py

at-2021-energy-green-brown:
	bash scripts/pipelines/at_2021_energy_green_brown.sh
