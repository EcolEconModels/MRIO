.PHONY: help at-2021-energy-green-brown

help:
	@echo "Targets:"
	@echo "  at-2021-energy-green-brown  Run the 3-step AT 2021 ENERGY_GREEN/BROWN pipeline"
	@echo ""
	@echo "Vars (optional): TEMPLATE_XLSX=..., OUT_SPLIT=..., OUT_AGG=..., EXIO_YEAR=..., EXIO_REGION=..., EUROSTAT_TIME=..., EUROSTAT_GEO=..."

at-2021-energy-green-brown:
	bash scripts/pipelines/at_2021_energy_green_brown.sh
