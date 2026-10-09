# Certified historical price-factor regression evidence

`exiobase3_common_eur_2022_2012_2022.csv` freezes all 539 factors (49 EXIOBASE
regions, 2012--2022) from the previously certified common-2022-EUR panel.
Its exact CSV SHA-256 is
`11827838f4fd1594ec4a7410eb5ab30fb2135b8bdba4881625cac21f0698f8f2`.
This is numerical regression evidence, not an alternative price-source owner.
The public `build_exiobase3_eur_price_factors` remains the production owner.

`scripts/audit_exiobase_price_factor_rebase.py` checks the frozen panel against
both the older 2012--2022 UNSD query vintage and the extended query vintage,
with exact round-trip float equality for every old observation and its metadata.
It then checks the region-wise change-of-base identity and all 49 base identities.
The emitted JSON includes source provenance for all three panels. Revisions are
blocking findings; updating the fixture to make a new source response pass is
not an accepted rebase procedure.
