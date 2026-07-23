from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, cast

import numpy as np
import pandas as pd

BlockName = Literal["Z", "Y", "V", "M"]


@dataclass(frozen=True)
class MatrixBlock:
    """One labelled monetary table block with an explicit applicability mask."""

    name: BlockName
    values: pd.DataFrame
    row_dimension: str
    column_dimension: str
    unit: str | None = None
    applicability_mask: pd.DataFrame | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = self.values.copy().astype("float64")
        values.index = values.index.astype(str)
        values.columns = values.columns.astype(str)
        if values.index.has_duplicates:
            duplicates = values.index[values.index.duplicated()].astype(str).tolist()
            raise ValueError(f"{self.name} contains duplicate row labels: {duplicates[:20]!r}")
        if values.columns.has_duplicates:
            duplicates = values.columns[values.columns.duplicated()].astype(str).tolist()
            raise ValueError(f"{self.name} contains duplicate column labels: {duplicates[:20]!r}")

        if self.applicability_mask is None:
            mask = values.notna()
        else:
            mask = self.applicability_mask.copy()
            mask.index = mask.index.astype(str)
            mask.columns = mask.columns.astype(str)
            if not mask.index.equals(values.index) or not mask.columns.equals(values.columns):
                raise ValueError(f"{self.name} applicability mask labels must exactly match block labels")
            mask = mask.astype(bool)

        applicable = values.where(mask).to_numpy(dtype="float64")
        if not np.isfinite(applicable[~np.isnan(applicable)]).all():
            raise ValueError(f"{self.name} contains non-finite values in applicable cells")
        if bool(values.where(~mask).notna().to_numpy().any()):
            raise ValueError(f"{self.name} non-applicable cells must be represented by NaN")

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "applicability_mask", mask)
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True)
class CanonicalIOT:
    """Validated canonical IOT blocks before or after a split transformation.

    ``final_demand_columns`` is the explicit non-overlapping computational basis.
    Published controls may remain in ``Y`` but never enter output unless selected.
    ``A`` and ``x`` are derived quantities and are therefore not stored as source
    blocks in this container.
    """

    Z: MatrixBlock
    Y: MatrixBlock
    final_demand_columns: tuple[str, ...]
    V: MatrixBlock | None = None
    M: MatrixBlock | None = None
    flow_role: str | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.Z.name != "Z" or self.Y.name != "Y":
            raise ValueError("CanonicalIOT requires blocks named Z and Y")
        if self.V is not None and self.V.name != "V":
            raise ValueError("CanonicalIOT V block must be named V")
        if self.M is not None and self.M.name != "M":
            raise ValueError("CanonicalIOT M block must be named M")

        Z = self.Z.values
        Y = self.Y.values
        if not Z.index.equals(Z.columns):
            raise ValueError("Canonical Z must be square with identical row and column labels")
        if not Y.index.equals(Z.index):
            raise ValueError("Canonical Y rows must exactly match the Z product domain")

        selected = tuple(map(str, self.final_demand_columns))
        if not selected:
            raise ValueError("final_demand_columns must contain at least one column")
        selected_index = pd.Index(selected)
        if selected_index.has_duplicates:
            duplicates = selected_index[selected_index.duplicated()].astype(str).tolist()
            raise ValueError(f"final_demand_columns contains duplicates: {duplicates[:20]!r}")
        missing = [code for code in selected if code not in set(Y.columns.astype(str))]
        if missing:
            raise ValueError(f"Selected final-demand columns are absent from Y: {missing[:20]!r}")

        selected_values = Y.loc[:, list(selected)]
        selected_mask = cast(pd.DataFrame, self.Y.applicability_mask).loc[:, list(selected)]
        if not bool(selected_mask.to_numpy(dtype=bool).all()):
            raise ValueError("Selected final-demand cells must all be applicable")
        if not np.isfinite(selected_values.to_numpy(dtype="float64")).all():
            raise ValueError("Selected final-demand block must be finite")

        if self.V is not None and not self.V.values.columns.equals(Z.columns):
            raise ValueError("Canonical V columns must exactly match the Z product domain")
        if self.M is not None and not self.M.values.columns.equals(Y.columns):
            raise ValueError("Canonical M columns must exactly match the Y column domain")

        object.__setattr__(self, "final_demand_columns", selected)
        object.__setattr__(self, "provenance", dict(self.provenance))

    def computational_final_demand(self) -> pd.DataFrame:
        """Return the explicit non-overlapping final-demand block."""
        return self.Y.values.loc[:, list(self.final_demand_columns)].copy()

    def output(self) -> pd.Series:
        """Derive product output from Z plus the computational final-demand block."""
        output = self.Z.values.sum(axis=1).add(
            self.computational_final_demand().sum(axis=1), fill_value=0.0
        )
        output = output.astype("float64")
        output.name = "x"
        return output
