"""Structural decomposition utilities for multi-year viability margin changes."""

from __future__ import annotations

import copy
import itertools
from typing import Any, Callable, Mapping, Sequence

import pandas as pd


def _clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, pd.Series):
            cloned[key] = value.copy(deep=True)
        elif isinstance(value, pd.DataFrame):
            cloned[key] = value.copy(deep=True)
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _assign_block(state: dict[str, Any], key: str, source: Mapping[str, Any]) -> None:
    value = source[key]
    if isinstance(value, pd.Series):
        state[key] = value.copy(deep=True)
    elif isinstance(value, pd.DataFrame):
        state[key] = value.copy(deep=True)
    else:
        state[key] = copy.deepcopy(value)


def _resolve_permutations(
    blocks: Sequence[str],
    *,
    max_permutations: int | None,
    random_seed: int,
) -> list[tuple[str, ...]]:
    all_perms = list(itertools.permutations(blocks))
    if max_permutations is None or max_permutations >= len(all_perms):
        return all_perms
    # deterministic down-sampling by seeded permutation index shuffling
    rng = pd.Series(range(len(all_perms))).sample(frac=1.0, random_state=int(random_seed)).tolist()
    chosen = [all_perms[i] for i in rng[: max_permutations]]
    return chosen


def _shapley_pair(
    state_prev: Mapping[str, Any],
    state_curr: Mapping[str, Any],
    viability_fn: Callable[[Mapping[str, Any]], float],
    blocks: Sequence[str],
    *,
    max_permutations: int | None,
    random_seed: int,
) -> dict[str, float]:
    perms = _resolve_permutations(blocks, max_permutations=max_permutations, random_seed=random_seed)
    contrib = {k: 0.0 for k in blocks}
    for perm in perms:
        running = _clone_state(state_prev)
        base_val = float(viability_fn(running))
        for key in perm:
            _assign_block(running, key, state_curr)
            next_val = float(viability_fn(running))
            contrib[key] += next_val - base_val
            base_val = next_val
    scale = 1.0 / float(len(perms))
    for key in contrib:
        contrib[key] *= scale
    return contrib


def decompose_slack_change(
    year_states: Mapping[int, Mapping[str, Any]],
    viability_fn: Callable[[Mapping[str, Any]], float],
    *,
    blocks: Sequence[str] = ("A", "y", "E", "S"),
    method: str = "shapley",
    random_seed: int = 0,
    max_permutations: int | None = None,
) -> pd.DataFrame:
    """Shapley/path-average decomposition over structural blocks.

    Parameters
    ----------
    year_states:
        Mapping year -> state dictionary. Each state should contain block keys
        declared in ``blocks``.
    viability_fn:
        Functional mapping state -> scalar viability metric.
    blocks:
        Blocks to decompose, defaulting to ``("A", "y", "E", "S")``.
    method:
        Only ``"shapley"`` is supported.
    random_seed:
        Seed for deterministic permutation subsampling.
    max_permutations:
        Optional cap on permutation count for large block sets.
    """
    if method != "shapley":
        raise ValueError("Only method='shapley' is supported.")
    years = sorted(year_states.keys())
    if len(years) < 2:
        return pd.DataFrame(
            columns=[
                "year_from",
                "year_to",
                "total_change",
                "residual",
                "n_permutations",
            ]
            + [f"{k}_component" for k in blocks]
        )

    rows: list[dict[str, float | int]] = []
    perms = _resolve_permutations(blocks, max_permutations=max_permutations, random_seed=random_seed)
    for i in range(1, len(years)):
        y0 = years[i - 1]
        y1 = years[i]
        s0 = year_states[y0]
        s1 = year_states[y1]
        missing0 = [k for k in blocks if k not in s0]
        missing1 = [k for k in blocks if k not in s1]
        if missing0 or missing1:
            raise ValueError(f"Missing decomposition blocks for {y0}->{y1}: prev={missing0}, curr={missing1}")

        base = float(viability_fn(s0))
        target = float(viability_fn(s1))
        contrib = _shapley_pair(
            s0,
            s1,
            viability_fn,
            blocks,
            max_permutations=max_permutations,
            random_seed=random_seed,
        )
        total_change = target - base
        comp_sum = sum(float(contrib[k]) for k in blocks)
        row: dict[str, float | int] = {
            "year_from": int(y0),
            "year_to": int(y1),
            "total_change": float(total_change),
            "residual": float(total_change - comp_sum),
            "n_permutations": int(len(perms)),
        }
        for key in blocks:
            row[f"{key}_component"] = float(contrib[key])
        rows.append(row)
    return pd.DataFrame(rows)
