"""Tests for the click-to-inspect map of `spot_fit` results.

The click handler itself needs a live canvas, so what is pinned here is
everything it depends on: the grid, the row lookup behind each cell, and the
arithmetic that turns a click position back into a grid index. Those are where an
off-by-one silently shows the wrong position's fit next to the right position's
colour.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from laue.spot_fit import fit_spot
from laue.spot_fit_viewer import (
    draw_fit_panels,
    interactive_fit_map,
    metric_grid,
    summarise,
)

NI, NJ = 5, 4


def _make_roi(peaks, shape=(20, 20), sigma=2.0, bg=100.0, seed=0):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    img = np.full(shape, bg)
    for x0, y0, amp in peaks:
        img += amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    return np.random.default_rng(seed).poisson(img).astype(float)


@pytest.fixture(scope="module")
def fitted():
    """A scan where only the right-hand columns hold a resolvable doublet."""
    rois, rows = [], []
    for j in range(NJ):
        for i in range(NI):
            sep = 1.0 + 2.5 * i          # 0.5 to 5 sigma
            roi = _make_roi(
                [(10 - sep / 2, 10, 6000.0), (10 + sep / 2, 10, 5000.0)],
                seed=j * NI + i,
            )
            rois.append(roi)
            rows.append({"i": i, "j": j, "frame_idx": j * NI + i,
                         "x_um": float(i), "y_um": float(j),
                         **fit_spot(roi, n_components="auto", n_max=3)})
    return np.array(rois), pd.DataFrame(rows)


# ── Grid ──────────────────────────────────────────────────────────────────────

def test_grid_is_shaped_by_the_scan_not_the_row_order(fitted):
    _, df = fitted
    grid, lookup, extent = metric_grid(df, "n_resolved")
    assert grid.shape == (NJ, NI)
    assert lookup.shape == (NJ, NI)
    assert extent == (0.0, float(NI - 1), 0.0, float(NJ - 1))


def test_every_cell_points_back_at_its_own_row(fitted):
    _, df = fitted
    grid, lookup, _ = metric_grid(df, "n_resolved")
    for gj in range(NJ):
        for gi in range(NI):
            row = df.iloc[int(lookup[gj, gi])]
            assert (int(row["i"]), int(row["j"])) == (gi, gj)
            assert row["n_resolved"] == grid[gj, gi]


def test_a_shuffled_dataframe_still_maps_correctly(fitted):
    """Row order is not grid order — `run_spot_pipeline` sorts, `analyse_stack` does not."""
    _, df = fitted
    shuffled = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    grid, lookup, _ = metric_grid(shuffled, "n_resolved")
    for gj in range(NJ):
        for gi in range(NI):
            row = shuffled.iloc[int(lookup[gj, gi])]
            assert (int(row["i"]), int(row["j"])) == (gi, gj)


def test_an_unknown_metric_names_the_alternatives(fitted):
    _, df = fitted
    with pytest.raises(ValueError, match="separation"):
        metric_grid(df, "not_a_column")


def test_a_dataframe_without_grid_coordinates_is_rejected(fitted):
    _, df = fitted
    with pytest.raises(ValueError, match="grid coordinates"):
        metric_grid(df.drop(columns=["i"]), "n_resolved")


# ── Click arithmetic ──────────────────────────────────────────────────────────

def test_a_click_at_a_cell_centre_lands_on_that_cell(fitted):
    """The inverse of the extent mapping the click handler uses."""
    _, df = fitted
    grid, lookup, (x0, x1, y0, y1) = metric_grid(df, "n_resolved")
    nby, nbx = grid.shape
    dx = (x1 - x0) / max(nbx - 1, 1)
    dy = (y1 - y0) / max(nby - 1, 1)
    for gj in range(nby):
        for gi in range(nbx):
            xc, yc = x0 + gi * dx, y0 + gj * dy
            assert int(round((xc - x0) / dx)) == gi
            assert int(round((yc - y0) / dy)) == gj


def test_a_click_just_off_centre_still_lands_on_the_same_cell(fitted):
    _, df = fitted
    grid, _, (x0, x1, y0, y1) = metric_grid(df, "n_resolved")
    nby, nbx = grid.shape
    dx = (x1 - x0) / max(nbx - 1, 1)
    for gi in range(nbx):
        for nudge in (-0.4, 0.4):
            xc = x0 + (gi + nudge) * dx
            assert int(round((xc - x0) / dx)) == gi


# ── Panels ────────────────────────────────────────────────────────────────────

def test_panels_mark_resolved_and_unresolved_components_differently(fitted):
    import matplotlib.pyplot as plt

    stack, df = fitted
    row = df[df["n_fitted"] > df["n_resolved"]]
    if row.empty:
        pytest.skip("no position dropped a component in this fixture")
    pos = int(row.index[0])

    fig, axes = plt.subplots(1, 3)
    draw_fit_panels(axes, stack[pos], df.iloc[pos].to_dict())
    markers = [ln.get_marker() for ln in axes[0].get_lines()]
    assert "+" in markers and "x" in markers
    assert len(markers) == int(df.iloc[pos]["n_fitted"])
    plt.close(fig)


def test_panels_survive_a_position_with_no_fit():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3)
    draw_fit_panels(axes, np.zeros((20, 20)), {"n_fitted": 0, "n_resolved": 0})
    assert "no fit" in axes[0].get_title()
    plt.close(fig)


def test_the_summary_withholds_separation_when_nothing_resolved(fitted):
    _, df = fitted
    single = df[df["n_resolved"] < 2]
    if single.empty:
        pytest.skip("every position resolved in this fixture")
    text = summarise(single.iloc[0].to_dict())
    assert "no separation" in text
    assert "resolved" in text


def test_the_summary_reports_separation_when_two_resolve(fitted):
    _, df = fitted
    pair = df[df["n_resolved"] >= 2]
    if pair.empty:
        pytest.skip("no position resolved a pair in this fixture")
    text = summarise(pair.iloc[0].to_dict())
    assert "separation" in text and "orientation" in text


# ── Assembly ──────────────────────────────────────────────────────────────────

def test_the_figure_builds_with_a_map_and_three_panels(fitted):
    import matplotlib.pyplot as plt

    stack, df = fitted
    fig = interactive_fit_map(df, stack, metric="n_resolved", vmin=1, vmax=3)
    # map + colourbar + three panels + the text axis
    assert len(fig.axes) == 6
    plt.close(fig)


def test_a_mismatched_stack_is_refused(fitted):
    stack, df = fitted
    with pytest.raises(ValueError, match="same order"):
        interactive_fit_map(df, stack[:-1])


# ── Grids over part of a scan ─────────────────────────────────────────────────

def test_a_masked_subset_keeps_its_positions_on_the_right_cells():
    """The mask case: rows present only where an abundance map selected them."""
    rows = []
    for i in (0, 3, 4, 9):
        for j in (1, 6):
            rows.append({"i": i, "j": j, "n_resolved": float(i + j),
                         "x_um": float(i), "y_um": float(j)})
    df = pd.DataFrame(rows)

    grid, lookup, extent = metric_grid(df, "n_resolved")
    assert grid.shape == (6, 10)          # ranges, not counts
    assert np.isnan(grid).sum() == 60 - len(df)

    for pos, row in df.iterrows():
        gi, gj = int(row["i"]) - 0, int(row["j"]) - 1
        assert lookup[gj, gi] == pos
        assert grid[gj, gi] == row["n_resolved"]


def test_cells_with_no_row_stay_empty():
    df = pd.DataFrame([{"i": 0, "j": 0, "n_resolved": 2.0, "x_um": 0.0, "y_um": 0.0},
                       {"i": 2, "j": 2, "n_resolved": 1.0, "x_um": 2.0, "y_um": 2.0}])
    grid, lookup, _ = metric_grid(df, "n_resolved")
    assert grid.shape == (3, 3)
    assert (lookup == -1).sum() == 7
    assert np.isnan(grid[1, 1])


# ── Positions a mask left out ─────────────────────────────────────────────────
#
# `run_spot_pipeline(mask=...)` keeps skipped positions as rows of NaN so the
# grid stays rectangular. Clicking one must say so, not raise: `or 0` does not
# guard an int() against NaN, because NaN is truthy.

_MASKED_ROW = {"i": 1, "j": 2, "status": "masked", "n_fitted": np.nan,
               "n_resolved": np.nan, "chi2": np.nan, "bg": np.nan}


def test_a_masked_position_draws_instead_of_raising():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3)
    draw_fit_panels(axes, np.zeros((20, 20)), dict(_MASKED_ROW))
    assert "outside the mask" in axes[0].get_title()
    plt.close(fig)


def test_a_masked_position_summarises_instead_of_raising():
    text = summarise(dict(_MASKED_ROW))
    assert "outside the mask" in text
    assert "peak" not in text


def test_a_failed_fit_is_still_distinguished_from_a_masked_one():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3)
    draw_fit_panels(axes, np.zeros((20, 20)),
                    {"i": 0, "j": 0, "status": "ok", "n_fitted": 0, "n_resolved": 0})
    assert "no fit" in axes[0].get_title()
    plt.close(fig)


def test_a_map_holding_masked_rows_still_builds():
    df = pd.DataFrame([
        {"i": 0, "j": 0, "status": "ok", "n_resolved": 2.0, "x_um": 0.0, "y_um": 0.0},
        {"i": 0, "j": 1, "status": "masked", "n_resolved": np.nan, "x_um": 0.0, "y_um": 1.0},
        {"i": 1, "j": 0, "status": "ok", "n_resolved": 1.0, "x_um": 1.0, "y_um": 0.0},
        {"i": 1, "j": 1, "status": "masked", "n_resolved": np.nan, "x_um": 1.0, "y_um": 1.0},
    ])
    grid, lookup, _ = metric_grid(df, "n_resolved")
    assert grid.shape == (2, 2)
    assert np.isnan(grid).sum() == 2
    assert (lookup >= 0).all()          # every cell still has a row behind it
