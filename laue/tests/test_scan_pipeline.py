"""Tests for the scan-level pipeline: the executor choice and the analysis_fn hook.

The process pool is the part worth pinning. Its worker has to survive a pickle
round trip, which is what forces the reader, the analysis function and the task
tuples to stay free of closures, h5py handles and the scan object — a constraint
that is easy to reintroduce by accident and only fails at run time, on a machine
that spawns rather than forks.
"""

from __future__ import annotations

import numpy as np
import pytest

from laue.scan_pipeline import (
    _RoiReader,
    analyse_stack,
    read_roi_stack,
    run_spot_pipeline,
)
from laue.spot_fit import fit_spot
from laue.spot_metrics import analyze_spot

NI, NJ = 4, 3
N_POS = NI * NJ
FRAME = 60


class StubScan:
    """The slice of the lauexplore Scan API the pipeline actually uses."""

    nbxpoints, nbypoints = NI, NJ
    length = N_POS

    def ij_to_index(self, i, j):
        return j * NI + i

    def ij_to_xy(self, i, j):
        return i * 1e-3, j * 1e-3


@pytest.fixture(scope="module")
def stack_h5(tmp_path_factory):
    """A scan whose sub-peak separation grows along i, so maps can be checked."""
    import h5py

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:FRAME, 0:FRAME].astype(float)
    frames = np.empty((N_POS, FRAME, FRAME), dtype=np.float32)
    for j in range(NJ):
        for i in range(NI):
            sep = 2.0 + 1.0 * i
            img = np.full((FRAME, FRAME), 80.0)
            for dx, amp in ((-sep / 2, 6000.0), (sep / 2, 3000.0)):
                img += amp * np.exp(
                    -((xx - (30 + dx)) ** 2 + (yy - 30) ** 2) / (2 * 2.0 ** 2)
                )
            frames[j * NI + i] = rng.poisson(img)

    path = tmp_path_factory.mktemp("scan") / "stack.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("frames", data=frames)
    return path


_RUN_KW = dict(roi_center=(31, 31), boxsize=10, coords="numpy", workers=2)


# ── Executor ──────────────────────────────────────────────────────────────────

def test_threads_and_processes_agree(stack_h5):
    """The executor is a performance choice; it must not change a single number."""
    scan = StubScan()
    a = run_spot_pipeline(stack_h5, scan, analysis_fn=fit_spot, executor="thread",
                          n_components=2, **_RUN_KW)
    b = run_spot_pipeline(stack_h5, scan, analysis_fn=fit_spot, executor="process",
                          n_components=2, **_RUN_KW)
    a = a.sort_values(["i", "j"]).reset_index(drop=True)
    b = b.sort_values(["i", "j"]).reset_index(drop=True)
    assert list(a.columns) == list(b.columns)
    for col in a.columns:
        if a[col].dtype.kind == "f":
            assert np.allclose(a[col], b[col], equal_nan=True), col


def test_auto_keeps_analyze_spot_on_threads(stack_h5, capsys):
    """The default analysis is read-bound; processes would only add start-up cost."""
    run_spot_pipeline(stack_h5, StubScan(), analysis_fn=analyze_spot, **_RUN_KW)
    assert "threads" in capsys.readouterr().out


def test_auto_moves_a_fit_to_processes(stack_h5, capsys):
    run_spot_pipeline(stack_h5, StubScan(), analysis_fn=fit_spot,
                      n_components=2, **_RUN_KW)
    assert "processes" in capsys.readouterr().out


def test_unknown_executor_is_rejected(stack_h5):
    with pytest.raises(ValueError, match="executor"):
        run_spot_pipeline(stack_h5, StubScan(), executor="fork", **_RUN_KW)


# ── The analysis_fn hook ──────────────────────────────────────────────────────

def test_default_analysis_is_unchanged(stack_h5):
    df = run_spot_pipeline(stack_h5, StubScan(), **_RUN_KW)
    assert len(df) == N_POS
    for col in ("aspect_ratio", "theta", "streak_D95", "n_local_maxima"):
        assert col in df.columns


def test_fit_columns_replace_the_morphology_ones(stack_h5):
    df = run_spot_pipeline(stack_h5, StubScan(), analysis_fn=fit_spot,
                           n_components=2, **_RUN_KW)
    assert "separation" in df.columns
    assert "aspect_ratio" not in df.columns


def test_the_fit_recovers_the_separation_built_into_the_scan(stack_h5):
    df = run_spot_pipeline(stack_h5, StubScan(), analysis_fn=fit_spot,
                           n_components=2, **_RUN_KW)
    got = df[df.status == "ok"].groupby("i")["separation"].median()
    for i in range(NI):
        assert got[i] == pytest.approx(2.0 + 1.0 * i, abs=0.5)


def test_spot_kwargs_reach_the_analysis_function(stack_h5):
    df = run_spot_pipeline(stack_h5, StubScan(), analysis_fn=fit_spot,
                           n_components="auto", n_max=2, executor="process",
                           **_RUN_KW)
    assert df["n_components"].max() <= 2


# ── Grid bookkeeping ──────────────────────────────────────────────────────────

def test_masked_positions_keep_the_grid_complete_through_processes(stack_h5):
    mask = np.ones(N_POS, dtype=bool)
    mask[2] = False
    df = run_spot_pipeline(stack_h5, StubScan(), analysis_fn=fit_spot,
                           n_components=2, executor="process", mask=mask, **_RUN_KW)
    assert len(df) == N_POS
    masked = df[df.status == "masked"]
    assert len(masked) == 1
    assert masked["separation"].isna().all()


def test_scan_subset_limits_the_positions(stack_h5):
    df = run_spot_pipeline(stack_h5, StubScan(), scan_subset=(1, 3, 0, 2), **_RUN_KW)
    assert len(df) == 4
    assert set(df["i"]) == {1, 2}


def test_rows_carry_the_scan_coordinates(stack_h5):
    df = run_spot_pipeline(stack_h5, StubScan(), **_RUN_KW)
    row = df[(df.i == 2) & (df.j == 1)].iloc[0]
    assert row["frame_idx"] == StubScan().ij_to_index(2, 1)
    assert row["x_um"] == pytest.approx(2.0)
    assert row["y_um"] == pytest.approx(1.0)


# ── Reader ────────────────────────────────────────────────────────────────────

def test_reader_survives_a_pickle_round_trip(stack_h5):
    """An open h5py handle cannot be pickled, so it must not be part of the state."""
    import pickle

    reader = _RoiReader(
        stack_h5, direct_mode=False, files=None, h5_img_key="frames",
        direct_h5_key="unused", squeeze=False,
        row_slice=slice(20, 41), col_slice=slice(20, 41), pad=None,
    )
    first = reader.read(0)                      # opens a handle
    revived = pickle.loads(pickle.dumps(reader))
    assert np.array_equal(revived.read(0), first)
    reader.close()
    revived.close()


def test_reader_pads_a_crop_that_runs_off_the_frame(stack_h5):
    """A ROI at the edge is zero-padded rather than silently returned smaller."""
    df = run_spot_pipeline(stack_h5, StubScan(), roi_center=(2, 2), boxsize=10,
                           coords="numpy", workers=2)
    assert len(df) == N_POS


# ── Stack-level path ──────────────────────────────────────────────────────────

def test_read_roi_stack_returns_one_crop_per_position(stack_h5):
    stack, index = read_roi_stack(stack_h5, StubScan(), roi_center=(31, 31),
                                  boxsize=10, coords="numpy", workers=2)
    assert stack.shape == (N_POS, 21, 21)
    assert list(index.columns) == ["i", "j", "frame_idx", "x_um", "y_um"]
    assert len(index) == N_POS


def test_the_stack_holds_the_same_pixels_the_streaming_pipeline_reads(stack_h5):
    """Both paths go through _prepare_reader, so a ROI must mean the same crop."""
    stack, index = read_roi_stack(stack_h5, StubScan(), roi_center=(31, 31),
                                  boxsize=10, coords="numpy", workers=2)
    streamed = run_spot_pipeline(stack_h5, StubScan(), analysis_fn=fit_spot,
                                 n_components=2, **_RUN_KW)
    from_stack = analyse_stack(stack, index, analysis_fn=fit_spot, n_components=2,
                               workers=2)
    a = streamed.sort_values(["i", "j"]).reset_index(drop=True)
    b = from_stack.sort_values(["i", "j"]).reset_index(drop=True)
    assert np.allclose(a["separation"], b["separation"], equal_nan=True)
    assert np.allclose(a["chi2"], b["chi2"], equal_nan=True)


def test_analyse_stack_carries_the_grid_coordinates(stack_h5):
    stack, index = read_roi_stack(stack_h5, StubScan(), roi_center=(31, 31),
                                  boxsize=10, coords="numpy", workers=2)
    df = analyse_stack(stack, index, analysis_fn=fit_spot, n_components=2, workers=2)
    assert {"i", "j", "x_um", "y_um"} <= set(df.columns)
    assert len(df) == N_POS


def test_analyse_stack_without_an_index_still_numbers_the_rows(stack_h5):
    stack, _ = read_roi_stack(stack_h5, StubScan(), roi_center=(31, 31),
                              boxsize=10, coords="numpy", workers=2)
    df = analyse_stack(stack, analysis_fn=fit_spot, n_components=2, workers=2)
    assert list(df["frame"]) == list(range(N_POS))


def test_analyse_stack_rejects_a_mismatched_index(stack_h5):
    stack, index = read_roi_stack(stack_h5, StubScan(), roi_center=(31, 31),
                                  boxsize=10, coords="numpy", workers=2)
    with pytest.raises(ValueError, match="index has"):
        analyse_stack(stack, index.iloc[:3], analysis_fn=fit_spot, workers=2)


def test_analyse_stack_rejects_a_single_roi(stack_h5):
    with pytest.raises(ValueError, match=r"\(n, h, w\)"):
        analyse_stack(np.zeros((20, 20)), analysis_fn=fit_spot)


def test_alignment_then_fit_is_the_intended_composition(stack_h5):
    """The workflow align_stack cannot be folded into the streaming pipeline."""
    from laue._imaging import align_stack

    stack, index = read_roi_stack(stack_h5, StubScan(), roi_center=(31, 31),
                                  boxsize=10, coords="numpy", workers=2)
    aligned, shifts = align_stack(stack, max_shift=3, crop_half=8)
    assert aligned.shape == (N_POS, 16, 16)
    assert shifts.shape == (N_POS, 2)

    df = analyse_stack(aligned, index, analysis_fn=fit_spot, n_components=2,
                       workers=2)
    assert len(df) == N_POS
    assert df["separation"].notna().any()
