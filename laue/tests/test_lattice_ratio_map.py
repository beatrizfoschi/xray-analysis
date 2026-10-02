"""Tests for `lattice_ratio_map`: which color limits end up on each panel.

The figure itself is built by lauexplore's `tiles`; what is pinned here is the
precedence between zmin/zmax, ref ± span and zmid, and the relative-deviation
arithmetic — the parts where a wrong branch silently recenters the colorbar.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from laue.lattice_ratio_map import lattice_ratio_map

NX, NY = 5, 4
REF = [1.0, 1.6259]


def _make_dataset(seed=0):
    scan = SimpleNamespace(
        nbxpoints=NX, nbypoints=NY, length=NX * NY, is_linear=False,
        xpoints=np.arange(NX) * 1e-3, ypoints=np.arange(NY) * 1e-3,
        ij_to_index=lambda i, j: j * NX + i,
        index_to_ij=lambda k: (k % NX, k // NX),
        ij_to_xy=lambda i, j: (i, j),
    )
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        scan=scan,
        boa=1.0 + rng.normal(0, 1e-4, NX * NY),
        coa=1.626 + rng.normal(0, 5e-4, NX * NY),
    )


def _limits(fig):
    return [(t.zmin, t.zmax, t.zmid) for t in fig.data]


def test_ref_alone_sets_zmid():
    fig = lattice_ratio_map(_make_dataset(), ref=REF)
    assert _limits(fig) == [(None, None, 1.0), (None, None, 1.6259)]


def test_ref_with_span_sets_symmetric_limits():
    fig = lattice_ratio_map(_make_dataset(), ref=REF, span=[2e-4, 1e-3])
    (lo0, hi0, _), (lo1, hi1, _) = _limits(fig)
    assert (lo0, hi0) == pytest.approx((0.9998, 1.0002))
    assert (lo1, hi1) == pytest.approx((1.6249, 1.6269))


def test_relative_plots_deviation_in_1e4():
    ds = _make_dataset()
    fig = lattice_ratio_map(ds, ref=REF, relative=True, span=5)
    assert [(t.zmin, t.zmax) for t in fig.data] == [(-5.0, 5.0), (-5.0, 5.0)]
    expected = (ds.coa - REF[1]) / REF[1] * 1e4
    np.testing.assert_allclose(np.sort(fig.data[1].z.ravel()), np.sort(expected))
    assert fig.data[0].colorbar.title.text == "× 1e-4"


def test_explicit_limits_take_priority_per_panel():
    fig = lattice_ratio_map(
        _make_dataset(), ref=REF, span=1e-3, zmin=[None, 1.62], zmax=[None, 1.63]
    )
    (lo0, hi0, _), (lo1, hi1, _) = _limits(fig)
    assert (lo0, hi0) == pytest.approx((0.999, 1.001))   # ref ± span
    assert (lo1, hi1) == (1.62, 1.63)                    # explícito


def test_mask_turns_points_into_nan():
    mask = np.ones(NX * NY, dtype=bool)
    mask[0] = False
    fig = lattice_ratio_map(_make_dataset(), mask=mask)
    assert np.isnan(fig.data[1].z).sum() == 1


def test_relative_without_ref_raises():
    with pytest.raises(ValueError, match="requires ref"):
        lattice_ratio_map(_make_dataset(), relative=True)


def test_wrong_number_of_panel_values_raises():
    with pytest.raises(ValueError, match="one per panel"):
        lattice_ratio_map(_make_dataset(), ref=[1.0, 1.6, 2.0])
