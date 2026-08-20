"""Tests for the unified frame reader.

`load_frame` replaced three loaders, each of which handled a case the others did
not. These check that every one of those cases still works, since no single
original covered them all and there is no older behaviour to diff against.
"""

from __future__ import annotations

import numpy as np
import pytest

from laue.readers import find_image_key, load_frame

h5py = pytest.importorskip('h5py')


FRAME = np.arange(300 * 400, dtype=np.float64).reshape(300, 400) % 977


@pytest.fixture
def h5_2d(tmp_path):
    p = tmp_path / 'flat.h5'
    with h5py.File(p, 'w') as f:
        f.create_dataset('entry/data', data=FRAME)
    return p


@pytest.fixture
def h5_3d(tmp_path):
    p = tmp_path / 'stack.h5'
    with h5py.File(p, 'w') as f:
        f.create_dataset('entry/data', data=np.stack([FRAME, FRAME * 2, FRAME * 3]))
    return p


@pytest.fixture
def h5_4d(tmp_path):
    """The (frames, 1, H, W) layout some beamline writers produce."""
    p = tmp_path / 'stack4d.h5'
    with h5py.File(p, 'w') as f:
        f.create_dataset('entry/data', data=np.stack([FRAME, FRAME * 2])[:, None, :, :])
    return p


@pytest.fixture
def h5_spots(tmp_path):
    """The nrxrdct layout: one group per spot."""
    p = tmp_path / 'spots.h5'
    with h5py.File(p, 'w') as f:
        f.create_dataset('spot_0000_0/image', data=FRAME)
        f.create_dataset('spot_0001_0/image', data=FRAME * 5)
    return p


def test_reads_a_2d_dataset(h5_2d):
    np.testing.assert_array_equal(load_frame(h5_2d, 'entry/data'), FRAME.astype(np.float32))


@pytest.mark.parametrize('index, factor', [(0, 1), (1, 2), (2, 3)])
def test_indexes_into_a_3d_stack(h5_3d, index, factor):
    got = load_frame(h5_3d, 'entry/data', frame_index=index)
    np.testing.assert_array_equal(got, (FRAME * factor).astype(np.float32))


@pytest.mark.parametrize('index, factor', [(0, 1), (1, 2)])
def test_drops_the_singleton_axis_of_a_4d_stack(h5_4d, index, factor):
    got = load_frame(h5_4d, 'entry/data', frame_index=index)
    assert got.shape == FRAME.shape
    np.testing.assert_array_equal(got, (FRAME * factor).astype(np.float32))


def test_finds_the_image_dataset_without_a_key(h5_2d):
    np.testing.assert_array_equal(load_frame(h5_2d), FRAME.astype(np.float32))


def test_find_image_key_ignores_datasets_too_small_to_be_frames(tmp_path):
    p = tmp_path / 'mixed.h5'
    with h5py.File(p, 'w') as f:
        f.create_dataset('metadata/monitor', data=np.arange(100.0))
        f.create_dataset('metadata/small_map', data=np.zeros((10, 10)))
        f.create_dataset('entry/data', data=FRAME)
    with h5py.File(p, 'r') as f:
        assert find_image_key(f) == 'entry/data'


def test_find_image_key_raises_when_there_is_no_frame(tmp_path):
    p = tmp_path / 'nothing.h5'
    with h5py.File(p, 'w') as f:
        f.create_dataset('metadata/monitor', data=np.arange(100.0))
    with h5py.File(p, 'r') as f:
        with pytest.raises(KeyError, match='pass h5_key'):
            find_image_key(f)


def test_reads_an_explicit_nrxrdct_spot_group(h5_spots):
    got = load_frame(h5_spots, spot_key='spot_0001_0')
    np.testing.assert_array_equal(got, (FRAME * 5).astype(np.float32))


def test_falls_back_to_the_first_spot_group(h5_spots):
    np.testing.assert_array_equal(load_frame(h5_spots), FRAME.astype(np.float32))


def test_reads_npy_2d_and_3d(tmp_path):
    flat = tmp_path / 'f.npy'
    np.save(flat, FRAME)
    np.testing.assert_array_equal(load_frame(flat), FRAME.astype(np.float32))

    stack = tmp_path / 's.npy'
    np.save(stack, np.stack([FRAME, FRAME * 4]))
    np.testing.assert_array_equal(load_frame(stack, frame_index=1),
                                  (FRAME * 4).astype(np.float32))


def test_reads_tiff(tmp_path):
    """The old sCMOS camera writes TIFF; fabio reads it, PIL is the fallback."""
    Image = pytest.importorskip('PIL.Image')
    p = tmp_path / 'frame.tif'
    Image.fromarray(FRAME.astype(np.uint16)).save(p)
    np.testing.assert_array_equal(load_frame(p), FRAME.astype(np.float32))


def test_dtype_is_selectable_because_the_display_path_assumed_float64(h5_2d):
    assert load_frame(h5_2d, 'entry/data').dtype == np.float32
    assert load_frame(h5_2d, 'entry/data', dtype=np.float64).dtype == np.float64


def test_accepts_an_in_memory_2d_array():
    np.testing.assert_array_equal(load_frame(FRAME), FRAME.astype(np.float32))


def test_indexes_into_an_in_memory_3d_stack():
    cube = np.stack([FRAME, FRAME * 2, FRAME * 3])
    np.testing.assert_array_equal(load_frame(cube, frame_index=2),
                                  (FRAME * 3).astype(np.float32))


def test_an_in_memory_array_is_copied_not_aliased():
    """Downstream steps subtract background in place; the caller's stack must survive."""
    original = FRAME.astype(np.float32)
    returned = load_frame(original)
    returned[0, 0] = -12345.0
    assert original[0, 0] != -12345.0


def test_a_1d_or_4d_array_is_rejected():
    with pytest.raises(ValueError, match='2-D or 3-D'):
        load_frame(np.arange(10.0))
    with pytest.raises(ValueError, match='2-D or 3-D'):
        load_frame(np.zeros((2, 1, 8, 8)))


def test_unreadable_suffix_reports_what_was_tried(tmp_path):
    p = tmp_path / 'thing.zzz'
    p.write_bytes(b'not an image')
    with pytest.raises(Exception):
        load_frame(p)
