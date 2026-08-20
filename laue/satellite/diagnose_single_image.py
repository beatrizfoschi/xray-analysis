#!/usr/bin/env python3
"""Diagnostic script for satellite peak detection — validate on a single spot image.

This is the REQUIRED FIRST deliverable.  Run it on one representative spot image
to visually confirm that satellites are detected correctly before running any
batch processing.

Usage
-----
# Generate a synthetic test image (no real data needed):
  python -m laue.satellite.diagnose_single_image

# Load from an nrxrdct HDF5 spot file:
  python -m laue.satellite.diagnose_single_image frame_00001.h5 --spot-key spot_0000_0

# Load from a numpy array:
  python -m laue.satellite.diagnose_single_image spot.npy

Tuning flags (adjust until detection looks correct, then use the same values
in scan_pipeline.py):
  --axis-angle DEGREES   Fix satellite axis direction (auto-detect if omitted)
  --n-max INT            Maximum satellite order to accept (default: 3)
  --prominence FLOAT     Peak prominence threshold, 0–1 fraction (default: 0.05)
  --strip-width FLOAT    Profile strip width in pixels (default: 5)
  --bg-sigma FLOAT       Background subtraction sigma in pixels (default: 20)
  --save PATH            Save figure to file instead of (or in addition to) showing it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from laue.satellite.detection import detect_satellites, make_synthetic_image


from laue.satellite._orders import order_color as _order_color


# ── Image loading ─────────────────────────────────────────────────────────────

from laue.readers import load_frame as load_image


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(
    image: np.ndarray,
    result: dict,
    save_path: str | None = None,
) -> None:
    """Three-panel diagnostic figure.

    Panel 1 — raw image (log scale)
    Panel 2 — background-subtracted image with detected satellite markers
    Panel 3 — 1-D profile along the satellite axis with fitted Gaussians
    """
    peaks = result['peaks']
    distances, intensities = result['profile']
    axis_angle = result['axis_angle']
    sl0_center = result['sl0_center']
    image_sub = result['image_sub']

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(
        f'Satellite peak detection  |  axis = {axis_angle:.1f}°',
        fontsize=13, fontweight='bold',
    )

    # ── Panel 1: raw image ────────────────────────────────────────────────────
    ax = axes[0]
    vmin = max(float(image[image > 0].min()) if (image > 0).any() else 1.0, 1.0)
    ax.imshow(image, norm=LogNorm(vmin=vmin, vmax=float(image.max())),
              cmap='inferno', origin='lower')
    ax.set_title('Raw image (log scale)')
    ax.set_xlabel('col (px)')
    ax.set_ylabel('row (px)')

    # ── Panel 2: subtracted + overlay ────────────────────────────────────────
    ax = axes[1]
    pos_sub = image_sub[image_sub > 0]
    vmin_sub = max(float(pos_sub.min()) if len(pos_sub) > 0 else 1.0, 1.0)
    ax.imshow(image_sub, norm=LogNorm(vmin=vmin_sub, vmax=float(image_sub.max())),
              cmap='inferno', origin='lower')
    ax.set_title('Background-subtracted + detected peaks')
    ax.set_xlabel('col (px)')

    # Satellite axis line
    rad = np.radians(axis_angle)
    half = float(min(image.shape)) * 0.5
    cx_px, cy_px = sl0_center[1], sl0_center[0]
    ax.plot(
        [cx_px - half * np.cos(rad), cx_px + half * np.cos(rad)],
        [cy_px - half * np.sin(rad), cy_px + half * np.sin(rad)],
        '--', color='white', alpha=0.5, lw=1,
    )

    for pk in peaks:
        r, c = pk['position_2d']
        col = _order_color(pk['order'])
        ax.plot(c, r, 'o', color=col, ms=11, mew=2.0, fillstyle='none')
        label = f"SL{pk['order']:+d}"
        ax.text(c + 4, r + 4, label, color=col, fontsize=8, fontweight='bold')

    # ── Panel 3: 1-D profile + Gaussians ─────────────────────────────────────
    ax = axes[2]
    ax.plot(distances, intensities, color='#333333', lw=1.5, label='profile', zorder=2)

    for pk in peaks:
        col = _order_color(pk['order'])
        s = pk['pos_along_axis']
        sigma = pk['sigma']
        x_fit = np.linspace(s - 4 * sigma, s + 4 * sigma, 200)
        near = np.abs(distances - s) < sigma * 5
        bkg = float(intensities[near].min()) if near.any() else 0.0
        y_fit = pk['amplitude'] * np.exp(-0.5 * ((x_fit - s) / sigma) ** 2) + bkg
        ax.fill_between(x_fit, bkg, y_fit, alpha=0.25, color=col)
        ax.axvline(s, color=col, lw=1.5, alpha=0.85)
        ax.text(
            s, intensities.max() * 1.02,
            f" SL{pk['order']:+d}",
            color=col, fontsize=8, ha='center', va='bottom',
        )

    ax.set_xlabel('Distance from SL₀ (px)')
    ax.set_ylabel('Summed intensity (counts)')
    ax.set_title('1-D profile along satellite axis')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'[INFO] Figure saved -> {save_path}')
    plt.show()


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(result: dict) -> None:
    peaks = result['peaks']
    w = 68
    sep = '=' * w
    print('\n' + sep)
    print('  SATELLITE PEAK DETECTION - SUMMARY')
    print(sep)
    sl0_r, sl0_c = result['sl0_center']
    print(f"  Axis angle   : {result['axis_angle']:.2f} deg  (from +x in image frame)")
    print(f"  SL0 centroid : row = {sl0_r:.1f},  col = {sl0_c:.1f}  (pixels)")
    print(f"  Peaks found  : {len(peaks)}")

    if not peaks:
        print('\n  No peaks detected.')
        print('  Try: lowering --prominence, widening --strip-width, or')
        print('       supplying --axis-angle if auto-detection is unreliable.')
        print(sep + '\n')
        return

    # Spacing estimate from linear fit of position vs order
    orders_arr = np.array([p['order'] for p in peaks])
    pos_arr = np.array([p['pos_along_axis'] for p in peaks])
    nonzero = orders_arr != 0
    if nonzero.sum() >= 2:
        spacing = float(np.polyfit(orders_arr[nonzero], pos_arr[nonzero], 1)[0])
        print(f"  Delta-q      : {abs(spacing):.2f} px / order")

    print()
    print(f"  {'Order':>7}  {'Pos (px)':>9}  {'Amplitude':>11}  {'FWHM (px)':>9}")
    print('  ' + '-' * 45)
    for pk in peaks:
        label = f"SL{pk['order']:+d}"
        print(
            f"  {label:>7}  {pk['pos_along_axis']:+9.2f}  "
            f"{pk['amplitude']:11.1f}  {pk['fwhm']:9.2f}"
        )
    print(sep + '\n')


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Satellite peak diagnostic — validate on one spot image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'image_path', nargs='?', default=None,
        help='Path to spot image (.h5, .npy, .tif). Omit to use synthetic test data.',
    )
    parser.add_argument('--spot-key', default=None,
                        help='HDF5 group key (e.g. spot_0000_0).')
    parser.add_argument('--axis-angle', type=float, default=None,
                        help='Fix satellite axis in degrees (auto-detect if omitted).')
    parser.add_argument('--n-max', type=int, default=3,
                        help='Max satellite order to accept (default: 3).')
    parser.add_argument('--prominence', type=float, default=0.05,
                        help='Peak prominence threshold 0–1 (default: 0.05).')
    parser.add_argument('--strip-width', type=float, default=5.0,
                        help='Profile strip width in pixels (default: 5).')
    parser.add_argument('--bg-sigma', type=float, default=20.0,
                        help='Background subtraction sigma in pixels (default: 20).')
    parser.add_argument('--save', default=None,
                        help='Save figure to this path (e.g. diagnostic.png).')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.image_path is None:
        print('[INFO] No image path supplied — generating synthetic test image.')
        print('[INFO] True parameters: n_sat=3, spacing=22 px, axis_angle=35°')
        image = make_synthetic_image(
            n_satellites=3, spacing=22.0, axis_angle=35.0,
            envelope_decay=0.5, noise_level=50.0,
        )
        print(f'[INFO] Synthetic image shape={image.shape}, '
              f'range=[{image.min():.0f}, {image.max():.0f}]')
    else:
        image = load_image(args.image_path, spot_key=args.spot_key)
        print(f'[INFO] Loaded: {args.image_path}  shape={image.shape}  '
              f'range=[{image.min():.0f}, {image.max():.0f}]')

    print(
        f'[INFO] Detection parameters: axis_angle={args.axis_angle}, '
        f'n_max={args.n_max}, prominence={args.prominence}, '
        f'strip_width={args.strip_width}, bg_sigma={args.bg_sigma}'
    )

    result = detect_satellites(
        image,
        axis_angle=args.axis_angle,
        n_max=args.n_max,
        min_prominence=args.prominence,
        strip_width=args.strip_width,
        bg_sigma=args.bg_sigma,
    )

    print_summary(result)
    plot_results(image, result, save_path=args.save)


if __name__ == '__main__':
    main()
