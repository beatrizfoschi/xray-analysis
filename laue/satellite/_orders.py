"""Colour coding for satellite orders, shared by the plotting entry points.

SL0 is red; each order away from it steps through the tab10 cycle, so the same
order keeps the same colour across the single-image and batch figures.
"""

from __future__ import annotations

ORDER_COLORS = {
    -3: '#1f77b4',
    -2: '#ff7f0e',
    -1: '#2ca02c',
     0: '#d62728',
     1: '#9467bd',
     2: '#8c564b',
     3: '#e377c2',
}

_FALLBACK = '#17becf'

# The superlattice zero order. Deliberately outside the palette above: it is
# not one of the detected orders but a separate feature on the bulk's flank,
# and must never be mistaken for a satellite on a figure.
SL0_COLOR = '#00e5ff'


def order_color(order: int) -> str:
    """Colour for a satellite order; orders beyond ±3 share a fallback colour."""
    return ORDER_COLORS.get(order, _FALLBACK)
