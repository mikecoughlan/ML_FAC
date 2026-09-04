"""
plotting_utils.py
=================

Shared plotting helpers, chiefly the polar (MLAT x MLT) projection used
for every field-aligned current map in this project.

Polar convention
----------------
Every polar plot in this project follows one convention, applied by
polar_axis():

  * midnight (00 MLT) at the BOTTOM        set_theta_zero_location('S')
  * MLT increasing counter-clockwise       set_theta_direction(1)
    which puts dawn (06) on the left and dusk (18) on the right
  * radial axis is COLATITUDE (90 - MLAT), so the pole sits at the
    centre and the equatorward edge at the rim, labelled in MLAT degrees

Deviating from this in one figure and not another makes maps silently
non-comparable, so route new polar plots through polar_axis() rather
than configuring axes by hand.

Radial extent
-------------
The radial axis always spans the full model grid: for the standard 50
MLAT bins of 1 degree each, that is 0-50 degrees colatitude, i.e. 40-90
degrees MLAT. It is derived from the grid shape rather than passed in,
so no figure silently crops rows of real data.

Comparisons against the Weimer model are the one case needing a
different radial extent, since Weimer is evaluated on its own native
grid. Those plots are not produced by these scripts and should set their
limits explicitly at the call site.

Colormap convention
-------------------
Held in CMAPS:
  'hss', 'nrmse'  -> magma     (sequential, one-sided metrics)
  'correlation'   -> bwr       (diverging, symmetric about zero)
  'hss_sector'    -> coolwarm  (sector fills)
  'fac'           -> bwr       (diverging: upward/downward current)

Contents
--------
CMAPS               Project colormap conventions.
polar_axis          Create or configure an axis to the convention above.
fac_polar_meshgrid  (theta, r) meshgrid matching a FAC grid.
plot_fac_polar      Draw a FAC grid as a polar pcolormesh.
"""

import matplotlib.pyplot as plt
import numpy as np

# Default FAC grid shape: 50 MLAT bins x 24 MLT bins.
N_MLAT = 50
N_MLT = 24

# Width of one MLAT bin, in degrees. With N_MLAT = 50 this puts the
# equatorward edge of the grid at 40 degrees MLAT.
MLAT_BIN_WIDTH = 1.0

CMAPS = {
    'hss':         'magma',
    'nrmse':       'magma',
    'correlation': 'bwr',
    'hss_sector':  'coolwarm',
    'fac':         'bwr',
}


def grid_colat_extent(n_mlat=N_MLAT):
    """
    Outer edge of the radial axis, in degrees colatitude, for a grid.

    Args:
        n_mlat: Number of MLAT bins in the grid.

    Returns:
        float: colatitude of the grid's equatorward edge.
    """
    return n_mlat * MLAT_BIN_WIDTH


def polar_axis(ax=None, n_mlat=N_MLAT, mlt_tick_step=3, colat_tick_step=10,
               figsize=(6, 6), grid=True, tick_color='dimgrey'):
    """
    Create or configure a polar axis in the project convention.

    Applies the orientation described in the module docstring and labels
    the radial axis in MLAT degrees while plotting in colatitude, so data
    indexed from the pole outward needs no transformation.

    Args:
        ax: Existing polar axis to configure. If None, a new figure and
            axis are created. Must have been created with
            projection='polar'.
        n_mlat: Number of MLAT bins, which sets the radial extent via
            grid_colat_extent.
        mlt_tick_step: Spacing of MLT tick labels, in hours.
        colat_tick_step: Spacing of radial ticks, in degrees.
        figsize: Figure size, used only when creating a new figure.
        grid: Whether to draw the dashed polar grid.
        tick_color: Colour for the radial tick labels.

    Returns:
        (fig, ax): the figure and the configured polar axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    else:
        fig = ax.get_figure()

    ax.set_theta_zero_location('S')   # midnight (00 MLT) at bottom
    ax.set_theta_direction(1)         # counter-clockwise: dawn left, dusk right

    # Radial axis runs pole -> equatorward edge in colatitude, but is
    # LABELLED in MLAT, since MLAT is what the science is discussed in.
    colat_max = grid_colat_extent(n_mlat)
    ax.set_ylim(0, colat_max)
    colat_ticks = np.arange(0, colat_max + 1, colat_tick_step)
    ax.set_yticks(colat_ticks)
    ax.set_yticklabels(
        [f'{int(90 - c)}°' for c in colat_ticks], fontsize=8, color=tick_color
    )

    # Angular axis: MLT hours converted to radians.
    mlt_tick_hrs = np.arange(0, 24, mlt_tick_step)
    ax.set_xticks(mlt_tick_hrs * 2.0 * np.pi / 24.0)
    ax.set_xticklabels([f'{h:02d}' for h in mlt_tick_hrs], fontsize=9)

    if grid:
        ax.grid(color='lightgrey', linestyle='--', linewidth=0.6, zorder=0)

    return fig, ax


def fac_polar_meshgrid(n_mlat=N_MLAT, n_mlt=N_MLT):
    """
    Build the (theta, r) meshgrid matching a FAC grid.

    Both axes use endpoint=False so the MLT cells tile the full circle
    without duplicating midnight, and the MLAT cells tile the radial
    range without duplicating an edge.

    Args:
        n_mlat: Number of MLAT bins (radial cells).
        n_mlt: Number of MLT bins (angular cells).

    Returns:
        (theta, r): meshgrid arrays of shape (n_mlt, n_mlat), matching
            the TRANSPOSE of an (n_mlat, n_mlt) data array. See
            plot_fac_polar for how that is handled.
    """
    r, theta = np.meshgrid(
        np.linspace(0, grid_colat_extent(n_mlat), n_mlat, endpoint=False),
        np.linspace(0, 2 * np.pi, n_mlt, endpoint=False),
    )
    return theta, r


def plot_fac_polar(data, ax=None, cmap=None, vmin=None, vmax=None,
                   symmetric=True, **kwargs):
    """
    Draw a FAC grid as a polar pcolormesh in the project convention.

    The radial extent follows the grid, so the whole array is shown.

    Args:
        data: (n_mlat, n_mlt) array, MLAT along axis 0 and MLT along
            axis 1. This is the standard orientation in this project --
            AMPERE arrays loaded flat from pickles must already have been
            restored with `.reshape(24, 50).T` before reaching here.
        ax: Existing polar axis, or None to create one.
        cmap: Colormap; defaults to CMAPS['fac'].
        vmin, vmax: Colour limits. If both are None and symmetric is
            True, limits are set symmetrically about zero.
        symmetric: Whether to auto-scale symmetrically about zero, so
            upward and downward currents are coloured comparably.
        **kwargs: Passed through to pcolormesh.

    Returns:
        (fig, ax, mesh): the figure, the axis, and the QuadMesh, the
            last of which can be handed to fig.colorbar().
    """
    data = np.asarray(data)
    n_mlat, n_mlt = data.shape

    fig, ax = polar_axis(ax=ax, n_mlat=n_mlat)
    theta, r = fac_polar_meshgrid(n_mlat=n_mlat, n_mlt=n_mlt)

    if cmap is None:
        cmap = CMAPS['fac']

    # Symmetric limits keep zero at the centre of a diverging colormap;
    # without this, a mostly-positive map reads as though it has no
    # downward current at all.
    if symmetric and vmin is None and vmax is None:
        finite = data[np.isfinite(data)]
        if finite.size:
            lim = np.max(np.abs(finite))
            vmin, vmax = -lim, lim

    # data is (mlat, mlt) but the meshgrid is (mlt, mlat), so transpose.
    mesh = ax.pcolormesh(theta, r, data.T, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    return fig, ax, mesh
