"""Interactive 3-axis PET slice viewer with optional tilted-plane overlay."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, RadioButtons


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _clip_line_to_rect(
    n1: float,
    n2: float,
    D: float,
    c1_min: float,
    c1_max: float,
    c2_min: float,
    c2_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the segment of line  n1·c1 + n2·c2 = D  inside a rectangle.

    Returns (c1_pts, c2_pts) – two end-points of the clipped segment,
    or empty arrays when the line misses the rectangle entirely.
    """
    tol = 1e-12
    pts: list[tuple[float, float]] = []

    if abs(n2) > tol:
        for c1_edge in (c1_min, c1_max):
            c2 = (D - n1 * c1_edge) / n2
            if c2_min - tol <= c2 <= c2_max + tol:
                pts.append((c1_edge, float(np.clip(c2, c2_min, c2_max))))

    if abs(n1) > tol:
        for c2_edge in (c2_min, c2_max):
            c1 = (D - n2 * c2_edge) / n1
            if c1_min - tol <= c1 <= c1_max + tol:
                pts.append((float(np.clip(c1, c1_min, c1_max)), c2_edge))

    if len(pts) < 2:
        return np.array([]), np.array([])

    # Deduplicate (corner intersections show up twice)
    unique: list[tuple[float, float]] = [pts[0]]
    for pt in pts[1:]:
        if abs(pt[0] - unique[-1][0]) > 1e-10 or abs(pt[1] - unique[-1][1]) > 1e-10:
            unique.append(pt)

    if len(unique) < 2:
        return np.array([]), np.array([])

    return np.array([p[0] for p in unique[:2]]), np.array([p[1] for p in unique[:2]])


def _plane_line_on_cut(
    p: np.ndarray,
    n: np.ndarray,
    fixed_axis: int,
    fixed_world: float,
    coord1: np.ndarray,
    coord2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersection of plane (point p, unit normal n) with a 2-D slice.

    Parameters
    ----------
    fixed_axis : 0 = x, 1 = y, 2 = z  (axis perpendicular to the slice)
    fixed_world : world coordinate of the slice along *fixed_axis*
    coord1, coord2 : 1-D world coordinate arrays for the two display axes

    Returns
    -------
    (c1_pts, c2_pts) : two end-points in world coordinates, or empty arrays
        when the plane is parallel to the slice.
    """
    axes = [0, 1, 2]
    axes.remove(fixed_axis)
    a1, a2 = axes

    n_a1, n_a2 = float(n[a1]), float(n[a2])
    n_f = float(n[fixed_axis])

    # (r − p) · n = 0  →  n_a1·c1 + n_a2·c2 = D
    D = -n_f * (fixed_world - float(p[fixed_axis])) + n_a1 * float(p[a1]) + n_a2 * float(p[a2])

    if abs(n_a1) < 1e-12 and abs(n_a2) < 1e-12:
        # Plane normal has no component in the slice plane → no intersection line
        return np.array([]), np.array([])

    return _clip_line_to_rect(
        n_a1, n_a2, D,
        float(coord1[0]), float(coord1[-1]),
        float(coord2[0]), float(coord2[-1]),
    )


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------

def show_pet_cuts(
    pet_crop: np.ndarray,
    x_crop: np.ndarray,
    y_crop: np.ndarray,
    z_crop: np.ndarray,
    tilted_plane_vectors: dict | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "Greys",
    figsize: tuple[int, int] = (14, 8),
    fig_title: str | None = None,
) -> tuple:
    """Interactive 3-axis PET slice viewer with optional tilted-plane overlay.

    Displays three orthogonal cuts through *pet_crop* with sliders to scroll
    each axis.  When *tilted_plane_vectors* is supplied a radio-button row
    lets the user pick a plane (by edge label, e.g. "xe", "ys") or "None";
    the selected plane is drawn as a dashed red line on all three panels.

    Parameters
    ----------
    pet_crop : ndarray, shape (nx, ny, nz)
        Cropped PET volume.
    x_crop, y_crop, z_crop : 1-D arrays
        World coordinates (mm) for each voxel axis.  The coordinate origin
        (0, 0, 0) mm is at the centre of the un-cropped image.
    tilted_plane_vectors : dict or None
        ``{edge_label: (point, normal)}`` in world coordinates.
        *point* and *normal* may be lists or numpy arrays.
    vmin, vmax : float or None
        Initial colour limits (default: global min / max of *pet_crop*).
    cmap : str
        Initial matplotlib colormap name.
    figsize : tuple of two ints
        Figure size in inches.
    fig_title : str or None
        Overall figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_im : list of three Axes (image panels)
    widgets : dict
        All interactive widgets keyed by name.  Keep a reference to prevent
        garbage collection of the callbacks.
    """
    nx, ny, nz = pet_crop.shape
    dx = float(x_crop[1] - x_crop[0]) if len(x_crop) > 1 else 1.0
    dy = float(y_crop[1] - y_crop[0]) if len(y_crop) > 1 else 1.0
    dz = float(z_crop[1] - z_crop[0]) if len(z_crop) > 1 else 1.0

    if vmin is None:
        vmin = float(pet_crop.min())
    if vmax is None:
        vmax = float(pet_crop.max())

    ix0, iy0, iz0 = nx // 2, ny // 2, nz // 2

    # World-coordinate extents for imshow (outer edges of border voxels)
    ext_x = [x_crop[0] - dx / 2, x_crop[-1] + dx / 2]
    ext_y = [y_crop[0] - dy / 2, y_crop[-1] + dy / 2]
    ext_z = [z_crop[0] - dz / 2, z_crop[-1] + dz / 2]

    # ---- plane selector options ----
    plane_labels = ["None"]
    if tilted_plane_vectors:
        plane_labels += list(tilted_plane_vectors.keys())
    selected_plane: list[str | None] = [None]  # mutable container for closures

    # ---- figure layout ----
    # rows: images | plane radio | slice sliders | vmin/vmax/cmap | colorbar
    n_radio_rows = 1 if len(plane_labels) <= 8 else 2  # single row fits ≤ 8 options
    fig = plt.figure(figsize=figsize)
    if fig_title is not None:
        fig.suptitle(fig_title)

    gs = fig.add_gridspec(
        5, 3,
        height_ratios=[8, max(0.8, 0.25 * len(plane_labels)), 0.9, 0.9, 0.4],
        hspace=0.75,
        wspace=0.45,
    )

    ax_im = [fig.add_subplot(gs[0, c]) for c in range(3)]
    ax_rb = fig.add_subplot(gs[1, :])   # radio buttons span all columns
    ax_sl = [fig.add_subplot(gs[2, c]) for c in range(3)]
    ax_tb = [fig.add_subplot(gs[3, c]) for c in range(3)]
    ax_cb = fig.add_subplot(gs[4, :])

    # ---- imshow panels (using world-coordinate extents) ----
    # Panel 0: cut at x=ix  → horiz = y (mm), vert = z (mm)
    # Panel 1: cut at y=iy  → horiz = x (mm), vert = z (mm)
    # Panel 2: cut at z=iz  → horiz = x (mm), vert = y (mm)
    extents = [
        [*ext_y, *ext_z],
        [*ext_x, *ext_z],
        [*ext_x, *ext_y],
    ]
    xlabels = ["y (mm)", "x (mm)", "x (mm)"]
    ylabels = ["z (mm)", "z (mm)", "y (mm)"]

    imkw = dict(vmin=vmin, vmax=vmax, cmap=cmap, origin="lower", aspect="equal")

    def _get_cuts(ix: int, iy: int, iz: int):
        return (
            pet_crop[ix, :, :].T,
            pet_crop[:, iy, :].T,
            pet_crop[:, :, iz].T,
        )

    c0, c1, c2 = _get_cuts(ix0, iy0, iz0)
    ims = [
        ax_im[0].imshow(c0, extent=extents[0], **imkw),
        ax_im[1].imshow(c1, extent=extents[1], **imkw),
        ax_im[2].imshow(c2, extent=extents[2], **imkw),
    ]
    for i, ax in enumerate(ax_im):
        ax.set_xlabel(xlabels[i], fontsize="small")
        ax.set_ylabel(ylabels[i], fontsize="small")
    ax_im[0].set_title(f"x = {x_crop[ix0]:.1f} mm", fontsize="small")
    ax_im[1].set_title(f"y = {y_crop[iy0]:.1f} mm", fontsize="small")
    ax_im[2].set_title(f"z = {z_crop[iz0]:.1f} mm", fontsize="small")

    # ---- plane overlay lines (one per panel, hidden until a plane is chosen) ----
    line_kw = dict(color="tab:red", lw=1.5, ls="--")
    plane_lines = [ax_im[k].plot([], [], **line_kw)[0] for k in range(3)]

    def _update_plane_lines(ix: int, iy: int, iz: int) -> None:
        edge = selected_plane[0]
        if edge is None or tilted_plane_vectors is None or edge not in tilted_plane_vectors:
            for line in plane_lines:
                line.set_data([], [])
            return

        p = np.asarray(tilted_plane_vectors[edge][0], dtype=float)
        n = np.asarray(tilted_plane_vectors[edge][1], dtype=float)

        # Panel 0: fixed x, displays (y, z)
        c1p, c2p = _plane_line_on_cut(p, n, 0, x_crop[ix], y_crop, z_crop)
        plane_lines[0].set_data(c1p, c2p)

        # Panel 1: fixed y, displays (x, z)
        c1p, c2p = _plane_line_on_cut(p, n, 1, y_crop[iy], x_crop, z_crop)
        plane_lines[1].set_data(c1p, c2p)

        # Panel 2: fixed z, displays (x, y)
        c1p, c2p = _plane_line_on_cut(p, n, 2, z_crop[iz], x_crop, y_crop)
        plane_lines[2].set_data(c1p, c2p)

    # ---- radio buttons (plane selector) ----
    ax_rb.set_axis_off()
    try:
        rb = RadioButtons(ax_rb, plane_labels, active=0, orientation="horizontal")
    except TypeError:
        # older matplotlib without orientation parameter
        rb = RadioButtons(ax_rb, plane_labels, active=0)

    ax_rb.set_title("Plane overlay:", fontsize="small", loc="left", pad=2)

    def _on_plane_select(label: str) -> None:
        selected_plane[0] = None if label == "None" else label
        _update_plane_lines(int(sx.val), int(sy.val), int(sz.val))
        fig.canvas.draw_idle()

    rb.on_clicked(_on_plane_select)

    # ---- slice sliders ----
    sx = Slider(ax_sl[0], "x", 0, nx - 1, valinit=ix0, valstep=1)
    sy = Slider(ax_sl[1], "y", 0, ny - 1, valinit=iy0, valstep=1)
    sz = Slider(ax_sl[2], "z", 0, nz - 1, valinit=iz0, valstep=1)

    def _update_slices(_) -> None:
        ix, iy, iz = int(sx.val), int(sy.val), int(sz.val)
        cuts = _get_cuts(ix, iy, iz)
        for j, im in enumerate(ims):
            im.set_data(cuts[j])
        ax_im[0].set_title(f"x = {x_crop[ix]:.1f} mm", fontsize="small")
        ax_im[1].set_title(f"y = {y_crop[iy]:.1f} mm", fontsize="small")
        ax_im[2].set_title(f"z = {z_crop[iz]:.1f} mm", fontsize="small")
        _update_plane_lines(ix, iy, iz)
        fig.canvas.draw_idle()

    sx.on_changed(_update_slices)
    sy.on_changed(_update_slices)
    sz.on_changed(_update_slices)

    # ---- colorbar ----
    cb = fig.colorbar(ims[0], cax=ax_cb, orientation="horizontal")

    # ---- vmin / vmax / cmap text boxes ----
    tb_vmin = TextBox(ax_tb[0], "vmin", initial=f"{vmin:.4g}")
    tb_vmax = TextBox(ax_tb[1], "vmax", initial=f"{vmax:.4g}")
    tb_cmap = TextBox(ax_tb[2], "cmap", initial=cmap)

    def _update_clim(_) -> None:
        try:
            lo, hi = float(tb_vmin.text), float(tb_vmax.text)
        except ValueError:
            return
        for im in ims:
            im.set_clim(lo, hi)
        cb.update_normal(ims[0])
        fig.canvas.draw_idle()

    def _update_cmap(name: str) -> None:
        try:
            cm = plt.get_cmap(name.strip())
        except (ValueError, KeyError):
            tb_cmap.set_val(ims[0].cmap.name)
            return
        for im in ims:
            im.set_cmap(cm)
        cb.update_normal(ims[0])
        fig.canvas.draw_idle()

    tb_vmin.on_submit(_update_clim)
    tb_vmax.on_submit(_update_clim)
    tb_cmap.on_submit(_update_cmap)

    widgets = {
        "s_x": sx,
        "s_y": sy,
        "s_z": sz,
        "rb_plane": rb,
        "tb_vmin": tb_vmin,
        "tb_vmax": tb_vmax,
        "tb_cmap": tb_cmap,
    }

    return fig, ax_im, widgets
