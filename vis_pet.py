"""Interactive 3-axis PET slice viewer with optional tilted-plane and ROI-mask overlay."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.widgets import Slider, TextBox, Button


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
    """Find the segment of line  n1·c1 + n2·c2 = D  inside a rectangle."""
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
    """Intersection of plane (point p, unit normal n) with a 2-D slice."""
    axes = [0, 1, 2]
    axes.remove(fixed_axis)
    a1, a2 = axes

    n_a1, n_a2 = float(n[a1]), float(n[a2])
    n_f = float(n[fixed_axis])

    D = -n_f * (fixed_world - float(p[fixed_axis])) + n_a1 * float(p[a1]) + n_a2 * float(p[a2])

    if abs(n_a1) < 1e-12 and abs(n_a2) < 1e-12:
        return np.array([]), np.array([])

    return _clip_line_to_rect(
        n_a1, n_a2, D,
        float(coord1[0]), float(coord1[-1]),
        float(coord2[0]), float(coord2[-1]),
    )


# ---------------------------------------------------------------------------
# Colour constants and mask colormap
# ---------------------------------------------------------------------------

_COLOR_ACTIVE   = "#4C9BE8"
_COLOR_INACTIVE = "white"
_COLOR_HOVER    = "#D0E8FF"

# 0 → fully transparent, 1 → semi-transparent red
_MASK_CMAP = ListedColormap([(0.0, 0.0, 0.0, 0.0), (1.0, 0.2, 0.2, 0.4)])


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------

def show_pet_cuts(
    pet_crop: np.ndarray,
    x_crop: np.ndarray,
    y_crop: np.ndarray,
    z_crop: np.ndarray,
    tilted_plane_vectors: dict | None = None,
    roi_masks: dict | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "Greys",
    interpolation: str = "nearest",
    arrow_scale_mm: float = 8.0,
    figsize: tuple[int, int] = (14, 9),
    fig_title: str | None = None,
) -> tuple:
    """Interactive 3-axis PET slice viewer with optional tilted-plane and ROI-mask overlay.

    Parameters
    ----------
    pet_crop : ndarray, shape (nx, ny, nz)
        Cropped PET volume.
    x_crop, y_crop, z_crop : 1-D arrays
        World coordinates (mm) for each voxel axis.
    tilted_plane_vectors : dict or None
        ``{edge_label: (point, normal)}`` – plane overlays.
    roi_masks : dict or None
        ``{edge_label: binary ndarray of shape (nx, ny, nz)}`` – ROI masks.
    vmin, vmax : float or None
        Initial colour limits.
    cmap : str
        Initial matplotlib colormap name.
    interpolation : str
        Initial imshow interpolation method (e.g. "nearest", "bilinear").
    arrow_scale_mm : float
        Length of the normal-vector arrow in mm.
    figsize : tuple
        Figure size in inches.
    fig_title : str or None
        Overall figure title.

    Returns
    -------
    fig, ax_im, widgets
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

    ext_x = [x_crop[0] - dx / 2, x_crop[-1] + dx / 2]
    ext_y = [y_crop[0] - dy / 2, y_crop[-1] + dy / 2]
    ext_z = [z_crop[0] - dz / 2, z_crop[-1] + dz / 2]

    plane_labels = ["None"] + (list(tilted_plane_vectors.keys()) if tilted_plane_vectors else [])
    mask_labels  = ["None"] + (list(roi_masks.keys())            if roi_masks            else [])
    selected_plane: list[str | None] = [None]
    selected_mask:  list[str | None] = [None]

    # ---- figure layout -------------------------------------------------------
    # rows: images | sliders | textboxes | colorbar | plane buttons | mask buttons
    fig = plt.figure(figsize=figsize)
    if fig_title is not None:
        fig.suptitle(fig_title)

    gs = fig.add_gridspec(
        6, 3,
        height_ratios=[8, 0.9, 0.9, 0.4, 0.7, 0.7],
        hspace=0.75,
        wspace=0.45,
    )

    ax_im        = [fig.add_subplot(gs[0, c]) for c in range(3)]
    ax_sl        = [fig.add_subplot(gs[1, c]) for c in range(3)]
    ax_cb        = fig.add_subplot(gs[3, :])
    ax_plane_row = fig.add_subplot(gs[4, :])
    ax_mask_row  = fig.add_subplot(gs[5, :])

    # textbox row: 4 items (vmin, vmax, cmap, interp) via sub-gridspec
    gs_tb = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2, :], wspace=0.6)
    ax_tb_vmin  = fig.add_subplot(gs_tb[0, 0])
    ax_tb_vmax  = fig.add_subplot(gs_tb[0, 1])
    ax_tb_cmap  = fig.add_subplot(gs_tb[0, 2])
    ax_tb_interp = fig.add_subplot(gs_tb[0, 3])

    # ---- imshow panels -------------------------------------------------------
    extents = [
        [*ext_y, *ext_z],   # panel 0: horiz=y, vert=z
        [*ext_x, *ext_z],   # panel 1: horiz=x, vert=z
        [*ext_x, *ext_y],   # panel 2: horiz=x, vert=y
    ]
    xlabels = ["y (mm)", "x (mm)", "x (mm)"]
    ylabels = ["z (mm)", "z (mm)", "y (mm)"]

    imkw = dict(vmin=vmin, vmax=vmax, cmap=cmap, origin="lower", aspect="equal",
                interpolation=interpolation)

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

    # ---- ROI mask overlay imshows -------------------------------------------
    mask_ims = [
        ax_im[0].imshow(
            np.zeros((nz, ny)), extent=extents[0],
            cmap=_MASK_CMAP, vmin=0, vmax=1, origin="lower", aspect="equal", zorder=2,
        ),
        ax_im[1].imshow(
            np.zeros((nz, nx)), extent=extents[1],
            cmap=_MASK_CMAP, vmin=0, vmax=1, origin="lower", aspect="equal", zorder=2,
        ),
        ax_im[2].imshow(
            np.zeros((ny, nx)), extent=extents[2],
            cmap=_MASK_CMAP, vmin=0, vmax=1, origin="lower", aspect="equal", zorder=2,
        ),
    ]

    # ---- plane overlay: line, origin dot, and normal-vector arrow -----------
    _arrow_kw = dict(
        arrowstyle="-|>",
        color="tab:red",
        lw=1.5,
        mutation_scale=10,
    )
    line_kw   = dict(color="tab:red", lw=1.5, ls="--", zorder=3)
    marker_kw = dict(color="tab:red", marker="o", ms=6, ls="none", zorder=4)

    plane_lines   = [ax_im[k].plot([], [], **line_kw)[0]   for k in range(3)]
    plane_markers = [ax_im[k].plot([], [], **marker_kw)[0] for k in range(3)]
    # arrows are FancyArrowPatch objects; stored so they can be removed on update
    plane_arrows: list = [None, None, None]

    def _remove_arrows() -> None:
        for k in range(3):
            if plane_arrows[k] is not None:
                plane_arrows[k].remove()
                plane_arrows[k] = None

    def _update_plane_lines(ix: int, iy: int, iz: int) -> None:
        _remove_arrows()
        edge = selected_plane[0]
        if edge is None or tilted_plane_vectors is None or edge not in tilted_plane_vectors:
            for line in plane_lines:
                line.set_data([], [])
            for marker in plane_markers:
                marker.set_data([], [])
            return

        p = np.asarray(tilted_plane_vectors[edge][0], dtype=float)
        n = np.asarray(tilted_plane_vectors[edge][1], dtype=float)

        # ---- Panel 0: fixed x, displays (y, z) ----
        c1p, c2p = _plane_line_on_cut(p, n, 0, x_crop[ix], y_crop, z_crop)
        plane_lines[0].set_data(c1p, c2p)
        at_p0 = abs(x_crop[ix] - p[0]) <= dx / 2
        plane_markers[0].set_data([p[1]] if at_p0 else [], [p[2]] if at_p0 else [])
        if at_p0:
            plane_arrows[0] = ax_im[0].annotate(
                "",
                xy=(p[1] + arrow_scale_mm * n[1], p[2] + arrow_scale_mm * n[2]),
                xytext=(p[1], p[2]),
                arrowprops=_arrow_kw,
                zorder=5,
            )

        # ---- Panel 1: fixed y, displays (x, z) ----
        c1p, c2p = _plane_line_on_cut(p, n, 1, y_crop[iy], x_crop, z_crop)
        plane_lines[1].set_data(c1p, c2p)
        at_p1 = abs(y_crop[iy] - p[1]) <= dy / 2
        plane_markers[1].set_data([p[0]] if at_p1 else [], [p[2]] if at_p1 else [])
        if at_p1:
            plane_arrows[1] = ax_im[1].annotate(
                "",
                xy=(p[0] + arrow_scale_mm * n[0], p[2] + arrow_scale_mm * n[2]),
                xytext=(p[0], p[2]),
                arrowprops=_arrow_kw,
                zorder=5,
            )

        # ---- Panel 2: fixed z, displays (x, y) ----
        c1p, c2p = _plane_line_on_cut(p, n, 2, z_crop[iz], x_crop, y_crop)
        plane_lines[2].set_data(c1p, c2p)
        at_p2 = abs(z_crop[iz] - p[2]) <= dz / 2
        plane_markers[2].set_data([p[0]] if at_p2 else [], [p[1]] if at_p2 else [])
        if at_p2:
            plane_arrows[2] = ax_im[2].annotate(
                "",
                xy=(p[0] + arrow_scale_mm * n[0], p[1] + arrow_scale_mm * n[1]),
                xytext=(p[0], p[1]),
                arrowprops=_arrow_kw,
                zorder=5,
            )

    def _update_mask_overlay(ix: int, iy: int, iz: int) -> None:
        edge = selected_mask[0]
        if edge is None or roi_masks is None or edge not in roi_masks:
            mask_ims[0].set_data(np.zeros((nz, ny)))
            mask_ims[1].set_data(np.zeros((nz, nx)))
            mask_ims[2].set_data(np.zeros((ny, nx)))
            return
        mask = roi_masks[edge]
        mask_ims[0].set_data(mask[ix, :, :].T.astype(float))
        mask_ims[1].set_data(mask[:, iy, :].T.astype(float))
        mask_ims[2].set_data(mask[:, :, iz].T.astype(float))

    # ---- helper: build a horizontal button strip ----------------------------
    def _build_btn_row(
        ax_placeholder,
        row_label: str,
        labels: list[str],
        selected: list[str | None],
        update_fn,
    ) -> list[Button]:
        ax_placeholder.set_axis_off()
        ax_placeholder.text(
            0.0, 0.5, row_label,
            transform=ax_placeholder.transAxes,
            va="center", ha="left", fontsize="small",
        )
        n_opts     = len(labels)
        label_frac = 0.12
        btn_gap    = 0.005
        btn_width  = (1.0 - label_frac - btn_gap * (n_opts - 1)) / n_opts
        bbox       = ax_placeholder.get_position()
        btn_axes   = []
        for k in range(n_opts):
            left = bbox.x0 + (label_frac + k * (btn_width + btn_gap)) * bbox.width
            btn_axes.append(fig.add_axes([left, bbox.y0, btn_width * bbox.width, bbox.height]))

        btns: list[Button] = []
        for k, label in enumerate(labels):
            color = _COLOR_ACTIVE if k == 0 else _COLOR_INACTIVE
            btn = Button(btn_axes[k], label, color=color, hovercolor=_COLOR_HOVER)
            btn.label.set_fontsize("small")
            btns.append(btn)

        def _make_cb(label: str, idx: int):
            def _cb(_event) -> None:
                for k, b in enumerate(btns):
                    b.color = _COLOR_ACTIVE if k == idx else _COLOR_INACTIVE
                    b.ax.set_facecolor(_COLOR_ACTIVE if k == idx else _COLOR_INACTIVE)
                selected[0] = None if label == "None" else label
                update_fn(int(sx.val), int(sy.val), int(sz.val))
                fig.canvas.draw_idle()
            return _cb

        for k, (btn, label) in enumerate(zip(btns, labels)):
            btn.on_clicked(_make_cb(label, k))

        return btns

    # Force layout so bounding boxes are available, then build button rows
    fig.canvas.draw()
    plane_btns = _build_btn_row(ax_plane_row, "Plane overlay:", plane_labels, selected_plane, _update_plane_lines)
    mask_btns  = _build_btn_row(ax_mask_row,  "ROI mask:",      mask_labels,  selected_mask,  _update_mask_overlay)

    # ---- slice sliders -------------------------------------------------------
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
        _update_mask_overlay(ix, iy, iz)
        fig.canvas.draw_idle()

    sx.on_changed(_update_slices)
    sy.on_changed(_update_slices)
    sz.on_changed(_update_slices)

    # ---- colorbar ------------------------------------------------------------
    cb = fig.colorbar(ims[0], cax=ax_cb, orientation="horizontal")

    # ---- textboxes: vmin, vmax, cmap, interpolation -------------------------
    tb_vmin  = TextBox(ax_tb_vmin,  "vmin",   initial=f"{vmin:.4g}")
    tb_vmax  = TextBox(ax_tb_vmax,  "vmax",   initial=f"{vmax:.4g}")
    tb_cmap  = TextBox(ax_tb_cmap,  "cmap",   initial=cmap)
    tb_interp = TextBox(ax_tb_interp, "interp", initial=interpolation)

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

    def _update_interp(name: str) -> None:
        try:
            for im in ims:
                im.set_interpolation(name.strip())
            fig.canvas.draw_idle()
        except (ValueError, KeyError):
            tb_interp.set_val(ims[0].get_interpolation())

    tb_vmin.on_submit(_update_clim)
    tb_vmax.on_submit(_update_clim)
    tb_cmap.on_submit(_update_cmap)
    tb_interp.on_submit(_update_interp)

    # ---- click-to-navigate ---------------------------------------------------
    def _on_click(event) -> None:
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        # Identify which panel was clicked and map world coords → voxel indices
        if event.inaxes is ax_im[0]:
            # panel 0: horiz=y, vert=z  →  update sy, sz
            new_iy = int(np.argmin(np.abs(y_crop - event.xdata)))
            new_iz = int(np.argmin(np.abs(z_crop - event.ydata)))
            sy.eventson = False
            sz.eventson = False
            sy.set_val(new_iy)
            sz.set_val(new_iz)
            sy.eventson = True
            sz.eventson = True
        elif event.inaxes is ax_im[1]:
            # panel 1: horiz=x, vert=z  →  update sx, sz
            new_ix = int(np.argmin(np.abs(x_crop - event.xdata)))
            new_iz = int(np.argmin(np.abs(z_crop - event.ydata)))
            sx.eventson = False
            sz.eventson = False
            sx.set_val(new_ix)
            sz.set_val(new_iz)
            sx.eventson = True
            sz.eventson = True
        elif event.inaxes is ax_im[2]:
            # panel 2: horiz=x, vert=y  →  update sx, sy
            new_ix = int(np.argmin(np.abs(x_crop - event.xdata)))
            new_iy = int(np.argmin(np.abs(y_crop - event.ydata)))
            sx.eventson = False
            sy.eventson = False
            sx.set_val(new_ix)
            sy.set_val(new_iy)
            sx.eventson = True
            sy.eventson = True
        else:
            return
        _update_slices(None)

    fig.canvas.mpl_connect("button_press_event", _on_click)

    widgets = {
        "s_x": sx,
        "s_y": sy,
        "s_z": sz,
        "plane_btns": plane_btns,
        "mask_btns":  mask_btns,
        "tb_vmin":    tb_vmin,
        "tb_vmax":    tb_vmax,
        "tb_cmap":    tb_cmap,
        "tb_interp":  tb_interp,
    }

    return fig, ax_im, widgets
