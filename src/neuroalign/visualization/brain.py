"""Brain visualization utilities for NeuroAlign.

Two main functions:
- render_brain_png: static composited brain image (yabplot, matching notebook style)
- render_interactive_html: nilearn-based interactive surface HTML for Streamlit embedding
"""

from __future__ import annotations

import os

# Must be set before pyvista/yabplot are imported
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import io
import logging
from typing import Optional

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

_SCHAEFER_ATLAS_SURFACE_CACHE: dict = {}


def _autocrop(img: np.ndarray, bg: int = 255, margin: int = 20) -> np.ndarray:
    """Crop white margins from a uint8 RGB image array."""
    mask = (img < bg - 10).any(axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return img
    r0 = max(0, rows[0] - margin)
    r1 = min(img.shape[0], rows[-1] + 1 + margin)
    c0 = max(0, cols[0] - margin)
    c1 = min(img.shape[1], cols[-1] + 1 + margin)
    return img[r0:r1, c0:c1]


def _pad_to_width(img: np.ndarray, target_w: int, fill: int = 255) -> np.ndarray:
    """Pad image symmetrically on left/right to reach target_w."""
    if img.shape[1] >= target_w:
        return img
    pad = target_w - img.shape[1]
    pl = pad // 2
    pr = pad - pl
    left = np.full((img.shape[0], pl, img.shape[2]), fill, dtype=img.dtype)
    right = np.full((img.shape[0], pr, img.shape[2]), fill, dtype=img.dtype)
    return np.concatenate([left, img, right], axis=1)


def render_brain_png(
    z_vector: np.ndarray,
    n_cortical: int = 400,
    tian_atlas_path: Optional[str] = None,
    cmap: str = "RdBu_r",
    vmin: float = -20,
    vmax: float = 20,
) -> bytes:
    """Render a composited cortical + subcortical brain image using yabplot.

    Parameters
    ----------
    z_vector : np.ndarray
        Per-parcel BAG z-scores. First ``n_cortical`` entries are cortical
        (Schaefer 400); the rest are subcortical (Tian S3).
    n_cortical : int
        Number of cortical parcels (default 400).
    tian_atlas_path : str or None
        Path to the Tian S3 surface atlas directory. If None, only the
        cortical map is rendered.
    cmap : str
        Matplotlib colormap name. Default ``"RdBu_r"`` (red=older, blue=younger).
    vmin, vmax : float
        Colormap range.

    Returns
    -------
    bytes
        PNG image bytes, suitable for ``st.image()``.
    """
    import yabplot as yab  # deferred import — pyvista needs OFF_SCREEN already set

    cort_data = z_vector[:n_cortical]
    sub_data = z_vector[n_cortical:] if len(z_vector) > n_cortical else np.array([])

    views_cort = ["left_medial", "left_lateral", "right_lateral", "right_medial"]
    views_sub = ["left_lateral", "inferior", "right_lateral"]

    # ── Cortical screenshot ───────────────────────────────────────────────────
    logger.info("Rendering cortical brain map (%d parcels)…", len(cort_data))
    cort_plotter = yab.plot_cortical(
        cort_data,
        atlas="schaefer_400",
        views=views_cort,
        bmesh_type="midthickness",
        cmap=cmap,
        vminmax=(vmin, vmax),
        display_type="object",
    )
    cort_img = cort_plotter.screenshot(return_img=True, transparent_background=False)
    cort_plotter.close()
    # Crop top/bottom padding (matches notebook)
    cort_img = cort_img[100:-100, :]

    # ── Subcortical screenshot ────────────────────────────────────────────────
    has_sub = tian_atlas_path is not None and len(sub_data) > 0
    if has_sub:
        logger.info("Rendering subcortical brain map (%d regions)…", len(sub_data))
        try:
            sub_plotter = yab.plot_subcortical(
                sub_data,
                custom_atlas_path=tian_atlas_path,
                views=views_sub,
                bmesh_type="midthickness",
                bmesh_alpha=0.1,
                bmesh_color="gray",
                cmap=cmap,
                vminmax=(vmin, vmax),
                figsize=(1000, 400),
                display_type="object",
            )
            sub_img = sub_plotter.screenshot(return_img=True, transparent_background=False)
            sub_plotter.close()
            # Auto-crop white margins so we only keep the actual brain content
            sub_img = _autocrop(sub_img)
        except Exception as exc:
            logger.warning("Subcortical rendering failed (%s); showing cortical only.", exc)
            has_sub = False

    # ── Composite: stack images into a single numpy array ────────────────────
    #
    # Working in pixel space avoids all matplotlib layout/centering surprises.
    # Both rows are padded to the same width (symmetric left/right), then
    # stacked with a thin white gap.
    #
    if has_sub:
        target_w = max(cort_img.shape[1], sub_img.shape[1])
        cort_img = _pad_to_width(cort_img, target_w)
        sub_img = _pad_to_width(sub_img, target_w)
        gap = np.full((15, target_w, 3), 255, dtype=np.uint8)
        composite = np.vstack([cort_img, gap, sub_img])
    else:
        composite = cort_img
        target_w = composite.shape[1]

    # ── Single-axes figure ────────────────────────────────────────────────────
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(target_w / 100, composite.shape[0] / 100),
    )
    ax.imshow(composite, aspect="equal")
    ax.axis("off")

    # Colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        fraction=0.015,
        pad=0.01,
        shrink=0.6,
    )
    cbar.set_label("BAG z-score\n← younger   older →", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _build_surface_lookup(n_rois: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Fetch Schaefer atlas and project onto fsaverage5 vertices.

    Returns (lh_labels, rh_labels) — integer parcel IDs per vertex.
    Cached in-process after first call.
    """
    key = f"schaefer_{n_rois}"
    if key in _SCHAEFER_ATLAS_SURFACE_CACHE:
        return _SCHAEFER_ATLAS_SURFACE_CACHE[key]

    from nilearn import datasets, surface  # deferred

    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=2)
    fsavg = datasets.fetch_surf_fsaverage("fsaverage5")

    lh_labels = surface.vol_to_surf(
        schaefer.maps, fsavg.pial_left, interpolation="nearest_most_frequent"
    ).astype(int)
    rh_labels = surface.vol_to_surf(
        schaefer.maps, fsavg.pial_right, interpolation="nearest_most_frequent"
    ).astype(int)

    _SCHAEFER_ATLAS_SURFACE_CACHE[key] = (lh_labels, rh_labels)
    return lh_labels, rh_labels


def render_interactive_html(
    z_vector: np.ndarray,
    n_cortical: int = 400,
    cmap: str = "RdBu_r",
    vmax: float = 2.5,
) -> tuple[str, str]:
    """Generate interactive nilearn surface HTML for both hemispheres.

    Parameters
    ----------
    z_vector : np.ndarray
        Per-parcel BAG z-scores. First ``n_cortical`` entries are cortical.
    n_cortical : int
        Number of cortical parcels.
    cmap : str
        Matplotlib colormap name.
    vmax : float
        Symmetric colormap range (-vmax to +vmax).

    Returns
    -------
    tuple[str, str]
        (lh_html, rh_html) — standalone HTML strings embeddable via
        ``st.components.v1.html()``.
    """
    from nilearn import datasets, plotting  # deferred

    lh_labels, rh_labels = _build_surface_lookup(n_cortical)
    fsavg = datasets.fetch_surf_fsaverage("fsaverage5")

    cort_z = z_vector[:n_cortical]

    # Map parcel z-scores to vertex arrays
    # Schaefer 400: labels 1-200 = LH, 201-400 = RH
    lh_surf = np.full(len(lh_labels), np.nan)
    rh_surf = np.full(len(rh_labels), np.nan)

    for i, z in enumerate(cort_z):
        pid = i + 1  # 1-indexed parcel ID
        lh_mask = lh_labels == pid
        rh_mask = rh_labels == pid
        if lh_mask.any():
            lh_surf[lh_mask] = z
        if rh_mask.any():
            rh_surf[rh_mask] = z

    lh_view = plotting.view_surf(
        fsavg.infl_left,
        lh_surf,
        bg_map=fsavg.sulc_left,
        cmap=cmap,
        vmax=vmax,
        title="Left hemisphere — BAG z-scores",
    )
    rh_view = plotting.view_surf(
        fsavg.infl_right,
        rh_surf,
        bg_map=fsavg.sulc_right,
        cmap=cmap,
        vmax=vmax,
        title="Right hemisphere — BAG z-scores",
    )

    return lh_view.get_standalone(), rh_view.get_standalone()
