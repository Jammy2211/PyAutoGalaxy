"""
Helper functions to extract autogalaxy-specific overlay data from Visuals2D
objects and convert them to plain numpy arrays suitable for plot_array().
"""
from typing import List, Optional

import numpy as np


def _critical_curves_from_visuals(visuals_2d) -> Optional[List[np.ndarray]]:
    """Return list of (N,2) arrays for tangential and radial critical curves."""
    if visuals_2d is None:
        return None
    curves = []
    for attr in ("tangential_critical_curves", "radial_critical_curves"):
        val = getattr(visuals_2d, attr, None)
        if val is None:
            continue
        try:
            for item in val:
                try:
                    arr = np.array(item.array if hasattr(item, "array") else item)
                    if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                        curves.append(arr)
                except Exception:
                    pass
        except TypeError:
            try:
                arr = np.array(val.array if hasattr(val, "array") else val)
                if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                    curves.append(arr)
            except Exception:
                pass
    return curves or None


def _caustics_from_visuals(visuals_2d) -> Optional[List[np.ndarray]]:
    """Return list of (N,2) arrays for tangential and radial caustics."""
    if visuals_2d is None:
        return None
    curves = []
    for attr in ("tangential_caustics", "radial_caustics"):
        val = getattr(visuals_2d, attr, None)
        if val is None:
            continue
        try:
            for item in val:
                try:
                    arr = np.array(item.array if hasattr(item, "array") else item)
                    if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                        curves.append(arr)
                except Exception:
                    pass
        except TypeError:
            try:
                arr = np.array(val.array if hasattr(val, "array") else val)
                if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                    curves.append(arr)
            except Exception:
                pass
    return curves or None


def _galaxy_lines_from_visuals(visuals_2d) -> Optional[List[np.ndarray]]:
    """
    Return all line overlays from an autogalaxy Visuals2D, combining regular
    lines, critical curves, and caustics into a single list.
    """
    if visuals_2d is None:
        return None

    lines = []

    # base lines from Visuals2D
    base_lines = getattr(visuals_2d, "lines", None)
    if base_lines is not None:
        try:
            for item in base_lines:
                try:
                    arr = np.array(item.array if hasattr(item, "array") else item)
                    if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                        lines.append(arr)
                except Exception:
                    pass
        except TypeError:
            try:
                arr = np.array(
                    base_lines.array if hasattr(base_lines, "array") else base_lines
                )
                if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                    lines.append(arr)
            except Exception:
                pass

    critical = _critical_curves_from_visuals(visuals_2d) or []
    caustics = _caustics_from_visuals(visuals_2d) or []
    combined = lines + critical + caustics
    return combined or None


def _galaxy_positions_from_visuals(visuals_2d) -> Optional[List[np.ndarray]]:
    """
    Return all scatter-point overlays from an autogalaxy Visuals2D, combining
    regular positions, light/mass profile centres, and multiple images.
    """
    if visuals_2d is None:
        return None

    result = []

    # base positions from Visuals2D
    base_positions = getattr(visuals_2d, "positions", None)
    if base_positions is not None:
        try:
            for item in base_positions:
                try:
                    arr = np.array(item.array if hasattr(item, "array") else item)
                    if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                        result.append(arr)
                except Exception:
                    pass
        except TypeError:
            try:
                arr = np.array(
                    base_positions.array
                    if hasattr(base_positions, "array")
                    else base_positions
                )
                if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                    result.append(arr)
            except Exception:
                pass

    for attr in ("light_profile_centres", "mass_profile_centres", "multiple_images"):
        val = getattr(visuals_2d, attr, None)
        if val is None:
            continue
        try:
            arr = np.array(val.array if hasattr(val, "array") else val)
            if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                result.append(arr)
        except Exception:
            pass

    return result or None
