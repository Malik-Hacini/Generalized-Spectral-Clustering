"""Shared method style configuration for plotting scripts."""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_METHOD_ORDER = [
    "SC-UN",
    "SC-N",
    "DSC+",
    "Chung",
    "DI-SIM-R",
    "DI-SIM-L",
    "DI-SIM-C",
    "GSC-UN",
    "GSC-N",
]

METHOD_STYLES: dict[str, dict[str, str]] = {
    "SC-UN": {"color": "#FF8C69", "linestyle": ":", "marker": "D", "label": "SC-UN"},
    "SC-N": {"color": "#FF6347", "linestyle": "--", "marker": "o", "label": "SC-N"},
    "DSC+": {"color": "#27A727", "linestyle": "-.", "marker": "^", "label": "DSC+"},
    "Chung": {"color": "#008F00", "linestyle": ":", "marker": "v", "label": "Chung"},
    "DI-SIM-R": {"color": "#7A3E9D", "linestyle": "-", "marker": "P", "label": "DI-SIM-R"},
    "DI-SIM-L": {"color": "#A55CC2", "linestyle": "--", "marker": "X", "label": "DI-SIM-L"},
    "DI-SIM-C": {"color": "#C084D8", "linestyle": "-.", "marker": "*", "label": "DI-SIM-C"},
    "GSC-UN": {"color": "#4C9AFF", "linestyle": "--", "marker": "P", "label": "GSC-UN"},
    "GSC-N": {"color": "#072AC8", "linestyle": "-", "marker": "s", "label": "GSC-N"},
}

_FALLBACK_COLORS = ["#6C757D", "#8C564B", "#BCBD22", "#17BECF"]
_FALLBACK_LINESTYLES = ["-", "--", "-.", ":"]
_FALLBACK_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "<", ">", "h", "*"]


def ordered_methods(
    available_methods: Iterable[str],
    preferred_order: list[str] | None = None,
) -> list[str]:
    """Return methods sorted by preferred order, then remaining methods alphabetically."""
    preferred = DEFAULT_METHOD_ORDER if preferred_order is None else preferred_order
    available = list(dict.fromkeys(available_methods))
    selected = [method for method in preferred if method in available]
    extras = sorted(method for method in available if method not in selected)
    return selected + extras


def style_for_method(method: str) -> dict[str, str]:
    """Return style mapping for method with deterministic fallback."""
    style = METHOD_STYLES.get(method)
    if style is not None:
        return dict(style)

    fallback_idx = sum(ord(ch) for ch in method)
    return {
        "color": _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)],
        "linestyle": _FALLBACK_LINESTYLES[fallback_idx % len(_FALLBACK_LINESTYLES)],
        "marker": _FALLBACK_MARKERS[fallback_idx % len(_FALLBACK_MARKERS)],
        "label": method,
    }


def styles_for_methods(methods: Iterable[str]) -> dict[str, dict[str, str]]:
    """Return style map for a set/list of methods."""
    return {method: style_for_method(method) for method in methods}
