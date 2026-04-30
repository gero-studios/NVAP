"""NVAP design system tokens.

Single source of truth for colors, spacing, radius, typography, motion.
Used by theme.py to compose QSS and by widgets that need programmatic access
to design values (icon tinting, dynamic styles).

Defaults to a cohesive dark theme tuned for scientific 3D visualization.
"""
from __future__ import annotations

from dataclasses import dataclass


# ─── COLOR ─────────────────────────────────────────────────────────────────
# Semantic tokens aligned with the scientific dark/gold theme defined in
# theme.py. Surfaces are warm-neutral near-black; accent is a muted brass.
@dataclass(frozen=True)
class Color:
    # Surfaces – ascending depth
    bg_base: str       = "#080A0D"   # outermost (window chrome, status bar)
    bg_canvas: str     = "#101216"   # main canvas (workspace, panels)
    bg_surface: str    = "#151922"   # raised (group boxes, cards)
    bg_surface_2: str  = "#1B202A"   # higher (buttons, hover bg)
    bg_surface_3: str  = "#222936"   # highest (active state)
    bg_overlay: str    = "#080A0DCC" # modal scrim

    # Borders – subtle to prominent
    border_subtle: str = "#202633"
    border_default: str= "#272E3B"
    border_strong: str = "#313947"
    border_focus: str  = "#d8bd78"

    # Text – warm near-white on dark
    text_primary: str  = "#eef2f6"
    text_secondary: str= "#d9dee7"
    text_tertiary: str = "#9aa4b2"
    text_disabled: str = "#56606f"
    text_inverse: str  = "#111318"

    # Brand accent – muted brass / gold
    accent: str        = "#d8bd78"
    accent_hover: str  = "#f0d98d"
    accent_pressed: str= "#b89c5b"
    accent_subtle: str = "#2a2418"   # tinted bg for selected/active

    # Channel colors (microscopy domain)
    channel_red: str       = "#e56f78"
    channel_red_strong: str= "#d04a55"
    channel_red_subtle: str= "#3f1c20"
    channel_green: str     = "#69d9a3"
    channel_green_strong: str = "#3fbf83"
    channel_green_subtle: str = "#1a3a2a"

    # Semantic state
    success: str       = "#69d9a3"
    warning: str       = "#d2a94d"
    danger: str        = "#e56f78"
    info: str          = "#d8bd78"

    # Selection
    selection_bg: str  = "#d8bd78"
    selection_fg: str  = "#111318"


COLOR = Color()


# ─── SPACING ───────────────────────────────────────────────────────────────
# 4-pt grid scale.
@dataclass(frozen=True)
class Spacing:
    xxs: int = 2
    xs: int  = 4
    sm: int  = 8
    md: int  = 12
    lg: int  = 16
    xl: int  = 24
    xxl: int = 32
    xxxl: int= 48
    huge: int= 64


SPACE = Spacing()


# ─── RADIUS ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Radius:
    sm: int  = 4
    md: int  = 6
    lg: int  = 8
    xl: int  = 12
    pill: int= 999


RADIUS = Radius()


# ─── TYPOGRAPHY ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FontSize:
    xs: int  = 10
    sm: int  = 11
    md: int  = 12
    base: int= 13
    lg: int  = 14
    xl: int  = 16
    xxl: int = 20
    display: int = 28
    hero: int    = 44


FONT = FontSize()

FONT_FAMILY_UI    = '"Inter", "Aptos", "Segoe UI", "Helvetica Neue", sans-serif'
FONT_FAMILY_MONO  = '"JetBrains Mono", "Cascadia Mono", "Consolas", monospace'


# ─── MOTION ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Motion:
    fast_ms: int   = 120
    base_ms: int   = 180
    slow_ms: int   = 280


MOTION = Motion()


# ─── ELEVATION (border-only, dark theme has no real shadows in QSS) ────────
ELEVATION_LOW = f"border: 1px solid {COLOR.border_subtle};"
ELEVATION_MED = f"border: 1px solid {COLOR.border_default};"
ELEVATION_HIGH= f"border: 1px solid {COLOR.border_strong};"


# ─── SIZES ─────────────────────────────────────────────────────────────────
SIDEBAR_WIDTH         = 220
SIDEBAR_COLLAPSED     = 56
WORKSPACE_HEADER_H    = 56
LEFT_PANEL_MIN_WIDTH  = 280
LEFT_PANEL_MAX_WIDTH  = 340
RIGHT_PANEL_MIN_WIDTH = 240
RIGHT_PANEL_MAX_WIDTH = 320
ICON_SM               = 14
ICON_MD               = 16
ICON_LG               = 20
ICON_XL               = 24
