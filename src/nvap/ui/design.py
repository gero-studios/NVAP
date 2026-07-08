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
    bg_base: str       = "#07090c"   # outermost (window chrome, status bar)
    bg_canvas: str     = "#0b0e12"   # main canvas (workspace, panels)
    bg_surface: str    = "#10141a"   # raised (group boxes, cards)
    bg_surface_2: str  = "#151a21"   # higher (buttons, hover bg)
    bg_surface_3: str  = "#1b222b"   # highest (active state)
    bg_overlay: str    = "#07090ccc" # modal scrim

    # Borders – subtle to prominent
    border_subtle: str = "#1a2028"
    border_default: str= "#242b35"
    border_strong: str = "#343d49"
    border_focus: str  = "#9fb7c8"

    # Text – warm near-white on dark
    text_primary: str  = "#edf2f7"
    text_secondary: str= "#cfd7df"
    text_tertiary: str = "#8d98a5"
    text_disabled: str = "#4e5966"
    text_inverse: str  = "#07090c"

    # Brand accent – muted brass / gold
    accent: str        = "#9fb7c8"
    accent_hover: str  = "#c5d7e3"
    accent_pressed: str= "#7792a6"
    accent_subtle: str = "#14202a"   # tinted bg for selected/active

    # Channel colors (microscopy domain)
    channel_red: str       = "#dd6872"
    channel_red_strong: str= "#c94d58"
    channel_red_subtle: str= "#2b1418"
    channel_green: str     = "#6fd6a0"
    channel_green_strong: str = "#48bf82"
    channel_green_subtle: str = "#13281f"

    # Semantic state
    success: str       = "#6fd6a0"
    warning: str       = "#caa65a"
    danger: str        = "#dd6872"
    info: str          = "#9fb7c8"

    # Selection
    selection_bg: str  = "#9fb7c8"
    selection_fg: str  = "#07090c"


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
    sm: int  = 1
    md: int  = 2
    lg: int  = 3
    xl: int  = 4
    pill: int= 4


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
