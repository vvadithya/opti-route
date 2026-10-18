"""
Opti-Route Theme Manager
Phase 3 Subsystem: Dual Themes System (SRS F11 & Use Case U3)

Provides centralized theme definitions, QSS stylesheets, and color palettes
supporting one-click switching between Dark Theme (#1E1E1E) and Light Theme (#FAFAFA).
"""

from enum import Enum
from typing import Callable, Dict, List, Optional

try:
    from PyQt6.QtCore import QObject, pyqtSignal
    from PyQt6.QtGui import QColor
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    # Fallback dummy class if PyQt6 is not installed in the inspection environment
    class QObject:  # type: ignore
        pass


class ThemeMode(str, Enum):
    """Enumeration of supported theme modes."""
    DARK = "dark"
    LIGHT = "light"


class ThemeManager(QObject if HAS_PYQT else object):
    """
    Centralized Theme Manager for Opti-Route.
    Manages dual-theme state, color palettes, and QSS style sheet generation.
    """

    if HAS_PYQT:
        theme_changed = pyqtSignal(str)

    DARK_PALETTE: Dict[str, str] = {
        # Backgrounds
        "canvas_bg": "#1E1E1E",
        "window_bg": "#18181B",
        "panel_bg": "#27272A",
        "panel_border": "#3F3F46",
        "card_bg": "#2D2D30",
        "input_bg": "#1F1F23",
        # Typography
        "text_primary": "#F4F4F5",
        "text_secondary": "#A1A1AA",
        "text_muted": "#71717A",
        # Accents & Brand
        "accent": "#00E5FF",
        "accent_hover": "#33EBFF",
        "accent_pressed": "#00B8CC",
        "accent_text": "#09090B",
        # Graph Elements
        "node_fill": "#2563EB",
        "node_stroke": "#60A5FA",
        "node_hover": "#38BDF8",
        "node_selected": "#F59E0B",
        "node_text": "#FFFFFF",
        "depot_fill": "#DC2626",
        "depot_stroke": "#F87171",
        "edge_line": "#52525B",
        "edge_hover": "#A1A1AA",
        "edge_weight_text": "#A1A1AA",
        "edge_weight_bg": "#18181B",
        # Overlays & Traversal
        "route_line": "#10B981",
        "route_glow": "#34D399",
        "vehicle_fill": "#F59E0B",
        "vehicle_stroke": "#FDE68A",
        # Status / Feedback
        "success": "#10B981",
        "warning": "#F59E0B",
        "error": "#EF4444",
        "info": "#00E5FF",
    }

    LIGHT_PALETTE: Dict[str, str] = {
        # Backgrounds
        "canvas_bg": "#FAFAFA",
        "window_bg": "#F4F4F5",
        "panel_bg": "#FFFFFF",
        "panel_border": "#E4E4E7",
        "card_bg": "#FFFFFF",
        "input_bg": "#F8FAFC",
        # Typography
        "text_primary": "#09090B",
        "text_secondary": "#52525B",
        "text_muted": "#A1A1AA",
        # Accents & Brand
        "accent": "#0284C7",
        "accent_hover": "#0369A1",
        "accent_pressed": "#075985",
        "accent_text": "#FFFFFF",
        # Graph Elements
        "node_fill": "#3B82F6",
        "node_stroke": "#1D4ED8",
        "node_hover": "#60A5FA",
        "node_selected": "#D97706",
        "node_text": "#FFFFFF",
        "depot_fill": "#E11D48",
        "depot_stroke": "#9F1239",
        "edge_line": "#CBD5E1",
        "edge_hover": "#64748B",
        "edge_weight_text": "#475569",
        "edge_weight_bg": "#F1F5F9",
        # Overlays & Traversal
        "route_line": "#059669",
        "route_glow": "#10B981",
        "vehicle_fill": "#D97706",
        "vehicle_stroke": "#78350F",
        # Status / Feedback
        "success": "#059669",
        "warning": "#D97706",
        "error": "#DC2626",
        "info": "#0284C7",
    }

    def __init__(self, initial_mode: ThemeMode = ThemeMode.DARK):
        if HAS_PYQT:
            super().__init__()
        self._current_mode: ThemeMode = initial_mode
        self._listeners: List[Callable[[ThemeMode], None]] = []

    @property
    def current_mode(self) -> ThemeMode:
        """Get the active ThemeMode."""
        return self._current_mode

    @property
    def is_dark(self) -> bool:
        """Check if active theme is Dark."""
        return self._current_mode == ThemeMode.DARK

    def toggle_theme(self) -> ThemeMode:
        """Toggle between Dark and Light themes."""
        new_mode = ThemeMode.LIGHT if self._current_mode == ThemeMode.DARK else ThemeMode.DARK
        self.set_theme(new_mode)
        return self._current_mode

    def set_theme(self, mode: ThemeMode) -> None:
        """Set the active theme mode and notify listeners."""
        if self._current_mode != mode:
            self._current_mode = mode
            self._notify_change()

    def register_listener(self, callback: Callable[[ThemeMode], None]) -> None:
        """Register a callback to be called on theme change."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def unregister_listener(self, callback: Callable[[ThemeMode], None]) -> None:
        """Unregister a previously registered callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_change(self) -> None:
        """Emit Qt signal and notify registered listener callbacks."""
        if HAS_PYQT:
            self.theme_changed.emit(self._current_mode.value)
        for listener in self._listeners:
            listener(self._current_mode)

    def get_palette(self) -> Dict[str, str]:
        """Return color dictionary for active theme."""
        return self.DARK_PALETTE if self.is_dark else self.LIGHT_PALETTE

    def get_color(self, key: str) -> str:
        """Get hex color string by key from active palette."""
        return self.get_palette().get(key, "#000000")

    def get_qcolor(self, key: str):
        """Return QColor instance for the given palette key."""
        if not HAS_PYQT:
            return None
        hex_val = self.get_color(key)
        return QColor(hex_val)

    def get_stylesheet(self) -> str:
        """
        Generate complete Qt Style Sheet (QSS) for desktop application widgets
        adhering to the active theme palette.
        """
        p = self.get_palette()
        return f"""
        QMainWindow, QWidget {{
            background-color: {p['window_bg']};
            color: {p['text_primary']};
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 13px;
        }}

        QFrame#control_panel, QFrame#sidebar {{
            background-color: {p['panel_bg']};
            border-right: 1px solid {p['panel_border']};
        }}

        QFrame#card {{
            background-color: {p['card_bg']};
            border: 1px solid {p['panel_border']};
            border-radius: 8px;
            padding: 12px;
        }}

        QGroupBox {{
            background-color: {p['card_bg']};
            border: 1px solid {p['panel_border']};
            border-radius: 6px;
            margin-top: 18px;
            padding-top: 14px;
            padding-left: 8px;
            padding-right: 8px;
            padding-bottom: 8px;
            font-weight: 600;
            color: {p['text_primary']};
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 0 4px;
            color: {p['accent']};
        }}

        QPushButton {{
            background-color: {p['card_bg']};
            color: {p['text_primary']};
            border: 1px solid {p['panel_border']};
            border-radius: 6px;
            padding: 7px 14px;
            font-weight: 500;
        }}

        QPushButton:hover {{
            background-color: {p['panel_border']};
            color: {p['text_primary']};
        }}

        QPushButton:pressed {{
            background-color: {p['input_bg']};
        }}

        QPushButton#primary_button {{
            background-color: {p['accent']};
            color: {p['accent_text']};
            border: none;
            font-weight: 600;
        }}

        QPushButton#primary_button:hover {{
            background-color: {p['accent_hover']};
        }}

        QPushButton#primary_button:pressed {{
            background-color: {p['accent_pressed']};
        }}

        QPushButton#toolbar_button {{
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 4px;
            padding: 4px 10px;
        }}

        QPushButton#toolbar_button:hover {{
            background-color: {p['panel_border']};
            border-color: {p['panel_border']};
        }}

        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background-color: {p['input_bg']};
            color: {p['text_primary']};
            border: 1px solid {p['panel_border']};
            border-radius: 5px;
            padding: 6px 10px;
            selection-background-color: {p['accent']};
            selection-color: {p['accent_text']};
        }}

        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {p['accent']};
        }}

        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid {p['panel_border']};
        }}

        QComboBox QAbstractItemView {{
            background-color: {p['panel_bg']};
            color: {p['text_primary']};
            border: 1px solid {p['panel_border']};
            selection-background-color: {p['accent']};
            selection-color: {p['accent_text']};
        }}

        QSlider::groove:horizontal {{
            height: 6px;
            background: {p['panel_border']};
            border-radius: 3px;
        }}

        QSlider::sub-page:horizontal {{
            background: {p['accent']};
            border-radius: 3px;
        }}

        QSlider::handle:horizontal {{
            background: {p['text_primary']};
            border: 2px solid {p['accent']};
            width: 14px;
            margin-top: -4px;
            margin-bottom: -4px;
            border-radius: 7px;
        }}

        QProgressBar {{
            background-color: {p['input_bg']};
            border: 1px solid {p['panel_border']};
            border-radius: 4px;
            text-align: center;
            color: {p['text_primary']};
        }}

        QProgressBar::chunk {{
            background-color: {p['accent']};
            border-radius: 3px;
        }}

        QToolBar {{
            background-color: {p['panel_bg']};
            border-bottom: 1px solid {p['panel_border']};
            padding: 4px;
            spacing: 6px;
        }}

        QStatusBar {{
            background-color: {p['panel_bg']};
            border-top: 1px solid {p['panel_border']};
            color: {p['text_secondary']};
        }}

        QTabWidget::pane {{
            border: 1px solid {p['panel_border']};
            background-color: {p['panel_bg']};
            border-radius: 4px;
        }}

        QTabBar::tab {{
            background-color: {p['card_bg']};
            color: {p['text_secondary']};
            border: 1px solid {p['panel_border']};
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}

        QTabBar::tab:selected {{
            background-color: {p['panel_bg']};
            color: {p['accent']};
            border-bottom-color: {p['panel_bg']};
            font-weight: 600;
        }}

        QLabel {{
            color: {p['text_primary']};
        }}

        QLabel#muted_label {{
            color: {p['text_muted']};
        }}

        QLabel#heading_label {{
            font-size: 15px;
            font-weight: bold;
            color: {p['text_primary']};
        }}
        """
