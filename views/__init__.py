"""
Opti-Route Views Package
Phase 3: GUI Canvas, Dual Themes, and Animated Traversal
"""

import sys
import types


def ensure_qt_bridge() -> bool:
    """
    Ensure PyQt6 is importable. If PyQt6 is not installed in the current environment
    but PyQt5 is available, transparently bridge PyQt6 module and enum namespaces
    to PyQt5 so the full desktop GUI application runs out-of-the-box.
    """
    try:
        import PyQt6.QtWidgets  # noqa: F401
        return True
    except ImportError:
        pass

    try:
        import PyQt5
        import PyQt5.QtCore as _QtCore
        import PyQt5.QtGui as _QtGui
        import PyQt5.QtWidgets as _QtWidgets
    except ImportError:
        return False

    # Bridge PyQt6 scoped enum namespaces onto PyQt5 classes
    _Qt = _QtCore.Qt
    for enum_alias in (
        "Orientation",
        "ScrollBarPolicy",
        "PenStyle",
        "PenCapStyle",
        "PenJoinStyle",
        "AspectRatioMode",
        "AlignmentFlag",
        "TextFlag",
    ):
        if not hasattr(_Qt, enum_alias):
            setattr(_Qt, enum_alias, _Qt)

    if not hasattr(_QtGui, "QAction") and hasattr(_QtWidgets, "QAction"):
        setattr(_QtGui, "QAction", _QtWidgets.QAction)

    if not hasattr(_QtGui.QPainter, "RenderHint"):
        setattr(_QtGui.QPainter, "RenderHint", _QtGui.QPainter)

    if not hasattr(_QtGui.QFont, "Weight"):
        setattr(_QtGui.QFont, "Weight", _QtGui.QFont)

    if not hasattr(_QtWidgets.QGraphicsItem, "GraphicsItemFlag"):
        setattr(_QtWidgets.QGraphicsItem, "GraphicsItemFlag", _QtWidgets.QGraphicsItem)

    if not hasattr(_QtWidgets.QGraphicsItem, "GraphicsItemChange"):
        setattr(_QtWidgets.QGraphicsItem, "GraphicsItemChange", _QtWidgets.QGraphicsItem)

    if not hasattr(_QtWidgets.QGraphicsView, "DragMode"):
        setattr(_QtWidgets.QGraphicsView, "DragMode", _QtWidgets.QGraphicsView)

    if not hasattr(_QtWidgets.QGraphicsView, "ViewportAnchor"):
        setattr(_QtWidgets.QGraphicsView, "ViewportAnchor", _QtWidgets.QGraphicsView)

    if not hasattr(_QtWidgets.QHeaderView, "ResizeMode"):
        setattr(_QtWidgets.QHeaderView, "ResizeMode", _QtWidgets.QHeaderView)

    if not hasattr(_QtWidgets.QApplication, "exec") and hasattr(_QtWidgets.QApplication, "exec_"):
        setattr(_QtWidgets.QApplication, "exec", _QtWidgets.QApplication.exec_)

    pyqt6_pkg = types.ModuleType("PyQt6")
    pyqt6_pkg.QtCore = _QtCore  # type: ignore
    pyqt6_pkg.QtGui = _QtGui  # type: ignore
    pyqt6_pkg.QtWidgets = _QtWidgets  # type: ignore

    sys.modules["PyQt6"] = pyqt6_pkg
    sys.modules["PyQt6.QtCore"] = _QtCore
    sys.modules["PyQt6.QtGui"] = _QtGui
    sys.modules["PyQt6.QtWidgets"] = _QtWidgets
    return True


ensure_qt_bridge()

from .theme_manager import ThemeManager, ThemeMode
from .map_canvas import MapCanvas
from .main_window import OptiRouteMainWindow

__all__ = [
    "ThemeManager",
    "ThemeMode",
    "MapCanvas",
    "OptiRouteMainWindow",
    "ensure_qt_bridge",
]
