#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch UI: load multiple images, run K-frame detection on all, show results in one scrollable grid.
"""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QImage, QPixmap, QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QHeaderView,
    QVBoxLayout,
    QWidget,
    QToolTip,
)

from gui.k_frame_detector import detect_k_frame_box
from gui.k_frame_face_detection import detect_color_faces_in_warp

THUMB_MAX_W = 280
K_FRAME_NORMALIZED_SIZE = 512


class FaceHoverLabel(QLabel):
    """Preview label that shows face area tooltip on hover."""

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self._orig_size: tuple[int, int] | None = None  # (w, h) in source image coordinates
        self._faces: list[np.ndarray] = []
        self._areas: list[float] = []

    def set_face_data(
        self,
        orig_size: tuple[int, int] | None,
        faces: list[np.ndarray] | None,
        areas: list[float] | None,
    ) -> None:
        self._orig_size = orig_size
        self._faces = [] if faces is None else [f.astype(np.float32) for f in faces]
        self._areas = [] if areas is None else [float(a) for a in areas]

    def clear_face_data(self) -> None:
        self._orig_size = None
        self._faces = []
        self._areas = []

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        pm = self.pixmap()
        if pm is None or self._orig_size is None or not self._faces:
            QToolTip.hideText()
            super().mouseMoveEvent(event)
            return

        pm_w = pm.width()
        pm_h = pm.height()
        if pm_w <= 0 or pm_h <= 0:
            QToolTip.hideText()
            super().mouseMoveEvent(event)
            return

        # Pixmap is centered in QLabel when alignment is center.
        x0 = (self.width() - pm_w) / 2.0
        y0 = (self.height() - pm_h) / 2.0
        lx = float(event.position().x())
        ly = float(event.position().y())
        if lx < x0 or ly < y0 or lx >= (x0 + pm_w) or ly >= (y0 + pm_h):
            QToolTip.hideText()
            super().mouseMoveEvent(event)
            return

        src_w, src_h = self._orig_size
        px = (lx - x0) * (src_w / pm_w)
        py = (ly - y0) * (src_h / pm_h)

        hit_idx = -1
        for i, face in enumerate(self._faces):
            contour = face.astype(np.float32).reshape(-1, 1, 2)
            inside = cv2.pointPolygonTest(contour, (float(px), float(py)), False)
            if inside >= 0:
                hit_idx = i
                break

        if hit_idx >= 0:
            area = self._areas[hit_idx] if hit_idx < len(self._areas) else 0.0
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"Face {hit_idx + 1}: {area:.1f} px^2",
                self,
            )
        else:
            QToolTip.hideText()

        super().mouseMoveEvent(event)


class ClickCopyLabel(QLabel):
    """Clickable preview label that copies its image to clipboard."""

    clicked = Signal()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class AreaClusterPlot(QWidget):
    """Simple clickable scatter plot for face-area clustering."""

    pointClicked = Signal(int, int)  # row_index, face_index

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(220)
        self._points: list[tuple[int, int, float, int]] = []  # row, face_idx, area, cluster
        self._draw_points: list[tuple[float, float, int, int]] = []  # x, y, row, face_idx

    @staticmethod
    def _kmeans_1d(values: list[float], k: int = 3, iters: int = 20) -> tuple[list[int], list[float]]:
        if not values:
            return [], []
        unique_sorted = sorted(set(values))
        k = max(1, min(k, len(unique_sorted)))
        # deterministic init: spread over sorted unique values
        if k == 1:
            centroids = [float(np.mean(values))]
        else:
            idxs = [int(round(i * (len(unique_sorted) - 1) / (k - 1))) for i in range(k)]
            centroids = [float(unique_sorted[i]) for i in idxs]
        labels = [0] * len(values)
        for _ in range(iters):
            changed = False
            for i, v in enumerate(values):
                dists = [abs(v - c) for c in centroids]
                lbl = int(np.argmin(dists))
                if lbl != labels[i]:
                    labels[i] = lbl
                    changed = True
            for c in range(k):
                members = [values[i] for i, l in enumerate(labels) if l == c]
                if members:
                    centroids[c] = float(np.mean(members))
            if not changed:
                break
        return labels, centroids

    def set_area_points(self, points: list[tuple[int, int, float]]) -> None:
        """
        points: list of (row_index, face_index, area)
        """
        self._points.clear()
        if not points:
            self.update()
            return
        areas = [float(p[2]) for p in points]
        labels, _ = self._kmeans_1d(areas, k=3, iters=25)
        for i, (row, face_idx, area) in enumerate(points):
            self._points.append((row, face_idx, float(area), labels[i] if i < len(labels) else 0))
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#1f1f1f"))

        margin_l, margin_r, margin_t, margin_b = 40, 20, 20, 30
        w = max(1, self.width() - margin_l - margin_r)
        h = max(1, self.height() - margin_t - margin_b)
        ox, oy = margin_l, margin_t + h

        # axes
        axis_pen = QPen(QColor("#888"))
        axis_pen.setWidth(1)
        painter.setPen(axis_pen)
        painter.drawLine(ox, oy, ox + w, oy)
        painter.drawLine(ox, oy, ox, margin_t)

        if not self._points:
            painter.setPen(QColor("#aaa"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No clustered points yet")
            self._draw_points = []
            return

        areas = [p[2] for p in self._points]
        min_a = min(areas)
        max_a = max(areas)
        if math.isclose(min_a, max_a):
            max_a = min_a + 1.0

        palette = [QColor("#4FC3F7"), QColor("#FFB74D"), QColor("#81C784"), QColor("#BA68C8")]
        self._draw_points = []

        # plot by sorted area index on x
        pts_sorted = sorted(self._points, key=lambda t: t[2])
        n = len(pts_sorted)
        for i, (row, face_idx, area, cluster) in enumerate(pts_sorted):
            x = ox + (w * (i + 0.5) / max(1, n))
            y_norm = (area - min_a) / (max_a - min_a)
            y = oy - (h * y_norm)
            color = palette[cluster % len(palette)]
            painter.setPen(QPen(QColor("#111"), 1))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(int(x - 5), int(y - 5), 10, 10)
            self._draw_points.append((x, y, row, face_idx))

        painter.setPen(QColor("#bbb"))
        painter.drawText(8, margin_t + 6, f"{max_a:.1f}")
        painter.drawText(8, oy, f"{min_a:.1f}")
        painter.drawText(ox + 4, self.height() - 8, "clustered face areas")

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if not self._draw_points:
            return super().mousePressEvent(event)
        x = float(event.position().x())
        y = float(event.position().y())
        best = None
        best_d2 = 10.0 * 10.0
        for px, py, row, face_idx in self._draw_points:
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best = (row, face_idx)
        if best is not None:
            self.pointClicked.emit(best[0], best[1])
        super().mousePressEvent(event)


def bgr_to_qpixmap(bgr: np.ndarray, max_w: int = THUMB_MAX_W) -> QPixmap:
    h, w = bgr.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        nh = max(1, int(h * scale))
        bgr = cv2.resize(bgr, (max_w, nh), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

def run_detection_on_bgr(
    img: np.ndarray,
) -> tuple[np.ndarray | None, str, np.ndarray, float | None, list[np.ndarray], list[float]]:
    """
    K-frame 검출 -> K-frame을 정사각형으로 warp -> warp 내부에서 색상 면(총 4개)을 검출합니다.
    """

    def warp_k_frame_to_square(bgr: np.ndarray, k_corners: np.ndarray) -> np.ndarray:
        pts = k_corners.astype(np.float32)
        side = int(K_FRAME_NORMALIZED_SIZE)

        dst = np.array(
            [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(bgr, M, (side, side))

    k_corners = detect_k_frame_box(img)
    if k_corners is None:
        return None, "K not found", img, None, [], []

    warped = warp_k_frame_to_square(img, k_corners)
    faces = detect_color_faces_in_warp(warped)

    overlay = warped.copy()
    draw_colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
    for idx, rect in enumerate(faces):
        color = draw_colors[idx % len(draw_colors)]
        pts = rect.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
        cx = int(np.mean(rect[:, 0]))
        cy = int(np.mean(rect[:, 1]))
        cv2.circle(overlay, (cx, cy), 3, (0, 0, 0), -1)
        cv2.circle(overlay, (cx, cy), 2, color, -1)

    side = overlay.shape[0]
    cv2.rectangle(overlay, (0, 0), (side - 1, side - 1), (120, 120, 120), 2)

    detail = f"faces found: {len(faces)}"
    # Face mean area(px^2): compute polygon area for each ordered face quad.
    face_areas: list[float] = []
    for rect in faces:
        quad = rect.astype(np.float32).reshape(-1, 1, 2)
        face_areas.append(float(cv2.contourArea(quad)))
    face_mean_area: float | None = None
    if face_areas:
        face_mean_area = float(np.mean(face_areas))

    return k_corners, detail, overlay, face_mean_area, faces, face_areas


class KFrameBatchWorker(QThread):
    finished_one = Signal(int, object, str, str, bool, object, object, object, object)
    all_done = Signal()

    def __init__(self, paths: list[str]):
        super().__init__()
        self.paths = paths
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        for i, path in enumerate(self.paths):
            if self._cancel:
                break
            try:
                img = cv2.imread(path)
                if img is None:
                    self.finished_one.emit(i, None, path, "Failed to load image", False, None, None, None, None)
                    continue
                k_corners, face_detail, overlay_bgr, face_mean_area, faces, face_areas = run_detection_on_bgr(img)
                ok = k_corners is not None
                detail = face_detail if ok else face_detail
                corners_out = None if k_corners is None else k_corners.copy()
                self.finished_one.emit(
                    i,
                    overlay_bgr.copy(),
                    path,
                    detail,
                    ok,
                    corners_out,
                    face_mean_area,
                    faces,
                    face_areas,
                )
            except Exception as e:
                self.finished_one.emit(i, None, path, f"Error: {e}", False, None, None, None, None)
        self.all_done.emit()


class KFrameBatchWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("K-Frame Batch Detection")
        self.setGeometry(80, 80, 1280, 820)
        self.paths: list[str] = []
        self.cells: list[tuple[ClickCopyLabel, QLabel]] = []
        self.worker: KFrameBatchWorker | None = None
        self._ok_count = 0
        self._fail_count = 0
        self._preview_cache: dict[int, np.ndarray] = {}
        self._face_meta_cache: dict[int, tuple[list[np.ndarray], list[float], tuple[int, int]]] = {}

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        bar = QHBoxLayout()
        self.btn_add = QPushButton("Add images…")
        self.btn_add.clicked.connect(self.add_images)
        self.btn_add_folder = QPushButton("Add folder…")
        self.btn_add_folder.clicked.connect(self.add_folder)
        self.btn_run = QPushButton("Run detection")
        self.btn_run.clicked.connect(self.run_detection)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_all)
        bar.addWidget(self.btn_add)
        bar.addWidget(self.btn_add_folder)
        bar.addWidget(self.btn_run)
        bar.addWidget(self.btn_clear)
        bar.addStretch()
        root.addLayout(bar)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self.summary = QLabel("No images loaded.")
        self.summary.setAlignment(Qt.AlignLeft)
        root.addWidget(self.summary)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        self.preview_tab = QWidget()
        preview_layout = QVBoxLayout(self.preview_tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.grid_host = QWidget()
        self.grid_layout = QGridLayout(self.grid_host)
        self.grid_layout.setSpacing(12)
        scroll.setWidget(self.grid_host)
        preview_layout.addWidget(scroll)
        self.tabs.addTab(self.preview_tab, "Preview")

        self.stats_tab = QWidget()
        stats_layout = QVBoxLayout(self.stats_tab)
        self.stats_table = QTableWidget(0, 3)
        self.stats_table.setHorizontalHeaderLabels(
            ["Image", "Status", "Face Mean Area(px2)"]
        )
        header = self.stats_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stats_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.stats_table.cellClicked.connect(self._on_stats_row_clicked)
        stats_layout.addWidget(self.stats_table)

        self.cluster_plot = AreaClusterPlot()
        self.cluster_plot.pointClicked.connect(self._on_cluster_point_clicked)
        stats_layout.addWidget(self.cluster_plot)

        self.stats_selected_name = QLabel("Select a row to preview image.")
        self.stats_selected_name.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.stats_selected_name)

        self.stats_selected_preview = FaceHoverLabel()
        self.stats_selected_preview.setAlignment(Qt.AlignCenter)
        self.stats_selected_preview.setMinimumHeight(220)
        self.stats_selected_preview.setStyleSheet("background:#1f1f1f; color:#888;")
        self.stats_selected_preview.setText("No image selected")
        stats_layout.addWidget(self.stats_selected_preview)
        self.tabs.addTab(self.stats_tab, "Line Stats")

    def add_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select images",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
        )
        if not files:
            return
        for f in files:
            if f not in self.paths:
                self.paths.append(f)
        self.rebuild_grid()
        self.summary.setText(f"{len(self.paths)} image(s) — click Run detection.")

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder", "")
        if not folder:
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        added = 0
        for p in Path(folder).rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            fp = str(p)
            if fp not in self.paths:
                self.paths.append(fp)
                added += 1
        self.rebuild_grid()
        self.summary.setText(f"{len(self.paths)} image(s) — added {added} from folder.")

    def clear_all(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        self.paths.clear()
        self._preview_cache.clear()
        self._face_meta_cache.clear()
        self._clear_grid_widgets()
        self.cells.clear()
        self._clear_stats_table()
        self.stats_selected_name.setText("Select a row to preview image.")
        self.stats_selected_preview.setPixmap(QPixmap())
        self.stats_selected_preview.setText("No image selected")
        self.stats_selected_preview.clear_face_data()
        self.cluster_plot.set_area_points([])
        self.summary.setText("No images loaded.")

    def _clear_grid_widgets(self) -> None:
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def rebuild_grid(self) -> None:
        self._clear_grid_widgets()
        self.cells.clear()
        self._build_stats_table()
        cols = 4
        for idx, path in enumerate(self.paths):
            box = QGroupBox()
            v = QVBoxLayout(box)
            img_lbl = ClickCopyLabel()
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setMinimumSize(THUMB_MAX_W, 120)
            img_lbl.setStyleSheet("background:#2a2a2a; color:#888;")
            img_lbl.setText("Pending")
            img_lbl.clicked.connect(lambda _=None, row=idx: self._copy_preview_image(row))
            cap = QLabel(os.path.basename(path))
            cap.setWordWrap(True)
            cap.setStyleSheet("font-size:11px;")
            v.addWidget(img_lbl)
            v.addWidget(cap)
            r, c = divmod(idx, cols)
            self.grid_layout.addWidget(box, r, c)
            self.cells.append((img_lbl, cap))
        self.summary.setText(f"{len(self.paths)} image(s) — click Run detection.")

    def _copy_preview_image(self, row: int) -> None:
        if row < 0 or row >= len(self.paths):
            return
        bgr = self._preview_cache.get(row)
        if bgr is None:
            bgr = cv2.imread(self.paths[row])
        if bgr is None:
            self.statusBar().showMessage("Copy failed: cannot load image")
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        QApplication.clipboard().setImage(qimg)
        self.statusBar().showMessage(f"Copied image: {os.path.basename(self.paths[row])}")

    def _clear_stats_table(self) -> None:
        self.stats_table.setRowCount(0)

    def _build_stats_table(self) -> None:
        self._clear_stats_table()
        self.stats_table.setRowCount(len(self.paths))
        for i, path in enumerate(self.paths):
            self.stats_table.setItem(i, 0, QTableWidgetItem(os.path.basename(path)))
            self.stats_table.setItem(i, 1, QTableWidgetItem("Pending"))
            self.stats_table.setItem(i, 2, QTableWidgetItem("-"))
        self._update_cluster_plot()

    @staticmethod
    def _edge_lengths(corners: np.ndarray) -> list[float]:
        pts = corners.astype(np.float32)
        lengths: list[float] = []
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i + 1) % 4]
            lengths.append(float(np.linalg.norm(p1 - p0)))
        return lengths

    def _update_cluster_plot(self) -> None:
        points: list[tuple[int, int, float]] = []
        for row, meta in self._face_meta_cache.items():
            faces, areas, _orig_size = meta
            for face_idx, area in enumerate(areas):
                points.append((row, face_idx, float(area)))
        self.cluster_plot.set_area_points(points)

    def run_detection(self) -> None:
        if not self.paths:
            QMessageBox.information(self, "K-Frame batch", "Add at least one image.")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "K-Frame batch", "Detection is already running.")
            return

        self.rebuild_grid()
        self._ok_count = 0
        self._fail_count = 0
        self.progress.setVisible(True)
        self.progress.setRange(0, len(self.paths))
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_add_folder.setEnabled(False)

        self.worker = KFrameBatchWorker(self.paths)
        self.worker.finished_one.connect(self._on_one_done)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.start()

    def _on_one_done(
        self,
        index: int,
        bgr: object,
        path: str,
        detail: str,
        ok: bool,
        corners: object,
        face_mean_area: object,
        faces: object,
        face_areas: object,
    ) -> None:
        self.progress.setValue(index + 1)
        if index >= len(self.cells):
            return
        if ok:
            self._ok_count += 1
        else:
            self._fail_count += 1
        img_lbl, cap = self.cells[index]
        if bgr is None:
            img_lbl.setText("Error")
            img_lbl.setPixmap(QPixmap())
            if index in self._preview_cache:
                del self._preview_cache[index]
            if index in self._face_meta_cache:
                del self._face_meta_cache[index]
        else:
            assert isinstance(bgr, np.ndarray)
            self._preview_cache[index] = bgr.copy()
            if isinstance(faces, list) and isinstance(face_areas, list):
                self._face_meta_cache[index] = (
                    [np.array(f, dtype=np.float32) for f in faces],
                    [float(a) for a in face_areas],
                    (int(bgr.shape[1]), int(bgr.shape[0])),
                )
            img_lbl.setPixmap(bgr_to_qpixmap(bgr))
            img_lbl.setText("")
        name = os.path.basename(path)
        status_color = "#0a0" if ok else "#c00"
        cap.setText(
            f'<span style="color:{status_color}; font-weight:bold;">'
            f'{"OK" if ok else "FAIL"}</span> {name}<br/><span style="color:#555;">{detail}</span>'
        )
        cap.setTextFormat(Qt.RichText)
        if index < self.stats_table.rowCount():
            self.stats_table.item(index, 1).setText("OK" if ok else "FAIL")
            if ok and face_mean_area is not None:
                self.stats_table.item(index, 2).setText(f"{float(face_mean_area):.1f}")
            else:
                self.stats_table.item(index, 2).setText("-")
        self._update_cluster_plot()

    def _on_stats_row_clicked(self, row: int, _column: int) -> None:
        self._show_stats_preview(row, highlight_face_idx=None)

    def _on_cluster_point_clicked(self, row: int, face_idx: int) -> None:
        # Sync table selection and jump preview to clicked clustered face.
        if 0 <= row < self.stats_table.rowCount():
            self.stats_table.selectRow(row)
        self._show_stats_preview(row, highlight_face_idx=face_idx)
        self.tabs.setCurrentWidget(self.stats_tab)

    def _show_stats_preview(self, row: int, highlight_face_idx: int | None) -> None:
        if row < 0 or row >= len(self.paths):
            return
        path = self.paths[row]
        name = os.path.basename(path)

        bgr = self._preview_cache.get(row)
        if bgr is None:
            bgr = cv2.imread(path)
        if bgr is None:
            self.stats_selected_preview.setPixmap(QPixmap())
            self.stats_selected_preview.setText("Failed to load image")
            self.stats_selected_name.setText(name)
            self.stats_selected_preview.clear_face_data()
            return

        draw = bgr.copy()
        face_meta = self._face_meta_cache.get(row)
        if face_meta is None:
            self.stats_selected_name.setText(name)
            self.stats_selected_preview.clear_face_data()
        else:
            faces, areas, orig_size = face_meta
            self.stats_selected_preview.set_face_data(orig_size, faces, areas)
            if highlight_face_idx is not None and 0 <= highlight_face_idx < len(faces):
                face = faces[highlight_face_idx].astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(draw, [face], isClosed=True, color=(0, 0, 255), thickness=3)
                self.stats_selected_name.setText(
                    f"{name} — selected face {highlight_face_idx + 1} ({areas[highlight_face_idx]:.1f} px^2)"
                )
            else:
                self.stats_selected_name.setText(name)

        pix = bgr_to_qpixmap(draw, max_w=900)
        self.stats_selected_preview.setText("")
        self.stats_selected_preview.setPixmap(pix)

    def _on_all_done(self) -> None:
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_add.setEnabled(True)
        self.btn_add_folder.setEnabled(True)
        self.summary.setText(
            f"Done: {len(self.paths)} image(s) — OK: {self._ok_count}, FAIL: {self._fail_count}"
        )


def main() -> None:
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 11))
    w = KFrameBatchWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
