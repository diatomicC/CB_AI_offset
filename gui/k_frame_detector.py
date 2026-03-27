#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K frame (registration square) detection — shape-based quadrilateral finder.

Used by the main analyzer and the batch K-frame preview tool.
"""

from __future__ import annotations

import cv2
import numpy as np

from color_registration_analysis import order_points


def _normalized(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def _edge_support_score(edge_map: np.ndarray, pts: np.ndarray) -> float:
    """Measure how well 4 square edges align with edge pixels."""
    if edge_map is None or edge_map.size == 0:
        return 0.0
    probe = cv2.dilate(edge_map, np.ones((3, 3), np.uint8), iterations=1)
    pts_i = pts.astype(np.float32)
    hit = 0
    total = 0
    for i in range(4):
        p0 = pts_i[i]
        p1 = pts_i[(i + 1) % 4]
        length = float(np.linalg.norm(p1 - p0))
        steps = max(8, int(length // 4))
        for t in np.linspace(0.0, 1.0, steps):
            p = p0 + (p1 - p0) * float(t)
            x = int(round(float(p[0])))
            y = int(round(float(p[1])))
            if x < 0 or y < 0 or y >= probe.shape[0] or x >= probe.shape[1]:
                continue
            total += 1
            if probe[y, x] > 0:
                hit += 1
    if total == 0:
        return 0.0
    return hit / float(total)


def _clamp_square_by_short_edge(pts: np.ndarray, max_diff_px: float = 15.0) -> np.ndarray:
    """
    If edge-length spread exceeds max_diff_px, rebuild as a square using the shortest
    side length, anchored at the touching corner (top-left in ordered points).
    """
    p = order_points(pts.astype(np.float32))
    lengths = [float(np.linalg.norm(p[(i + 1) % 4] - p[i])) for i in range(4)]
    min_len = min(lengths)
    max_len = max(lengths)
    if (max_len - min_len) <= max_diff_px:
        return p

    # Anchor at p0 and clamp both adjacent sides to shortest length.
    u = _normalized(p[1] - p[0])  # top edge direction
    v = _normalized(p[3] - p[0])  # left edge direction
    if float(np.linalg.norm(u)) < 1e-6 or float(np.linalg.norm(v)) < 1e-6:
        return p

    p0 = p[0]
    side = float(min_len)
    p1 = p0 + u * side
    p3 = p0 + v * side
    p2 = p1 + (p3 - p0)
    return order_points(np.array([p0, p1, p2, p3], dtype=np.float32))


def _reconstruct_square_from_orthogonal_lines(
    edge_map: np.ndarray,
    img_shape: tuple[int, int, int],
    min_area_ratio: float,
) -> np.ndarray | None:
    """
    Fallback:
    If contour closure is still difficult, find two near-orthogonal touching lines
    and reconstruct a square from that corner.
    """
    h, w = img_shape[:2]
    img_area = float(h * w)

    lines = cv2.HoughLinesP(
        edge_map,
        rho=1,
        theta=np.pi / 180.0,
        threshold=40,
        minLineLength=max(20, int(min(h, w) * 0.08)),
        maxLineGap=12,
    )
    if lines is None or len(lines) < 2:
        return None

    segs: list[tuple[np.ndarray, np.ndarray, float]] = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        p1 = np.array([float(x1), float(y1)], dtype=np.float32)
        p2 = np.array([float(x2), float(y2)], dtype=np.float32)
        ln = float(np.linalg.norm(p2 - p1))
        if ln < 8.0:
            continue
        segs.append((p1, p2, ln))
    if len(segs) < 2:
        return None

    best_score = -1.0
    best_pts: np.ndarray | None = None
    corner_dist_th = max(6.0, 0.015 * max(h, w))

    for i in range(len(segs)):
        a1, a2, la = segs[i]
        for j in range(i + 1, len(segs)):
            b1, b2, lb = segs[j]
            endpoints_a = [a1, a2]
            endpoints_b = [b1, b2]
            for ea in endpoints_a:
                for eb in endpoints_b:
                    if float(np.linalg.norm(ea - eb)) > corner_dist_th:
                        continue
                    corner = (ea + eb) * 0.5
                    va = a2 - a1 if np.allclose(ea, a1) else a1 - a2
                    vb = b2 - b1 if np.allclose(eb, b1) else b1 - b2
                    ua = _normalized(va)
                    ub = _normalized(vb)
                    if float(np.linalg.norm(ua)) < 1e-6 or float(np.linalg.norm(ub)) < 1e-6:
                        continue
                    cosang = float(abs(np.dot(ua, ub)))
                    if cosang > 0.25:  # ~75-105 deg
                        continue

                    # Enforce side-length consistency:
                    # if two touching lines differ by more than 15 px, trim the longer one
                    # to the shorter length from the touching corner.
                    short_len = min(la, lb)
                    long_len = max(la, lb)
                    if (long_len - short_len) > 15.0:
                        side_a = short_len
                        side_b = short_len
                    else:
                        side_a = la
                        side_b = lb

                    p0 = corner
                    p1 = p0 + ua * side_a
                    p3 = p0 + ub * side_b
                    # Final square uses a single side length; if the two are close, use their mean,
                    # otherwise both were already clamped to short_len above.
                    side = 0.5 * (side_a + side_b)
                    p1 = p0 + ua * side
                    p3 = p0 + ub * side
                    p2 = p1 + ub * side
                    pts = np.array([p0, p1, p2, p3], dtype=np.float32)

                    x_min = float(np.min(pts[:, 0]))
                    y_min = float(np.min(pts[:, 1]))
                    x_max = float(np.max(pts[:, 0]))
                    y_max = float(np.max(pts[:, 1]))
                    rect_area = max(1.0, (x_max - x_min) * (y_max - y_min))
                    if rect_area < min_area_ratio * img_area:
                        continue
                    # Avoid selecting the image border as a fake square.
                    border_margin = 3.0
                    if (
                        x_min <= border_margin
                        or y_min <= border_margin
                        or x_max >= (w - 1 - border_margin)
                        or y_max >= (h - 1 - border_margin)
                    ):
                        continue
                    if x_max < 0 or y_max < 0 or x_min >= w or y_min >= h:
                        continue

                    edge_score = _edge_support_score(edge_map, pts)
                    if edge_score < 0.25:
                        continue
                    center = np.mean(pts, axis=0)
                    dist_norm = np.hypot(center[0] - w / 2.0, center[1] - h / 2.0) / max(h, w)
                    score = rect_area * (1.0 - dist_norm) * edge_score
                    if score > best_score:
                        best_score = score
                        best_pts = pts

    if best_pts is None:
        return None
    return _clamp_square_by_short_edge(order_points(best_pts), max_diff_px=15.0)


def detect_k_frame_box(img: np.ndarray) -> np.ndarray | None:
    """
    Detect K frame box (square frame) using shape-based criteria.

    Uses grayscale + blur + Canny edges to find external contours, then
    selects the best square candidate based on:
    - 4 vertices (quadrilateral)
    - convex
    - sufficiently large area
    - near-square aspect ratio
    - closer to the image center

    Returns:
        np.ndarray | None: 4 corner points [top-left, top-right, bottom-right, bottom-left], or None
    """
    if img is None or img.size == 0:
        return None

    h, w = img.shape[:2]
    img_area = float(h * w)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Dotted halftone suppression: preserve strong frame edges while reducing micro-dot texture.
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Extra channels to handle color overlaps on black lines.
    # - LAB L channel is robust to chroma changes
    # - HSV V channel captures intensity changes even when hue/sat vary
    lab_l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
    lab_l = cv2.medianBlur(lab_l, 3)
    lab_l = cv2.bilateralFilter(lab_l, 7, 40, 40)
    lab_l = cv2.GaussianBlur(lab_l, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_v = hsv[:, :, 2]
    hsv_v = cv2.medianBlur(hsv_v, 3)
    hsv_v = cv2.bilateralFilter(hsv_v, 7, 40, 40)
    hsv_v = cv2.GaussianBlur(hsv_v, (5, 5), 0)

    # Black-ish line prior mask (low saturation + low value) helps when colored patches overlap.
    # Keep this permissive; it is used as an extra edge source, not a hard segmentation.
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0], dtype=np.uint8), np.array([180, 95, 115], dtype=np.uint8))
    black_mask = cv2.morphologyEx(
        black_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    # Notes on robustness:
    # - Small gaps in contours are common for thin frames; we aggressively apply morphological closing
    #   to connect edges before contour extraction.
    # - Thin square frames can have very small contour area vs bounding rect area; we score using the
    #   min-area rectangle and do NOT require a high fill ratio.
    strategies = [
        {
            "name": "canny_cfg3",
            "mode": "canny",
            "canny1": 30,
            "canny2": 120,
            "min_area_ratio": 0.01,
            "aspect_min": 0.70,
            "aspect_max": 1.30,
            "close_ksize": 5,
            "close_iter": 2,
        },
        {
            "name": "canny_soft",
            "mode": "canny",
            "canny1": 20,
            "canny2": 80,
            "min_area_ratio": 0.008,
            "aspect_min": 0.65,
            "aspect_max": 1.35,
            "close_ksize": 7,
            "close_iter": 2,
        },
        {
            "name": "adaptive",
            "mode": "adaptive",
            "min_area_ratio": 0.008,
            "aspect_min": 0.65,
            "aspect_max": 1.35,
            "close_ksize": 7,
            "close_iter": 2,
        },
    ]

    kernel3 = np.ones((3, 3), np.uint8)

    for cfg in strategies:
        close_k = max(3, int(cfg.get("close_ksize", 5)))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))

        if cfg["mode"] == "canny":
            # Combine edges from multiple luminance-like channels so colored overlaps
            # don't erase the black-frame boundary.
            e0 = cv2.Canny(gray, cfg["canny1"], cfg["canny2"])
            e1 = cv2.Canny(lab_l, cfg["canny1"], cfg["canny2"])
            e2 = cv2.Canny(hsv_v, cfg["canny1"], cfg["canny2"])
            e3 = cv2.Canny(black_mask, 20, 80)
            edges = cv2.bitwise_or(cv2.bitwise_or(e0, e1), cv2.bitwise_or(e2, e3))
            # Thicken and close tiny gaps so open contours become closed.
            edges = cv2.dilate(edges, kernel3, iterations=2)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=int(cfg.get("close_iter", 2)))
            # One more close pass helps when edges are very thin / broken.
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel3, iterations=1)
            bin_img = edges
        else:
            bin_img = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, close_kernel, iterations=int(cfg.get("close_iter", 2)))
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel3, iterations=1)

        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        candidates: list[tuple[float, np.ndarray]] = []

        for c in contours:
            peri = cv2.arcLength(c, True)
            if peri < 1e-3:
                continue

            # Use minAreaRect so we can still get a stable 4-corner rectangle even when
            # the contour is imperfect (e.g., thin frame, small gaps).
            rect = cv2.minAreaRect(c)
            (rcx, rcy), (rw, rh), _angle = rect
            rw = float(rw)
            rh = float(rh)
            if rw < 1.0 or rh < 1.0:
                continue

            rect_area = rw * rh
            if rect_area < cfg["min_area_ratio"] * img_area:
                continue
            box = cv2.boxPoints(rect).astype("float32")
            x_min = float(np.min(box[:, 0]))
            y_min = float(np.min(box[:, 1]))
            x_max = float(np.max(box[:, 0]))
            y_max = float(np.max(box[:, 1]))
            border_margin = 3.0
            if (
                x_min <= border_margin
                or y_min <= border_margin
                or x_max >= (w - 1 - border_margin)
                or y_max >= (h - 1 - border_margin)
            ):
                continue

            aspect = max(rw, rh) / min(rw, rh)
            # Hard constraint: it must still be reasonably square-ish.
            if not (cfg["aspect_min"] <= aspect <= cfg["aspect_max"]):
                continue

            # Thin frames have low fill ratio; allow wide range but reject extreme noise.
            cnt_area = float(cv2.contourArea(c))
            fill_ratio = cnt_area / rect_area if rect_area > 1e-6 else 0.0
            if fill_ratio < 0.01 or fill_ratio > 1.20:
                continue

            dist_norm = np.hypot(rcx - w / 2.0, rcy - h / 2.0) / max(h, w)

            # Prefer perfect squares strongly (aspect ~= 1.0).
            square_bonus = 1.0 / (1.0 + 8.0 * abs(aspect - 1.0))

            # Prefer shapes whose perimeter roughly matches a rectangle's perimeter.
            rect_peri = 2.0 * (rw + rh)
            peri_match = 1.0 / (1.0 + abs(peri - rect_peri) / max(rect_peri, 1e-6))

            score = rect_area * (1.0 - dist_norm) * square_bonus * peri_match

            candidates.append((score, box))

        if candidates:
            _, best_pts = max(candidates, key=lambda x: x[0])
            rect = _clamp_square_by_short_edge(order_points(best_pts), max_diff_px=15.0)
            return rect

        # Fallback: recover square from two touching near-orthogonal lines.
        recovered = _reconstruct_square_from_orthogonal_lines(bin_img, img.shape, float(cfg["min_area_ratio"]))
        if recovered is not None:
            return _clamp_square_by_short_edge(recovered, max_diff_px=15.0)

    return None


def draw_k_frame_overlay(
    img: np.ndarray,
    corners: np.ndarray | None,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Return a copy of BGR image with K-frame polygon drawn (or unchanged if corners is None)."""
    out = img.copy()
    if corners is None:
        return out
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out
