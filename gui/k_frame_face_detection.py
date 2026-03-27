#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-frame warped image: color-agnostic face (colored patch) detection."""

from __future__ import annotations

import cv2
import numpy as np

from color_registration_analysis import order_points


def detect_color_faces_in_warp(warped_bgr: np.ndarray) -> list[np.ndarray]:
    """
    색상에 상관없이(컬러 분류 없이) warp 내부의 색상 면들을 찾습니다.
    정사각형 가정 없이 직사각형/비정형 후보도 폭넓게 허용합니다.
    Return: 각 면의 4-corner box(4x2) 리스트.
    """
    if warped_bgr is None or warped_bgr.size == 0:
        return []
    h, w = warped_bgr.shape[:2]
    img_area = float(h * w)

    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    def _contour_touches_border(contour: np.ndarray, margin: float = 2.0) -> bool:
        x_min = float(np.min(contour[:, 0, 0]))
        y_min = float(np.min(contour[:, 0, 1]))
        x_max = float(np.max(contour[:, 0, 0]))
        y_max = float(np.max(contour[:, 0, 1]))
        return (
            x_min <= margin
            or y_min <= margin
            or x_max >= (w - 1 - margin)
            or y_max >= (h - 1 - margin)
        )

    def _refine_contour_to_core(contour: np.ndarray, border_tolerant: bool) -> tuple[np.ndarray | None, float]:
        """
        후보 면 내부의 연한 번짐 영역을 걸러내고, 진한 코어 영역만 남깁니다.
        """
        raw_area = float(cv2.contourArea(contour))
        if raw_area < 1.0:
            return None, 0.0

        c_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(c_mask, [contour], -1, 255, thickness=-1)
        ys, xs = np.where(c_mask > 0)
        if len(xs) < 16:
            return None, 0.0

        s_vals = s_ch[ys, xs].astype(np.float32)
        v_vals = v_ch[ys, xs].astype(np.float32)

        # Adaptive thresholds to reject washed-out smear pixels.
        s_thr = max(22.0, float(np.percentile(s_vals, 28)))
        v_thr = max(20.0, float(np.percentile(v_vals, 15)))

        core = np.zeros((h, w), dtype=np.uint8)
        core[(c_mask > 0) & (s_ch >= s_thr) & (v_ch >= v_thr)] = 255
        core = cv2.morphologyEx(
            core,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        core = cv2.morphologyEx(
            core,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

        core_contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not core_contours:
            return None, 0.0
        best_core = max(core_contours, key=cv2.contourArea)
        core_area = float(cv2.contourArea(best_core))
        if core_area < 1.0:
            return None, 0.0

        # Too small core compared to raw candidate => likely smear/noise.
        # For border-touching partial faces, use more tolerant ratio.
        min_core_ratio = 0.10 if border_tolerant else 0.18
        if core_area / max(raw_area, 1e-6) < min_core_ratio:
            return None, 0.0
        return best_core, core_area

    def _collect_contours_with_overlapped_quadrants(bin_mask: np.ndarray) -> list[np.ndarray]:
        """
        Collect contours from full mask + overlapped quadrant ROIs.
        This helps split connected smear regions into local candidates.
        """
        all_contours: list[np.ndarray] = []
        cnts_full, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(cnts_full if cnts_full is not None else [])

        ov = 0.15
        regions = [
            (0, 0, int(w * (0.5 + ov)), int(h * (0.5 + ov))),  # TL
            (int(w * (0.5 - ov)), 0, w, int(h * (0.5 + ov))),  # TR
            (0, int(h * (0.5 - ov)), int(w * (0.5 + ov)), h),  # BL
            (int(w * (0.5 - ov)), int(h * (0.5 - ov)), w, h),  # BR
        ]
        for x1, y1, x2, y2 in regions:
            if x2 <= x1 or y2 <= y1:
                continue
            roi = bin_mask[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts if cnts is not None else []:
                c2 = c.copy()
                c2[:, 0, 0] += x1
                c2[:, 0, 1] += y1
                all_contours.append(c2)
        return all_contours

    # Two-pass thresholds: strict -> relax
    pass_cfgs = [
        {"S": 60, "V": 50, "min_area_ratio": 0.0015, "max_area_ratio": 0.35, "aspect_min": 0.35, "aspect_max": 3.00},
        {"S": 30, "V": 30, "min_area_ratio": 0.0006, "max_area_ratio": 0.50, "aspect_min": 0.25, "aspect_max": 4.50},
        # Border-friendly pass for partially cut faces
        {"S": 22, "V": 22, "min_area_ratio": 0.00025, "max_area_ratio": 0.60, "aspect_min": 0.12, "aspect_max": 8.00},
    ]

    chosen: list[np.ndarray] = []
    for cfg in pass_cfgs:
        mask = cv2.inRange(
            hsv,
            np.array([0, cfg["S"], cfg["V"]], dtype=np.uint8),
            np.array([180, 255, 255], dtype=np.uint8),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )

        contours = _collect_contours_with_overlapped_quadrants(mask)
        if not contours:
            continue

        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for c in contours:
            is_border = _contour_touches_border(c, margin=2.0)
            refined_c, area = _refine_contour_to_core(c, border_tolerant=is_border)
            if refined_c is None:
                continue
            # Partial faces on border are allowed to be smaller.
            min_area_ratio = cfg["min_area_ratio"] * (0.35 if is_border else 1.0)
            if area < min_area_ratio * img_area or area > cfg["max_area_ratio"] * img_area:
                continue
            rect = cv2.minAreaRect(refined_c)
            (cx, cy), (rw, rh), _ang = rect
            rw = float(rw)
            rh = float(rh)
            if rw < 5.0 or rh < 5.0:
                continue
            aspect = max(rw, rh) / max(1e-6, min(rw, rh))
            if aspect < cfg["aspect_min"] or aspect > cfg["aspect_max"]:
                continue

            box = cv2.boxPoints(rect).astype("float32")
            dist_norm = np.hypot(cx - w / 2.0, cy - h / 2.0) / max(h, w)
            score = area * (1.0 - 0.5 * dist_norm)
            candidates.append((score, box, np.array([cx, cy], dtype=np.float32)))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = []
        chosen_centers: list[np.ndarray] = []
        center_dist_th = 0.18 * max(h, w)

        for _score, box, center in candidates:
            if len(chosen) >= 6:
                break
            if any(float(np.linalg.norm(center - c)) < center_dist_th for c in chosen_centers):
                continue
            chosen.append(box)
            chosen_centers.append(center)
            if len(chosen) >= 10:
                break

        if len(chosen) >= 6:
            break

    # If we found too few blocks, do one more relaxed selection without strict distance gating.
    if len(chosen) < 2:
        all_boxes: list[np.ndarray] = []
        for cfg in pass_cfgs:
            mask = cv2.inRange(
                hsv,
                np.array([0, cfg["S"], cfg["V"]], dtype=np.uint8),
                np.array([180, 255, 255], dtype=np.uint8),
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            contours = _collect_contours_with_overlapped_quadrants(mask)
            for c in contours:
                is_border = _contour_touches_border(c, margin=2.0)
                refined_c, area = _refine_contour_to_core(c, border_tolerant=is_border)
                if refined_c is None:
                    continue
                min_area_ratio = cfg["min_area_ratio"] * (0.35 if is_border else 1.0)
                if area < min_area_ratio * img_area or area > cfg["max_area_ratio"] * img_area:
                    continue
                rect = cv2.minAreaRect(refined_c)
                (cx, cy), (rw, rh), _ang = rect
                rw = float(rw)
                rh = float(rh)
                if rw < 5.0 or rh < 5.0:
                    continue
                aspect = max(rw, rh) / max(1e-6, min(rw, rh))
                if aspect < cfg["aspect_min"] or aspect > cfg["aspect_max"]:
                    continue
                box = cv2.boxPoints(rect).astype("float32")
                dist_norm = np.hypot(cx - w / 2.0, cy - h / 2.0) / max(h, w)
                score = area * (1.0 - 0.5 * dist_norm)
                all_boxes.append((score, box, np.array([cx, cy], dtype=np.float32)))  # type: ignore[arg-type]

        if all_boxes:
            # type: ignore[arg-type] above keeps typing simple; handle runtime below.
            all_boxes_sorted = sorted(all_boxes, key=lambda x: x[0], reverse=True)  # type: ignore[index]
            picked = []
            for _score, box, _center in all_boxes_sorted:  # type: ignore[misc]
                picked.append(box)
                if len(picked) >= 10:
                    break
            if len(picked) >= 1:
                chosen = picked[:10]

    if len(chosen) == 0:
        return []

    # Outlier removal + quadrant-aware soft assignment (allows cross-quadrant overlap).
    def _touches_border(box: np.ndarray, img_w: int, img_h: int, margin: float = 3.0) -> bool:
        x_min = float(np.min(box[:, 0]))
        y_min = float(np.min(box[:, 1]))
        x_max = float(np.max(box[:, 0]))
        y_max = float(np.max(box[:, 1]))
        return (
            x_min <= margin
            or y_min <= margin
            or x_max >= (img_w - 1 - margin)
            or y_max >= (img_h - 1 - margin)
        )

    def _box_wh(box: np.ndarray) -> tuple[float, float]:
        x_min = float(np.min(box[:, 0]))
        y_min = float(np.min(box[:, 1]))
        x_max = float(np.max(box[:, 0]))
        y_max = float(np.max(box[:, 1]))
        return max(1.0, x_max - x_min), max(1.0, y_max - y_min)

    def _bbox(box: np.ndarray) -> tuple[float, float, float, float]:
        x_min = float(np.min(box[:, 0]))
        y_min = float(np.min(box[:, 1]))
        x_max = float(np.max(box[:, 0]))
        y_max = float(np.max(box[:, 1]))
        return x_min, y_min, x_max, y_max

    def _inter_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return float((x2 - x1) * (y2 - y1))

    def _iou_bbox(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        inter = _inter_area(a, b)
        if inter <= 0.0:
            return 0.0
        area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / max(1e-6, area_a + area_b - inter)

    def _enhanced_candidates_when_few(existing_boxes: list[np.ndarray]) -> list[tuple[float, np.ndarray, bool, np.ndarray]]:
        """
        Fallback path used only when <4 faces are found:
        strengthen separation for background-similar colors using local chroma contrast.
        """
        lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ab = lab[:, :, 1:3]
        # Local background estimate on chroma channels
        bg_ab = cv2.GaussianBlur(ab, (0, 0), sigmaX=19, sigmaY=19)
        diff = np.linalg.norm(ab - bg_ab, axis=2).astype(np.float32)
        sat = s_ch.astype(np.float32)

        diff_n = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        sat_n = cv2.normalize(sat, None, 0, 255, cv2.NORM_MINMAX)
        score = (0.7 * diff_n + 0.3 * sat_n).astype(np.uint8)

        # Global threshold + per-quadrant relaxed threshold to rescue partial faces.
        thr_g = float(np.percentile(score, 78))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[score >= thr_g] = 255
        qh, qw = h // 2, w // 2
        quads = [(0, 0, qw, qh), (qw, 0, w, qh), (0, qh, qw, h), (qw, qh, w, h)]
        for x1, y1, x2, y2 in quads:
            roi = score[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            thr_q = float(np.percentile(roi, 70))
            sub = np.zeros_like(roi, dtype=np.uint8)
            sub[roi >= thr_q] = 255
            mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], sub)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return []

        exist_bbs = [_bbox(b) for b in existing_boxes]
        out: list[tuple[float, np.ndarray, bool, np.ndarray]] = []
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < 0.00035 * img_area or area > 0.60 * img_area:
                continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), _a = rect
            rw = float(rw)
            rh = float(rh)
            if rw < 4.0 or rh < 4.0:
                continue
            aspect = max(rw, rh) / max(1e-6, min(rw, rh))
            if aspect > 9.0:
                continue
            box = cv2.boxPoints(rect).astype(np.float32)
            bb = _bbox(box)
            if any(_iou_bbox(bb, eb) > 0.65 for eb in exist_bbs):
                continue
            border_touch = _touches_border(box, w, h)
            center = np.array([float(np.mean(box[:, 0])), float(np.mean(box[:, 1]))], dtype=np.float32)
            out.append((area, box, border_touch, center))

        out.sort(key=lambda x: x[0], reverse=True)
        return out

    candidates_with_area: list[tuple[float, np.ndarray, bool, np.ndarray]] = []
    for b in chosen:
        area = float(cv2.contourArea(b.astype(np.float32).reshape(-1, 1, 2)))
        if area > 1.0:
            center = np.array([float(np.mean(b[:, 0])), float(np.mean(b[:, 1]))], dtype=np.float32)
            candidates_with_area.append((area, b, _touches_border(b, w, h), center))
    if not candidates_with_area:
        return []
    candidates_with_area.sort(key=lambda t: t[0], reverse=True)

    # 2) robust outlier threshold from largest group
    head_areas = [a for a, _, _, _ in candidates_with_area[: min(6, len(candidates_with_area))]]
    ref = float(np.median(head_areas)) if head_areas else float(candidates_with_area[0][0])
    rel_min = 0.25 * ref
    abs_min = 0.0025 * img_area
    min_keep_area = max(rel_min, abs_min)

    filtered = [(a, b, t, c) for a, b, t, c in candidates_with_area if a >= min_keep_area]

    # Partial faces near crop border can be much smaller.
    # If we have fewer than 4 faces, allow border-touching candidates with lower threshold.
    if len(filtered) < 4:
        border_min = max(0.15 * ref, 0.0012 * img_area)
        rescued = [
            (a, b, t, c)
            for a, b, t, c in candidates_with_area
            if t and a >= border_min and not any(np.array_equal(b, fb) for _, fb, _, _ in filtered)
        ]
        filtered.extend(rescued)
        filtered.sort(key=lambda x: x[0], reverse=True)

    # Thin border-face rule:
    # Preserve thin elongated candidates touching border (partially visible faces).
    thin_min = max(0.06 * ref, 0.0006 * img_area)
    thin_extra: list[tuple[float, np.ndarray, bool, np.ndarray]] = []
    for a, b, t, c in candidates_with_area:
        if not t or a < thin_min:
            continue
        bw, bh = _box_wh(b)
        aspect = max(bw, bh) / max(1e-6, min(bw, bh))
        if aspect >= 2.0 and not any(np.array_equal(b, fb) for _, fb, _, _ in filtered):
            thin_extra.append((a, b, t, c))
    if thin_extra:
        filtered.extend(thin_extra)
        filtered.sort(key=lambda x: x[0], reverse=True)

    if not filtered:
        # fallback: at least keep the biggest one
        filtered = [candidates_with_area[0]]

    # 3) soft quadrant assignment
    overlap = 0.14  # allow some cross-quadrant coverage
    q_rects = {
        "TL": (0.0, 0.0, w * (0.5 + overlap), h * (0.5 + overlap)),
        "TR": (w * (0.5 - overlap), 0.0, float(w), h * (0.5 + overlap)),
        "BR": (w * (0.5 - overlap), h * (0.5 - overlap), float(w), float(h)),
        "BL": (0.0, h * (0.5 - overlap), w * (0.5 + overlap), float(h)),
    }
    q_centers = {
        "TL": np.array([w * 0.25, h * 0.25], dtype=np.float32),
        "TR": np.array([w * 0.75, h * 0.25], dtype=np.float32),
        "BR": np.array([w * 0.75, h * 0.75], dtype=np.float32),
        "BL": np.array([w * 0.25, h * 0.75], dtype=np.float32),
    }

    selected: dict[str, tuple[float, np.ndarray, bool, np.ndarray]] = {}
    used: set[int] = set()
    for qn in ["TL", "TR", "BR", "BL"]:
        best_idx = -1
        best_score = -1.0
        qr = q_rects[qn]
        qc = q_centers[qn]
        for idx, (a, b, t, c) in enumerate(filtered):
            if idx in used:
                continue
            bb = _bbox(b)
            inter = _inter_area(bb, qr)
            bb_area = max(1.0, (bb[2] - bb[0]) * (bb[3] - bb[1]))
            overlap_ratio = inter / bb_area
            spill = 1.0 - overlap_ratio
            dist_norm = float(np.linalg.norm(c - qc)) / max(h, w)
            score = a * (0.35 + overlap_ratio) * (1.0 - 0.4 * spill) * (1.0 - 0.35 * dist_norm)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0:
            selected[qn] = filtered[best_idx]
            used.add(best_idx)

    chosen_boxes: list[tuple[float, np.ndarray, bool, np.ndarray]] = []
    for qn in ["TL", "TR", "BR", "BL"]:
        if qn in selected:
            chosen_boxes.append(selected[qn])

    # Fill missing slots from remaining largest candidates.
    if len(chosen_boxes) < 4:
        for idx, item in enumerate(filtered):
            if idx in used:
                continue
            chosen_boxes.append(item)
            if len(chosen_boxes) >= 4:
                break

    # Overlapping-rectangle rule:
    # for thin border faces, add one overlapped inward-shifted rectangle.
    overlap_extra: list[tuple[float, np.ndarray, bool, np.ndarray]] = []
    for a, b, t, c in chosen_boxes:
        if not t:
            continue
        bw, bh = _box_wh(b)
        aspect = max(bw, bh) / max(1e-6, min(bw, bh))
        if aspect < 2.0:
            continue

        x_min, y_min, x_max, y_max = _bbox(b)
        shift_x = 0.18 * bw if (x_max >= w - 2 or x_min <= 2) else 0.0
        shift_y = 0.18 * bh if (y_max >= h - 2 or y_min <= 2) else 0.0

        if x_max >= w - 2:
            nx1, nx2 = x_min - shift_x, x_max - shift_x
        elif x_min <= 2:
            nx1, nx2 = x_min + shift_x, x_max + shift_x
        else:
            nx1, nx2 = x_min, x_max
        if y_max >= h - 2:
            ny1, ny2 = y_min - shift_y, y_max - shift_y
        elif y_min <= 2:
            ny1, ny2 = y_min + shift_y, y_max + shift_y
        else:
            ny1, ny2 = y_min, y_max

        nx1 = max(0.0, min(float(w - 1), nx1))
        nx2 = max(0.0, min(float(w - 1), nx2))
        ny1 = max(0.0, min(float(h - 1), ny1))
        ny2 = max(0.0, min(float(h - 1), ny2))
        if nx2 <= nx1 + 2 or ny2 <= ny1 + 2:
            continue
        b2 = np.array([[nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]], dtype=np.float32)
        c2 = np.array([float(np.mean(b2[:, 0])), float(np.mean(b2[:, 1]))], dtype=np.float32)
        a2 = float(max(1.0, (nx2 - nx1) * (ny2 - ny1)))
        overlap_extra.append((a2, b2, True, c2))
    if overlap_extra:
        chosen_boxes.extend(overlap_extra)

    # Remove over-merged giant boxes (typical smear-connection artifact).
    compact: list[tuple[float, np.ndarray, bool, np.ndarray]] = []
    for a, b, t, c in chosen_boxes:
        x_min, y_min, x_max, y_max = _bbox(b)
        bw = x_max - x_min
        bh = y_max - y_min
        if bw > 0.72 * w and bh > 0.72 * h:
            continue
        compact.append((a, b, t, c))
    if len(compact) >= 3:
        chosen_boxes = compact

    # Keep 4 best by area for stable output.
    chosen_boxes.sort(key=lambda x: x[0], reverse=True)
    top4 = chosen_boxes[:4]

    # Enhanced fallback only when fewer than 4 are found.
    if len(top4) < 4:
        existing = [b for _a, b, _t, _c in top4]
        extra = _enhanced_candidates_when_few(existing)
        for item in extra:
            top4.append(item)
            if len(top4) >= 4:
                break
        top4.sort(key=lambda x: x[0], reverse=True)
        top4 = top4[:4]
    return [order_points(b) for _a, b, _t, _c in top4]


# Named quadrants in warped K-frame (y-down image coordinates).
FACE_POSITION_KEYS: tuple[str, ...] = (
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
)


def pivot_point_from_face_quad(face_quad: np.ndarray, position: str) -> tuple[float, float]:
    """
    Registration pivot on the face quadrilateral (not the centroid).
    `face_quad` is re-ordered via order_points to [TL, TR, BR, BL] (indices 0..3).

    Mapping (inner corner toward K-frame center):
      upper_left   -> bottom-right vertex (index 2)
      upper_right  -> bottom-left vertex (index 3)
      lower_left   -> top-right vertex (index 1)
      lower_right  -> top-left vertex (index 0)
    """
    ordered = order_points(face_quad.astype(np.float32))
    idx = {
        "upper_left": 2,
        "upper_right": 3,
        "lower_left": 1,
        "lower_right": 0,
    }[position]
    p = ordered[idx]
    return float(p[0]), float(p[1])


def assign_face_boxes_to_positions(
    faces: list[np.ndarray], w: int, h: int
) -> dict[str, np.ndarray | None]:
    """
    Assign each face quad to a quadrant by nearest quadrant center; if multiple
    faces map to the same quadrant, keep the largest-area face.
    """
    if w <= 0 or h <= 0:
        return {k: None for k in FACE_POSITION_KEYS}

    centers = {
        "upper_left": np.array([w * 0.25, h * 0.25], dtype=np.float64),
        "upper_right": np.array([w * 0.75, h * 0.25], dtype=np.float64),
        "lower_left": np.array([w * 0.25, h * 0.75], dtype=np.float64),
        "lower_right": np.array([w * 0.75, h * 0.75], dtype=np.float64),
    }
    buckets: dict[str, list[tuple[float, np.ndarray]]] = {k: [] for k in FACE_POSITION_KEYS}

    for face in faces:
        cx = float(np.mean(face[:, 0]))
        cy = float(np.mean(face[:, 1]))
        c = np.array([cx, cy], dtype=np.float64)
        quad = face.astype(np.float32).reshape(-1, 1, 2)
        area = float(cv2.contourArea(quad))
        nearest = min(FACE_POSITION_KEYS, key=lambda p: float(np.linalg.norm(c - centers[p])))
        buckets[nearest].append((area, face))

    out: dict[str, np.ndarray | None] = {}
    for p in FACE_POSITION_KEYS:
        if not buckets[p]:
            out[p] = None
        else:
            best = max(buckets[p], key=lambda x: x[0])
            out[p] = best[1]
    return out
