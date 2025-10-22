# processor.py

import numpy as np
import cv2 as cv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from contour import (
    threshold_binary,
    morph_open_close,
    find_valid_contours,
    draw_rectangle, draw_min_area_rect, draw_convex_hull, draw_ellipse_fit
)


@dataclass
class ProcParams:
    gray_threshold: int = 50
    min_area: float = 200.0
    max_area: float = 30000.0
    morph_ellipse_size: int = 23    # odd
    fit_shape: str = "Rectangle"    # Rectangle | Min Area Rectangle | Convex Hull | Ellipse
    enable_smoothing: bool = True
    enable_bg_median: bool = False  # safer than KNN for embryos
    # smoothing specifics
    smooth_close: int = 7
    smooth_open: int  = 3
    smooth_blur: int  = 7
    smooth_sigma: float = 1.4
    ellipse_alpha: float = 0.25     # 0..0.4 nice; 0 disables


class Processor:
    def __init__(self):
        self._bg_stack = []  # rolling median background (grayscale)
        self._bg_len = 25

    def reset_bg_median(self):
        self._bg_stack = []

    def _apply_bg_median(self, gray: np.ndarray) -> np.ndarray:
        """Subtract rolling median background (robust for static embryos)."""
        if len(self._bg_stack) < self._bg_len:
            self._bg_stack.append(gray.copy())
            med = np.median(np.stack(self._bg_stack, axis=0), axis=0).astype(np.uint8)
        else:
            self._bg_stack.pop(0); self._bg_stack.append(gray.copy())
            med = np.median(np.stack(self._bg_stack, axis=0), axis=0).astype(np.uint8)
        diff = cv.absdiff(gray, med)
        return cv.normalize(diff, None, 0, 255, cv.NORM_MINMAX)

    def _smooth_mask(self, mask: np.ndarray, cnt: np.ndarray, p: ProcParams) -> np.ndarray:
        m = mask.copy()
        kclose = cv.getStructuringElement(cv.MORPH_ELLIPSE, (p.smooth_close, p.smooth_close))
        kopen  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (p.smooth_open,  p.smooth_open))
        m = cv.morphologyEx(m, cv.MORPH_CLOSE, kclose)
        m = cv.morphologyEx(m, cv.MORPH_OPEN,  kopen)
        m = cv.GaussianBlur(m, (p.smooth_blur, p.smooth_blur), p.smooth_sigma)
        _, m = cv.threshold(m, 128, 255, cv.THRESH_BINARY)

        # optional convex hull
        cnts,_ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv.contourArea)
            hull = cv.convexHull(c)
            hullmask = np.zeros_like(m); cv.drawContours(hullmask, [hull], -1, 255, -1)
            m = hullmask

        if p.ellipse_alpha > 0 and cnt is not None and len(cnt) >= 5:
            emask = np.zeros_like(m)
            ellipse = cv.fitEllipse(cnt)
            cv.ellipse(emask, ellipse, 255, -1)
            mf = m.astype(np.float32)/255.0; ef = emask.astype(np.float32)/255.0
            blended = (1.0 - p.ellipse_alpha)*mf + p.ellipse_alpha*ef
            m = (blended*255.0).astype(np.uint8)
            _, m = cv.threshold(m, 128, 255, cv.THRESH_BINARY)
        return m

    def process_fullframe(self, frame_bgr: np.ndarray, p: ProcParams) -> Dict:
        gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)

        src_for_threshold = gray
        if p.enable_bg_median:
            src_for_threshold = self._apply_bg_median(gray)

        binary = threshold_binary(src_for_threshold, p.gray_threshold)
        morphed = morph_open_close(binary, p.morph_ellipse_size)

        contours = find_valid_contours(
            morphed,
            min_area=p.min_area,
            max_area=p.max_area
        )

        # optional smoothing (per contour -> mask refine)
        if p.enable_smoothing and contours:
            refined = []
            for c in contours:
                # mask from contour
                m = np.zeros_like(morphed); cv.drawContours(m, [c], -1, 255, -1)
                m = self._smooth_mask(m, c, p)
                cnts,_ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                if cnts:
                    refined.append(max(cnts, key=cv.contourArea))
            if refined:
                contours = refined

        return {
            "binary": binary,
            "morphed": morphed,
            "contours": contours
        }

    def draw_fits(self, img_bgr: np.ndarray, contours: List[np.ndarray], fit_shape: str) -> np.ndarray:
        if not contours: return img_bgr
        if fit_shape == "Rectangle":
            for c in contours: draw_rectangle(img_bgr, c)
        elif fit_shape == "Min Area Rectangle":
            for c in contours: draw_min_area_rect(img_bgr, c)
        elif fit_shape == "Convex Hull":
            for c in contours: draw_convex_hull(img_bgr, c)
        elif fit_shape == "Ellipse":
            for c in contours: draw_ellipse_fit(img_bgr, c)
        else:
            for c in contours: draw_rectangle(img_bgr, c)
        return img_bgr
