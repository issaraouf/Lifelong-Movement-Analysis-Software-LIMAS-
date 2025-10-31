# contour.py — segmentation par seuillage (avec option d'inversion) + fusion éventuelle avec un masque de BG
# Conserve les filtres de forme (ratio d'aspect, circularité) et les fonctions de dessin.

import cv2 as cv
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional


# ----------------- helpers: drawing funcs -----------------
def _draw_rectangle(img, cnt):
    x, y, w, h = cv.boundingRect(cnt)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)


def _draw_min_area_rect(img, cnt):
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int32(box)
    cv.polylines(img, [box], True, (0, 255, 255), 2)


def _draw_convex_hull(img, cnt):
    hull = cv.convexHull(cnt)
    cv.polylines(img, [hull], True, (0, 255, 255), 2)


def _draw_ellipse(img, cnt):
    if len(cnt) >= 5:
        ellipse = cv.fitEllipse(cnt)
        cv.ellipse(img, ellipse, (0, 255, 255), 2)


# ----------------- core processing -----------------
def _binary_threshold(gray: np.ndarray, thresh: float, invert_polarity: bool) -> np.ndarray:
    t = int(np.clip(thresh, 0, 255))
    mode = cv.THRESH_BINARY_INV if invert_polarity else cv.THRESH_BINARY
    _, bin_img = cv.threshold(gray, t, 255, mode)
    return bin_img


def _binarize_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Assure un masque binaire uint8 {0,255} ou None."""
    if mask is None:
        return None
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # Traite les masques de BG qui contiennent 127 (ombres) : tout >0 devient 255
    _, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
    return mask


def _combine_masks(bin_img: np.ndarray, bg_mask: Optional[np.ndarray], mode: str) -> np.ndarray:
    """Fusionne le seuillage et le masque BG selon OR/AND/REPLACE."""
    bg_mask = _binarize_mask(bg_mask)
    if bg_mask is None:
        return bin_img

    m = (mode or "OR").upper()
    if m.startswith("AND"):
        return cv.bitwise_and(bin_img, bg_mask)
    if m.startswith("REPLACE"):
        return bg_mask
    # défaut : OR
    return cv.bitwise_or(bin_img, bg_mask)


def _morph(bin_img: np.ndarray, ksize: int) -> np.ndarray:
    if ksize < 1:
        return bin_img
    if ksize % 2 == 0:
        ksize += 1
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv.morphologyEx(bin_img, cv.MORPH_OPEN, k, iterations=1)
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, k, iterations=1)
    return closed


def _shape_metrics(cnt):
    # Aire
    a = float(cv.contourArea(cnt))
    # Ratio d'aspect (>=1)
    x, y, w, h = cv.boundingRect(cnt)
    ar = (max(w, h) / max(1, min(w, h)))
    # Circularité [0..1] (1 = cercle parfait)
    p = cv.arcLength(cnt, True)
    circ = (4.0 * np.pi * a / (p * p)) if p > 1e-6 else 0.0
    circ = float(np.clip(circ, 0.0, 1.0))
    return a, ar, circ, (w, h)


def detect_larvae_contours(
    frame: np.ndarray,
    gray_threshold: float,
    contour_area_range: Tuple[float, float],
    morph_ellipse_size: int = 0,
    invert_polarity: bool = False,
    *,
    # --- pour compatibilité avec main.py :
    bg_mask: Optional[np.ndarray] = None,    # masque binaire 0/255 optionnel (venant de KNN/MOG2)
    combine_mode: str = "OR",                # "OR" | "AND" | "REPLACE"
    # --- rétro-compatibilité : si on passe un subtracteur et pas de bg_mask, on le génère ici
    back_sub: Optional[cv.BackgroundSubtractor] = None,
    bs_learning_rate: float = 0.01,
    # --- filtres de forme ---
    min_aspect_ratio: float = 1.40,          # >=1.40 : plutôt allongé
    max_circularity: float = 0.80            # <=0.80 : pas trop rond
) -> Dict[str, np.ndarray | List[np.ndarray]]:
    """
    Retourne un dict:
      {
        "binary_image":  <uint8>,
        "morphed_image": <uint8>,
        "contours":      [np.ndarray, ...]
      }
    """
    # 1) Grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 1.b) Si pas de bg_mask mais un subtracteur fourni, on produit le bg_mask ici (compat)
    if bg_mask is None and back_sub is not None:
        fg = back_sub.apply(gray, learningRate=float(bs_learning_rate))
        # Binarisation (supprime le 127 "shadows")
        _, bg_mask = cv.threshold(fg, 0, 255, cv.THRESH_BINARY)
        # Un léger morph pour nettoyer le BS (plus doux que le morph principal)
        if morph_ellipse_size > 1:
            bg_mask = _morph(bg_mask, max(1, morph_ellipse_size // 2))

    # 2) Seuillage global (option inversion)
    bin_img = _binary_threshold(gray, gray_threshold, invert_polarity)

    # 3) Fusion éventuelle avec le masque BG
    combined = _combine_masks(bin_img, bg_mask, combine_mode)

    # 4) Morphologie (ouvre + ferme)
    morphed = _morph(combined, morph_ellipse_size)

    # 5) Contours externes
    cnts, _ = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 6) Filtres d'aire + forme
    a_min, a_max = contour_area_range
    kept: List[np.ndarray] = []
    for c in cnts:
        a, ar, circ, _wh = _shape_metrics(c)
        if a < a_min or a > a_max:
            continue
        # garder objets allongés OU peu circulaires
        if (ar >= float(min_aspect_ratio)) or (circ <= float(max_circularity)):
            kept.append(c)

    kept.sort(key=lambda c: cv.contourArea(c), reverse=True)

    return {
        "binary_image": bin_img,
        "morphed_image": morphed,
        "contours": kept
    }


def fit_contours(img: np.ndarray, contours: List[np.ndarray], draw_fn: Callable):
    for c in contours:
        draw_fn(img, c)
    return img
