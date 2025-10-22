# contour.py  (version avec warm-up anti-parasites + combine "safe")

import cv2 as cv
import numpy as np
from typing import List, Tuple, Dict, Callable

# ----------------- helpers: drawing funcs (inchangés) -----------------
def _draw_rectangle(img, cnt):
    x, y, w, h = cv.boundingRect(cnt)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

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
    """
    Seuil binaire sur l'image en niveaux de gris.
    """
    t = int(np.clip(thresh, 0, 255))
    mode = cv.THRESH_BINARY_INV if invert_polarity else cv.THRESH_BINARY
    _, bin_img = cv.threshold(gray, t, 255, mode)
    return bin_img

def _combine_masks(bin_img: np.ndarray, bg_mask: np.ndarray | None, mode: str) -> np.ndarray:
    """
    Combine le masque de seuillage (thr) et le masque BG (KNN/MOG2) selon 'mode'.

    Pare-chocs "safe":
      - si bg_mask est None  -> retourne bin_img
      - si bg_mask est quasi vide (<0.05% de pixels on) -> retourne bin_img
      - sinon applique: AND / OR / Replace (BG only)

    Le parsing du mode est insensible à la casse et tolère des suffixes:
      "or", "OR (Safe)", "and", "AND (Strict)", "replace", "Replace (BG only)"...
    """
    if bg_mask is None:
        return bin_img

    # S'assurer du type 8 bits
    if bg_mask.dtype != np.uint8:
        bg_mask = bg_mask.astype(np.uint8)

    # Si le masque BG est quasi vide, éviter d'éteindre la détection
    fg_ratio = float((bg_mask > 0).sum()) / (bg_mask.size + 1e-9)
    if fg_ratio < 0.0005:  # 0.05% de pixels foreground
        return bin_img

    m = (mode or "or").strip().lower()
    if m.startswith("and"):
        return cv.bitwise_and(bin_img, bg_mask)
    if m.startswith("replace"):
        return bg_mask
    # défaut: OR = comportement "safe"
    return cv.bitwise_or(bin_img, bg_mask)

def _morph(bin_img: np.ndarray, ksize: int) -> np.ndarray:
    """
    Morphologie (open puis close) avec noyau ellipse de taille impaire.
    """
    if ksize < 1:
        return bin_img
    if ksize % 2 == 0:
        ksize += 1
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv.morphologyEx(bin_img, cv.MORPH_OPEN, k, iterations=1)
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, k, iterations=1)
    return closed

def _shape_metrics(cnt):
    """
    Retourne (aire, aspect_ratio>=1, circularité[0..1], (w,h)).
    """
    a = float(cv.contourArea(cnt))
    x, y, w, h = cv.boundingRect(cnt)
    ar = (max(w, h) / max(1, min(w, h)))
    p = cv.arcLength(cnt, True)
    circ = (4.0 * np.pi * a / (p * p)) if p > 1e-6 else 0.0
    circ = float(np.clip(circ, 0.0, 1.0))
    return a, ar, circ, (w, h)

class WarmupDetector:
    """Classe pour gérer la période de warm-up et réduire les parasites au démarrage"""
    def __init__(self, warmup_frames: int = 30):
        self.warmup_frames = warmup_frames
        self.frame_count = 0
        self.bg_initialized = False  # (réservé, si besoin)

    def reset(self):
        """Reset le compteur de warm-up"""
        self.frame_count = 0
        self.bg_initialized = False

    def is_warmed_up(self) -> bool:
        """Vérifie si la période de warm-up est terminée"""
        return self.frame_count >= self.warmup_frames

    def should_suppress_detections(self) -> bool:
        """Détermine si on doit filtrer plus strictement (pendant warm-up)"""
        return not self.is_warmed_up()

    def get_warmup_progress(self) -> float:
        """Retourne le progrès du warm-up (0.0 à 1.0)"""
        if self.warmup_frames <= 0:
            return 1.0
        return min(1.0, self.frame_count / self.warmup_frames)

# Instance globale du détecteur de warm-up
_warmup_detector = WarmupDetector()

def reset_warmup():
    """Fonction publique pour reset le warm-up"""
    global _warmup_detector
    _warmup_detector.reset()

def get_warmup_status() -> Tuple[bool, float]:
    """Retourne (is_warmed_up, progress)"""
    global _warmup_detector
    return _warmup_detector.is_warmed_up(), _warmup_detector.get_warmup_progress()

def detect_larvae_contours(
    frame: np.ndarray,
    gray_threshold: float,
    contour_area_range: Tuple[float, float],
    morph_ellipse_size: int = 0,
    back_sub = None,
    bs_learning_rate: float = 0.01,
    combine_mode: str = "OR (Safe)",
    invert_polarity: bool = False,
    *,
    # --- filtres de forme ---
    min_aspect_ratio: float = 1.40,
    max_circularity: float = 0.80,
    # --- paramètres de warm-up ---
    enable_warmup: bool = True,
    warmup_frames: int = 30
) -> Dict[str, np.ndarray | List[np.ndarray]]:
    """
    Exécute la segmentation et retourne:
      - 'binary_image'  : masque seuillé (avant morph)
      - 'morphed_image' : masque final morphé
      - 'contours'      : liste de contours filtrés
      - 'warmup_active' : bool
      - 'warmup_progress' : [0..1]
    """
    global _warmup_detector

    # Mise à jour du warm-up
    _warmup_detector.frame_count += 1
    if _warmup_detector.warmup_frames != warmup_frames:
        _warmup_detector.warmup_frames = warmup_frames

    # grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # background subtraction (optionnel)
    bg_mask = None
    if back_sub is not None:
        # Pendant le warm-up, learning rate temporairement plus élevé
        current_lr = bs_learning_rate
        if enable_warmup and not _warmup_detector.is_warmed_up():
            progress = _warmup_detector.get_warmup_progress()
            current_lr = bs_learning_rate + (0.3 - bs_learning_rate) * (1.0 - progress)
            current_lr = max(bs_learning_rate, min(0.5, current_lr))

        fg = back_sub.apply(gray, learningRate=float(current_lr))
        _, bg_mask = cv.threshold(fg, 127, 255, cv.THRESH_BINARY)

        # Morphologie renforcée pendant warm-up pour débruiter le BG
        morph_size = morph_ellipse_size
        if enable_warmup and not _warmup_detector.is_warmed_up():
            morph_size = max(morph_ellipse_size, morph_ellipse_size + 4)
        bg_mask = _morph(bg_mask, max(1, morph_size // 2))

    # threshold (masque de base)
    bin_img = _binary_threshold(gray, gray_threshold, invert_polarity)

    # combine avec BG mask (safe si None ou quasi vide)
    combined = _combine_masks(bin_img, bg_mask, combine_mode)

    # morph finale (sur le masque combiné)
    morphed = _morph(combined, morph_ellipse_size)

    # find contours
    cnts, _ = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Filtrage pendant le warm-up (progressif et plus strict)
    kept: List[np.ndarray] = []
    if enable_warmup and _warmup_detector.should_suppress_detections():
        a_min, a_max = contour_area_range
        progress = _warmup_detector.get_warmup_progress()

        # Aire plus stricte au début, relâchée avec le temps
        strict_factor = 1.0 + 2.0 * (1.0 - progress)  # 3x plus strict au départ
        a_min_strict = a_min * strict_factor
        a_max_strict = a_max / strict_factor

        # Forme plus stricte au début
        min_ar_strict = min_aspect_ratio + 0.5 * (1.0 - progress)
        max_circ_strict = max_circularity - 0.2 * (1.0 - progress)

        for c in cnts:
            a, ar, circ, _wh = _shape_metrics(c)
            if a < a_min_strict or a > a_max_strict:
                continue
            if (ar >= float(min_ar_strict)) or (circ <= float(max_circ_strict)):
                kept.append(c)

        # Limiter le nombre de détections pendant le warm-up
        max_detections = max(1, int(5 * progress))  # 0->1 au début, jusqu'à ~5
        if len(kept) > max_detections:
            kept = sorted(kept, key=cv.contourArea, reverse=True)[:max_detections]

    else:
        # Fonctionnement normal après warm-up
        a_min, a_max = contour_area_range
        for c in cnts:
            a, ar, circ, _wh = _shape_metrics(c)
            if a < a_min or a > a_max:
                continue
            if (ar >= float(min_aspect_ratio)) or (circ <= float(max_circularity)):
                kept.append(c)

    return {
        "binary_image": bin_img,
        "morphed_image": morphed,
        "contours": kept,
        "warmup_active": enable_warmup and not _warmup_detector.is_warmed_up(),
        "warmup_progress": _warmup_detector.get_warmup_progress()
    }

def fit_contours(img: np.ndarray, contours: List[np.ndarray], draw_fn: Callable):
    for c in contours:
        draw_fn(img, c)
