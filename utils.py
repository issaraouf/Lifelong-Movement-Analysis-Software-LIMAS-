# utils.py
from PySide6 import QtGui
import cv2 as cv

def cv_to_qimage(img_bgr_or_gray):
    if img_bgr_or_gray is None:
        return QtGui.QImage()
    if img_bgr_or_gray.ndim == 2:
        img_rgb = cv.cvtColor(img_bgr_or_gray, cv.COLOR_GRAY2RGB)
    else:
        img_rgb = cv.cvtColor(img_bgr_or_gray, cv.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    bytes_per_line = 3 * w
    return QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

def qimage_to_pixmap(qimg):
    return QtGui.QPixmap.fromImage(qimg)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))
