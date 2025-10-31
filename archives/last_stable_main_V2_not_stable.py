# main.py — adds Background Subtraction chooser (Off/KNN/MOG2) + Combine mode (OR/AND/REPLACE)
# Keeps: text editors on sliders, warm-up overlay, invert polarity, dark theme, reset, FPS, 0.75×, frame/detections counter.

import sys, os
import cv2 as cv
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from contour import (
    detect_larvae_contours,
    fit_contours,
    _draw_rectangle, _draw_min_area_rect, _draw_convex_hull, _draw_ellipse
)

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


class VideoReader(QtCore.QThread):
    frameReady = QtCore.Signal(object, int, int)
    ended = QtCore.Signal()

    def __init__(self, path=None, parent=None, target_fps=20):
        super().__init__(parent)
        self.path = path
        self._cap = None
        self._running = False
        self._paused = False
        self.target_fps = max(1, int(target_fps))

    def set_fps(self, fps: int):
        self.target_fps = max(1, int(fps))

    def open(self, path):
        self.path = path

    def run(self):
        if not self.path:
            self.ended.emit(); return
        self._cap = cv.VideoCapture(self.path)
        if not self._cap.isOpened():
            self.ended.emit(); return

        total_frames = int(self._cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
        self._running = True
        self._paused = False

        interval_ms = int(1000 / self.target_fps)
        timer = QtCore.QElapsedTimer(); timer.start()

        frame_index = 0
        while self._running:
            if self._paused:
                self.msleep(5); continue
            ok, frame = self._cap.read()
            if not ok:
                self.ended.emit(); break

            frame_index += 1
            self.frameReady.emit(frame, frame_index, total_frames)

            elapsed = timer.elapsed()
            wait = max(1, interval_ms - (elapsed % interval_ms))
            self.msleep(wait)

        if self._cap:
            self._cap.release()

    def stop(self):
        self._running = False

    def pause(self, p=True):
        self._paused = p


class MainWindow(QtWidgets.QMainWindow):
    DEFAULTS = {
        "fps": 20,
        "gray_threshold": 50,
        "area_min": 200,
        "area_max": 30000,
        "morph_ks": 23,
        "fit_shape": "Rectangle",
        "fast_downscale": False,
        "warmup_frames": 30,
        "invert_polarity": False,
        "bg_method": "Off",          # <— NEW: Off | KNN | MOG2
        "combine_mode": "OR",        # <— NEW: OR | AND | REPLACE
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Embryo Detector (Qt) — original logic")
        self.resize(1480, 900)

        # params
        self.gray_threshold = int(self.DEFAULTS["gray_threshold"])
        self.area_min = int(self.DEFAULTS["area_min"])
        self.area_max = int(self.DEFAULTS["area_max"])
        self.morph_ks = int(self.DEFAULTS["morph_ks"])
        self.fit_shape = self.DEFAULTS["fit_shape"]
        self.fast_downscale = self.DEFAULTS["fast_downscale"]
        self.warmup_frames = int(self.DEFAULTS["warmup_frames"])
        self.invert_polarity = self.DEFAULTS["invert_polarity"]
        self.bg_method = self.DEFAULTS["bg_method"]          # <— NEW
        self.combine_mode = self.DEFAULTS["combine_mode"]    # <— NEW

        self.video_path = ""
        self.frame_index = 0
        self.total_frames = 0
        self.max_display_width = 540
        self.processing_busy = False

        # background subtractor holder
        self.bg_subtractor = None   # created when bg_method != Off

        self._build_ui()
        self._apply_dark_theme()

        self.reader = VideoReader()
        self.reader.frameReady.connect(self.on_frame)
        self.reader.ended.connect(self.on_video_end)

    # ---------- theme ----------
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            * { color: #E8E8E8; }
            QMainWindow { background-color: #1a1b1e; }
            QLabel { color: #E8E8E8; }
            QLabel:hover { color: #79b8ff; }
            QGroupBox {
                color: #E8E8E8;
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 8px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QGroupBox::title:hover { color: #79b8ff; }

            QPushButton { background: #2a2d32; border: 1px solid #555; padding: 6px 10px; }
            QPushButton:hover { background: #33373d; }

            QSlider::groove:horizontal { height: 6px; background: #333; margin: 0 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #888; width: 14px; border-radius: 7px; margin: -5px 0; }
            QSlider::sub-page:horizontal { background: #4a90e2; }

            QLineEdit {
                background: #222; color: #E8E8E8; border: 1px solid #555;
                padding: 2px 6px; border-radius: 3px; min-width: 70px;
            }
            QLineEdit:focus { border: 1px solid #4a90e2; }

            QCheckBox { padding: 2px 0; }

            QComboBox {
                background-color: #2d2d2d;
                color: #E8E8E8;
                border: 1px solid #555;
                padding: 4px 6px;
            }
            QComboBox:hover { color: #79b8ff; }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: none; width: 0; height: 0; }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d; color: #E8E8E8;
                selection-background-color: #444444; selection-color: #FFFFFF;
                outline: none; border: 1px solid #555;
            }

            QToolTip { background-color: #2d2d2d; color: #E8E8E8; border: 1px solid #555; padding: 6px; }
        """)

    # ---------- UI build ----------
    def _build_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        H = QtWidgets.QHBoxLayout(central)

        # controls column
        C = QtWidgets.QVBoxLayout()
        row = QtWidgets.QHBoxLayout()
        self.btn_open  = QtWidgets.QPushButton("Open")
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_stop  = QtWidgets.QPushButton("Stop")
        self.btn_reset = QtWidgets.QPushButton("Reset parameters")
        row.addWidget(self.btn_open); row.addWidget(self.btn_start)
        row.addWidget(self.btn_pause); row.addWidget(self.btn_stop)
        C.addLayout(row)
        C.addWidget(self.btn_reset)

        # Reader FPS
        self.sl_fps, self.ed_fps = self._slider_with_editor(
            C, "Reader FPS", 5, 60, self.DEFAULTS["fps"],
            "Limits how many frames per second are sent from the reader thread to the UI."
        )
        self._connect_slider_editor(self.sl_fps, self.ed_fps, "fps", cast=int)

        # Gray threshold
        self.sl_thresh, self.ed_thresh = self._slider_with_editor(
            C, "Gray threshold", 0, 255, int(self.gray_threshold),
            "Global threshold on grayscale (before morphology)."
        )
        self._connect_slider_editor(self.sl_thresh, self.ed_thresh, "gray_threshold", cast=int)

        # Area min
        self.sl_area_min, self.ed_area_min = self._slider_with_editor(
            C, "Area min", 0, 200000, int(self.area_min),
            "Reject tiny blobs (noise)."
        )
        self._connect_slider_editor(self.sl_area_min, self.ed_area_min, "area_min", cast=int)

        # Area max
        self.sl_area_max, self.ed_area_max = self._slider_with_editor(
            C, "Area max", 0, 300000, int(self.area_max),
            "Reject huge blobs (merged/background)."
        )
        self._connect_slider_editor(self.sl_area_max, self.ed_area_max, "area_max", cast=int)

        # Morph ellipse (odd)
        self.sl_morph, self.ed_morph = self._slider_with_editor(
            C, "Morph ellipse (odd)", 1, 61, int(self.morph_ks),
            "Elliptical kernel size for OPEN then CLOSE."
        )
        self._connect_slider_editor(self.sl_morph, self.ed_morph, "morph_ks", cast=int, enforce_odd=True)
        self.sl_morph.setSingleStep(2); self.sl_morph.setPageStep(2)

        # Fit shape
        C.addWidget(self._label("Fit shape"))
        self.cb_shape = QtWidgets.QComboBox()
        self.cb_shape.addItems(["Rectangle","Min Area Rectangle","Convex Hull","Ellipse"])
        self.cb_shape.setCurrentText(self.fit_shape)
        self.cb_shape.setToolTip("Visualization only.")
        C.addWidget(self.cb_shape)

        # Perf & polarity
        self.cb_fast = QtWidgets.QCheckBox("Process at 0.75× scale (perf)")
        self.cb_fast.setChecked(self.fast_downscale)
        C.addWidget(self.cb_fast)

        self.cb_invert = QtWidgets.QCheckBox("Invert polarity (threshold)")
        self.cb_invert.setChecked(self.invert_polarity)
        self.cb_invert.setToolTip("Use THRESH_BINARY_INV if embryos are darker than background.")
        C.addWidget(self.cb_invert)

        # NEW: Background subtraction method
        C.addWidget(self._label("Background subtraction"))
        self.cb_bg = QtWidgets.QComboBox()
        self.cb_bg.addItems(["Off", "KNN", "MOG2"])
        self.cb_bg.setCurrentText(self.bg_method)
        self.cb_bg.setToolTip("Foreground mask from background subtraction.")
        C.addWidget(self.cb_bg)

        # NEW: Combine mode
        C.addWidget(self._label("Combine mode"))
        self.cb_combine = QtWidgets.QComboBox()
        self.cb_combine.addItems(["OR", "AND", "REPLACE"])
        self.cb_combine.setCurrentText(self.combine_mode)
        self.cb_combine.setToolTip(
            "OR: keep union of threshold and BG mask (safe)\n"
            "AND: intersection (strict)\n"
            "REPLACE: use only BG mask"
        )
        C.addWidget(self.cb_combine)

        # Warm-up
        self.sl_warmup, self.ed_warmup = self._slider_with_editor(
            C, "Warm-up frames", 0, 300, int(self.warmup_frames),
            "Skip detections during first N frames; BG model still learns."
        )
        self._connect_slider_editor(self.sl_warmup, self.ed_warmup, "warmup_frames", cast=int)

        # Live info
        self.live_info = QtWidgets.QLabel("frame: 0 / 0 | detections: 0")
        self.live_info.setStyleSheet("color:#E8E8E8; padding:6px;")
        C.addSpacing(10)
        C.addWidget(self.live_info)

        C.addStretch(1)

        # views
        V = QtWidgets.QVBoxLayout()
        thumbs = QtWidgets.QHBoxLayout()
        self.lbl_bin   = self._pane("Binary")
        self.lbl_morph = self._pane("Morphed")
        self.lbl_cont  = self._pane("Contoured")
        thumbs.addWidget(self.lbl_bin["card"]); thumbs.addWidget(self.lbl_morph["card"]); thumbs.addWidget(self.lbl_cont["card"])
        V.addLayout(thumbs)

        H.addLayout(C, 0); H.addLayout(V, 1)

        # wire buttons
        self.btn_open.clicked.connect(self.pick_video)
        self.btn_start.clicked.connect(self.start)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_reset.clicked.connect(self.reset_params)

        # wire controls
        self.cb_shape.currentTextChanged.connect(lambda s: setattr(self, "fit_shape", s))
        self.cb_fast.toggled.connect(lambda b: setattr(self, "fast_downscale", b))
        self.cb_invert.toggled.connect(lambda b: setattr(self, "invert_polarity", b))
        self.cb_bg.currentTextChanged.connect(self._on_bg_changed)
        self.cb_combine.currentTextChanged.connect(lambda s: setattr(self, "combine_mode", s))

    # ---- UI helpers ----
    def _label(self, txt):
        lab = QtWidgets.QLabel(txt)
        lab.setStyleSheet("color:#E8E8E8; margin-top:8px;")
        return lab

    def _pane(self, title):
        box = QtWidgets.QGroupBox(title)
        L = QtWidgets.QVBoxLayout(box)
        lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        lbl.setMinimumSize(360, 300)
        lbl.setStyleSheet("background:#111; border:1px solid #333;")
        L.addWidget(lbl)
        return {"card": box, "label": lbl}

    def _slider_with_editor(self, parent_layout, title, mn, mx, val, tooltip=""):
        header = QtWidgets.QHBoxLayout()
        lab = self._label(title)
        edit = QtWidgets.QLineEdit(str(val))
        edit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        edit.setValidator(QtGui.QIntValidator(mn, mx, self))
        header.addWidget(lab); header.addStretch(1); header.addWidget(edit)
        parent_layout.addLayout(header)

        s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s.setRange(mn, mx); s.setValue(int(val))
        if tooltip:
            lab.setToolTip(tooltip); s.setToolTip(tooltip); edit.setToolTip(tooltip)
        parent_layout.addWidget(s)
        return s, edit

    def _connect_slider_editor(self, slider, editor, attr_name, cast=int, enforce_odd=False):
        def on_slider(v):
            v = cast(v)
            if enforce_odd and v % 2 == 0:
                v = v + 1 if v < slider.maximum() else v - 1
            if slider.value() != v:
                slider.blockSignals(True); slider.setValue(v); slider.blockSignals(False)
            editor.blockSignals(True); editor.setText(str(v)); editor.blockSignals(False)
            if attr_name == "fps":
                self.reader.set_fps(v)
            elif attr_name == "gray_threshold":
                self.gray_threshold = v
            elif attr_name == "area_min":
                self.area_min = v
            elif attr_name == "area_max":
                self.area_max = v
            elif attr_name == "morph_ks":
                self.morph_ks = v
            elif attr_name == "warmup_frames":
                self.warmup_frames = v
        slider.valueChanged.connect(on_slider)

        def on_edit():
            text = editor.text().strip()
            try:
                v = cast(text)
            except Exception:
                v = slider.value()
            v = max(slider.minimum(), min(slider.maximum(), v))
            if enforce_odd and v % 2 == 0:
                v = v + 1 if v < slider.maximum() else v - 1
            if slider.value() != v:
                slider.setValue(v)
            else:
                on_slider(v)
        editor.editingFinished.connect(on_edit)

    # ---------- controls ----------
    def pick_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open video", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.video_path = path
            self.setWindowTitle(f"Embryo Detector (Qt) — {os.path.basename(path)}")
            self.live_info.setText("frame: 0 / 0 | detections: 0")
            # Reset BG subtractor when new video is chosen
            self._init_bg_subtractor()

    def start(self):
        if not self.video_path:
            self.live_info.setText("Pick a video first."); return
        self.frame_index = 0
        self.total_frames = 0
        # sync reader fps with UI editor
        try:
            self.reader.set_fps(int(self.ed_fps.text()))
        except Exception:
            self.reader.set_fps(self.sl_fps.value())
        self.reader.open(self.video_path)
        if not self.reader.isRunning():
            self.reader.start()
        else:
            self.reader.pause(False)

    def pause(self):
        if self.reader.isRunning():
            self.reader.pause(True)

    def stop(self):
        if self.reader.isRunning():
            self.reader.stop(); self.reader.wait(500)

    def reset_params(self):
        def set_pair(sl, ed, val, odd=False):
            sl.blockSignals(True); sl.setValue(int(val)); sl.blockSignals(False)
            v = int(val)
            if odd and v % 2 == 0:
                v = v + 1 if v < sl.maximum() else v - 1
                sl.setValue(v)
            ed.blockSignals(True); ed.setText(str(int(v))); ed.blockSignals(False)

        set_pair(self.sl_fps, self.ed_fps, self.DEFAULTS["fps"])
        set_pair(self.sl_thresh, self.ed_thresh, self.DEFAULTS["gray_threshold"])
        set_pair(self.sl_area_min, self.ed_area_min, self.DEFAULTS["area_min"])
        set_pair(self.sl_area_max, self.ed_area_max, self.DEFAULTS["area_max"])
        set_pair(self.sl_morph, self.ed_morph, self.DEFAULTS["morph_ks"], odd=True)
        set_pair(self.sl_warmup, self.ed_warmup, self.DEFAULTS["warmup_frames"])

        self.gray_threshold = int(self.DEFAULTS["gray_threshold"])
        self.area_min = int(self.DEFAULTS["area_min"])
        self.area_max = int(self.DEFAULTS["area_max"])
        self.morph_ks = int(self.DEFAULTS["morph_ks"])
        self.warmup_frames = int(self.DEFAULTS["warmup_frames"])
        self.fast_downscale = self.DEFAULTS["fast_downscale"]
        self.invert_polarity = self.DEFAULTS["invert_polarity"]
        self.bg_method = self.DEFAULTS["bg_method"]
        self.combine_mode = self.DEFAULTS["combine_mode"]

        self.cb_shape.setCurrentText(self.DEFAULTS["fit_shape"])
        self.cb_fast.setChecked(self.fast_downscale)
        self.cb_invert.setChecked(self.invert_polarity)
        self.cb_bg.setCurrentText(self.bg_method)
        self.cb_combine.setCurrentText(self.combine_mode)
        self._init_bg_subtractor()

        self.live_info.setText("frame: 0 / 0 | detections: 0")

    # ---------- BG subtractor ----------
    def _on_bg_changed(self, method):
        self.bg_method = method
        self._init_bg_subtractor()

    def _init_bg_subtractor(self):
        m = (self.bg_method or "Off").upper()
        self.bg_subtractor = None
        if m == "KNN":
            # detectShadows=False => binary 0/255; True => adds 127 for shadows
            self.bg_subtractor = cv.createBackgroundSubtractorKNN(detectShadows=True)
        elif m == "MOG2":
            self.bg_subtractor = cv.createBackgroundSubtractorMOG2(detectShadows=True)

    def _apply_bg(self, frame_gray):
        if self.bg_subtractor is None:
            return None
        # Learning rate: default None lets OpenCV choose; you can expose it later if needed
        fg = self.bg_subtractor.apply(frame_gray)
        # Binarize: treat >0 as foreground (removes 127 shadow mid-level)
        _, fg = cv.threshold(fg, 0, 255, cv.THRESH_BINARY)
        return fg

    # ---------- per-frame ----------
    @QtCore.Slot(object, int, int)
    def on_frame(self, frame_bgr, idx, total):
        if self.processing_busy:
            return
        self.processing_busy = True
        try:
            self.frame_index = idx
            self.total_frames = total

            # Pre-resize for perf if requested
            if self.fast_downscale:
                h, w = frame_bgr.shape[:2]
                frame_bgr = cv.resize(frame_bgr, (int(w*0.75), int(h*0.75)), interpolation=cv.INTER_AREA)

            # Prepare gray and (optionally) BG mask first so warm-up can still train the model
            gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
            bg_mask = self._apply_bg(gray) if self.bg_subtractor is not None else None

            # Warm-up: show overlay; still update BG model above
            remaining = max(0, self.warmup_frames - self.frame_index)
            if remaining > 0:
                overlay = frame_bgr.copy()
                cv.putText(
                    overlay, f"Warm-up — {remaining} frames restantes",
                    (12, 28), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA
                )
                def _resize(img):
                    h, w = img.shape[:2]
                    scale = min(1.0, self.max_display_width / float(w))
                    return cv.resize(img, (int(w*scale), int(h*scale)), interpolation=cv.INTER_AREA)
                for img, slot in (
                    (_resize(overlay), self.lbl_bin["label"]),
                    (_resize(overlay), self.lbl_morph["label"]),
                    (_resize(overlay), self.lbl_cont["label"]),
                ):
                    qimg = cv_to_qimage(img)
                    slot.setPixmap(qimage_to_pixmap(qimg))
                total_txt = str(self.total_frames) if self.total_frames > 0 else "?"
                self.live_info.setText(f"frame: {self.frame_index} / {total_txt} | detections: 0")
                return

            # Detection with BG fusion
            results = detect_larvae_contours(
                frame=frame_bgr,
                gray_threshold=int(self.gray_threshold),
                contour_area_range=(int(self.area_min), int(self.area_max)),
                morph_ellipse_size=int(self.morph_ks),
                invert_polarity=bool(self.invert_polarity),
                bg_mask=bg_mask,
                combine_mode=self.combine_mode,
            )
            contours = results.get("contours", [])

            # Overlay with chosen fitted shape
            fit_map = {
                "Rectangle": _draw_rectangle,
                "Min Area Rectangle": _draw_min_area_rect,
                "Convex Hull": _draw_convex_hull,
                "Ellipse": _draw_ellipse,
            }
            fit_fn = fit_map.get(self.fit_shape, _draw_rectangle)

            contoured = frame_bgr.copy()
            if contours:
                fit_contours(contoured, contours, fit_fn)

            # display
            bin_vis = results["binary_image"]
            morph_vis = results["morphed_image"]
            if bin_vis.ndim == 2:   bin_vis = cv.cvtColor(bin_vis, cv.COLOR_GRAY2BGR)
            if morph_vis.ndim == 2: morph_vis = cv.cvtColor(morph_vis, cv.COLOR_GRAY2BGR)

            def _resize(img):
                h, w = img.shape[:2]
                scale = min(1.0, self.max_display_width / float(w))
                return cv.resize(img, (int(w*scale), int(h*scale)), interpolation=cv.INTER_AREA)

            for img, slot in (
                (_resize(bin_vis),   self.lbl_bin["label"]),
                (_resize(morph_vis), self.lbl_morph["label"]),
                (_resize(contoured), self.lbl_cont["label"]),
            ):
                qimg = cv_to_qimage(img)
                slot.setPixmap(qimage_to_pixmap(qimg))

            total_txt = str(self.total_frames) if self.total_frames > 0 else "?"
            self.live_info.setText(f"frame: {self.frame_index} / {total_txt} | detections: {len(contours)}")
        finally:
            self.processing_busy = False

    def on_video_end(self):
        pass

    # ---------- misc ----------
    def _label(self, txt):
        lab = QtWidgets.QLabel(txt)
        lab.setStyleSheet("color:#E8E8E8; margin-top:8px;")
        return lab

    def _pane(self, title):
        box = QtWidgets.QGroupBox(title)
        L = QtWidgets.QVBoxLayout(box)
        lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        lbl.setMinimumSize(360, 300)
        lbl.setStyleSheet("background:#111; border:1px solid #333;")
        L.addWidget(lbl)
        return {"card": box, "label": lbl}


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
