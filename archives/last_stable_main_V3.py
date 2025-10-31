# main.py — Embryo Detector (Qt) + Integrated MultiTracker (IDs + CSV)
# Keeps your warm-up pipeline & UI; adds a small "Tracking & CSV" panel.
import sys, os
import cv2 as cv
import numpy as np
from collections import deque
from PySide6 import QtCore, QtGui, QtWidgets

from contour import (
    detect_larvae_contours,
    fit_contours,
    reset_warmup,
    get_warmup_status,
    _draw_rectangle, _draw_min_area_rect, _draw_convex_hull, _draw_ellipse
)
from tracker import MultiTracker, TrackParams  # <-- integrate tracker (IDs, CSV)  # noqa

# optional plotting (if installed)
try:
    import pyqtgraph as pg
    PG_AVAILABLE = True
except Exception:
    PG_AVAILABLE = False


# ---------------- utils ----------------
def cv_to_qimage(img_bgr_or_gray):
    if img_bgr_or_gray is None:
        return QtGui.QImage()
    if img_bgr_or_gray.ndim == 2:
        img_rgb = cv.cvtColor(img_bgr_or_gray, cv.COLOR_GRAY2RGB)
    else:
        img_rgb = cv.cvtColor(img_bgr_or_gray, cv.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    return QtGui.QImage(img_rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)

def qimage_to_pixmap(qimg):
    return QtGui.QPixmap.fromImage(qimg)

def contour_centroid_w_h_area(c):
    m = cv.moments(c)
    if m["m00"] == 0:
        x, y, w, h = cv.boundingRect(c)
        return (x + w/2.0, y + h/2.0), (w, h), float(cv.contourArea(c))
    cx = m["m10"]/m["m00"]; cy = m["m01"]/m["m00"]
    x, y, w, h = cv.boundingRect(c)
    return (cx, cy), (w, h), float(cv.contourArea(c))


class ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal()
    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)


class VideoReader(QtCore.QThread):
    frameReady = QtCore.Signal(object)
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

        self._running, self._paused = True, False
        interval_ms = int(1000 / self.target_fps)
        timer = QtCore.QElapsedTimer(); timer.start()

        while self._running:
            if self._paused:
                self.msleep(5); continue
            ok, frame = self._cap.read()
            if not ok:
                self.ended.emit(); break
            self.frameReady.emit(frame)
            elapsed = timer.elapsed()
            wait = max(1, interval_ms - (elapsed % interval_ms))
            self.msleep(wait)

        if self._cap:
            self._cap.release()

    def stop(self):
        self._running = False

    def pause(self, p=True):
        self._paused = p


# ---------------- main window ----------------
class MainWindow(QtWidgets.QMainWindow):
    DEFAULTS = {
        "fps": 20,
        "gray_threshold": 50.0,
        "area_min": 200.0,
        "area_max": 30000.0,
        "morph_ks": 23,
        "fit_shape": "Rectangle",
        "fast_downscale": False,
        "bg_mode": "None",
        "bg_combine": "OR (Safe)",
        "bg_lr": 0.01,
        "move_thresh": 5,
        "warmup_frames": 30,  # warm-up default
    }

    # tracking (for plots)
    MAX_TRACKS = 16
    MATCH_MAX_DIST = 80.0

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Embryo Detector (Qt) - Anti-parasites + IDs/CSV")
        self.resize(1450, 900)

        # params
        self.gray_threshold = self.DEFAULTS["gray_threshold"]
        self.area_min = self.DEFAULTS["area_min"]
        self.area_max = self.DEFAULTS["area_max"]
        self.morph_ks = int(self.DEFAULTS["morph_ks"])
        self.fit_shape = self.DEFAULTS["fit_shape"]
        self.fast_downscale = self.DEFAULTS["fast_downscale"]
        self.bg_mode = self.DEFAULTS["bg_mode"]
        self.bg_combine = self.DEFAULTS["bg_combine"]
        self.bg_lr = float(self.DEFAULTS["bg_lr"])
        self.move_thresh = int(self.DEFAULTS["move_thresh"])
        self.warmup_frames = int(self.DEFAULTS["warmup_frames"])
        self.bg_subtractor = None

        # state
        self.video_path = ""
        self.frame_index = 0
        self._warmup_finished_notified = False
        self.total_frames = 0
        self.max_display_width = 540
        self.processing_busy = False

        # plotting toggle
        self.enable_plots = False

        # tracks for plotting
        self.tracks = {}
        self.next_track_id = 1
        self.color_cache = {}

        self._plot_refresh_enabled = PG_AVAILABLE

        # ---- NEW: MultiTracker integration (IDs + CSV) ----
        self.tracker = MultiTracker(
            TrackParams(
                max_dist=120.0,       # distance gate for association
                area_tolerance=0.50,  # +-50% area gate
                ema_pos=0.40,         # smoothing for positions
                ema_area=0.40,        # smoothing for area
            )
        )  # IDs + CSV handled internally. :contentReference[oaicite:5]{index=5}
        
        """ self.tracker = MultiTracker(TrackParams(
        max_dist=300.0, area_tolerance=0.60, ema_pos=0.40, ema_area=0.40,
        max_miss=12, min_age_csv=6
        ))
 """
        self._build_ui()
        self._apply_dark_theme()
        self._init_plots()

        # reader thread
        self.reader = VideoReader()
        self.reader.frameReady.connect(self.on_frame)
        self.reader.ended.connect(self.on_video_end)

        # throttle timer (only when plots enabled)
        if PG_AVAILABLE:
            self._plot_timer = QtCore.QTimer(self)
            self._plot_timer.setInterval(120)
            self._plot_timer.timeout.connect(self._update_plots_timer_tick)

        # debounce resize
        self._resize_debounce = QtCore.QTimer(self)
        self._resize_debounce.setSingleShot(True)
        self._resize_debounce.setInterval(300)
        self._resize_debounce.timeout.connect(self._resume_plot_refresh)
        
        # throttle UI updates to prevent vibration
        self._ui_update_timer = QtCore.QTimer(self)
        self._ui_update_timer.setSingleShot(True)
        self._ui_update_timer.setInterval(100)
        self._ui_update_timer.timeout.connect(self._update_ui_labels)
        self._pending_ui_data = None

    # ---------- UI / theme ----------
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            * { color:#E8E8E8; }
            QMainWindow { background:#1a1b1e; }
            QLabel { color:#E8E8E8; }
            QLabel:hover { color:#79b8ff; }

            QGroupBox { color:#E8E8E8; border:1px solid #444; border-radius:4px; margin-top:8px; }
            QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 4px; }

            QPushButton { background:#2a2d32; border:1px solid #555; padding:6px 10px; }
            QPushButton:hover { background:#33373d; }

            QSlider::groove:horizontal { height:6px; background:#333; margin:0 6px; border-radius:3px; }
            QSlider::handle:horizontal { background:#888; width:14px; border-radius:7px; margin:-5px 0; }
            QSlider::sub-page:horizontal { background:#4a90e2; }

            QCheckBox { padding:2px 0; }

            QComboBox { background:#2d2d2d; color:#E8E8E8; border:1px solid #555; padding:4px 6px; }
            QComboBox:hover { color:#79b8ff; }
            QComboBox QAbstractItemView { background:#2d2d2d; color:#E8E8E8;
                selection-background-color:#444; border:1px solid #555; }

            QToolTip { background:#2d2d2d; color:#E8E8E8; border:1px solid #555; padding:6px; }

            QProgressBar { border:1px solid #555; border-radius:3px; text-align:center; }
            QProgressBar::chunk { background:#ff6b35; border-radius:2px; }

            /* --- NOUVEAU : garder la colonne gauche sombre malgré le scroll --- */
            QScrollArea { background:#1a1b1e; border:0; }
            QScrollArea > QWidget { background:#1a1b1e; }           /* viewport */
            QScrollArea > QWidget > QWidget { background:#1a1b1e; } /* left_panel */

            QScrollBar:vertical, QScrollBar:horizontal { background:#1a1b1e; }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background:#444; border-radius:4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                height:0; width:0;
            }
        """)

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

    def _labeled_slider(self, parent_layout, title, mn, mx, val, tooltip=""):
        header = QtWidgets.QHBoxLayout()
        lab = self._label(title)
        val_lbl = ClickableLabel(str(val))
        val_lbl.setMinimumWidth(60)
        val_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        header.addWidget(lab); header.addStretch(1); header.addWidget(val_lbl)
        parent_layout.addLayout(header)
        s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s.setRange(mn, mx); s.setValue(val)
        tip = (tooltip + "\n\nTip: click the number to type an exact value.") if tooltip else "Click the number to type."
        lab.setToolTip(tooltip); s.setToolTip(tooltip); val_lbl.setToolTip(tip)
        parent_layout.addWidget(s)
        return s, val_lbl

    def _install_value_editor(self, value_label: ClickableLabel, slider: QtWidgets.QSlider, *, morph_odd: bool=False):
        def on_click():
            mn, mx = slider.minimum(), slider.maximum()
            val, ok = QtWidgets.QInputDialog.getInt(self, "Set value", f"Enter a value [{mn}..{mx}]:", slider.value(), mn, mx, 1)
            if ok:
                if morph_odd and val % 2 == 0:
                    val += 1
                slider.setValue(val); value_label.setText(str(val))
        value_label.clicked.connect(on_click)

    def _build_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        H = QtWidgets.QHBoxLayout(central)
        
        # left controls -> dans un QWidget pour pouvoir le mettre dans un QScrollArea
        left_panel = QtWidgets.QWidget()
        C = QtWidgets.QVBoxLayout(left_panel)
        C.setContentsMargins(8, 8, 8, 8)
        C.setSpacing(6)

        row = QtWidgets.QHBoxLayout()
        self.btn_open  = QtWidgets.QPushButton("Open")
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_stop  = QtWidgets.QPushButton("Stop")
        row.addWidget(self.btn_open); row.addWidget(self.btn_start)
        row.addWidget(self.btn_pause); row.addWidget(self.btn_stop)
        C.addLayout(row)

        self.btn_reset = QtWidgets.QPushButton("Reset parameters")
        C.addWidget(self.btn_reset)

        # --- Warm-up controls (kept) ---
        warmup_group = QtWidgets.QGroupBox("Anti-parasites (warm-up)")
        WL = QtWidgets.QVBoxLayout(warmup_group)
        
        self.cb_enable_warmup = QtWidgets.QCheckBox("Activer le warm-up anti-parasites")
        self.cb_enable_warmup.setChecked(True)
        self.cb_enable_warmup.setToolTip(
            "Active le système anti-parasites qui réduit les fausses détections "
            "pendant les premières frames le temps que la soustraction de fond se stabilise."
        )
        WL.addWidget(self.cb_enable_warmup)
        
        self.sl_warmup, self.val_warmup = self._labeled_slider(
            WL, "Durée warm-up (frames)", 10, 100, self.warmup_frames,
            "Nombre de frames pendant lesquelles les filtres anti-parasites sont actifs."
        )
        self._install_value_editor(self.val_warmup, self.sl_warmup)
        
        self.warmup_progress = QtWidgets.QProgressBar()
        self.warmup_progress.setVisible(False)
        self.warmup_progress.setFormat("Warm-up: %p%")
        WL.addWidget(self.warmup_progress)
        C.addWidget(warmup_group)

        # --- Usual detection controls (kept) ---
        self.sl_fps, self.val_fps = self._labeled_slider(C, "Reader FPS", 5, 60, int(self.DEFAULTS["fps"]),
            "Playback speed (not detection accuracy).")
        self.sl_thresh, self.val_thresh = self._labeled_slider(C, "Gray threshold", 0, 255, int(self.DEFAULTS["gray_threshold"]),
            "Global threshold on grayscale. Pixels > threshold become foreground (unless inverted).")
        self.sl_area_min, self.val_area_min = self._labeled_slider(C, "Area min", 0, 200000, int(self.DEFAULTS["area_min"]),
            "Reject small blobs/noise.")
        self.sl_area_max, self.val_area_max = self._labeled_slider(C, "Area max", 0, 300000, int(self.DEFAULTS["area_max"]),
            "Reject very large blobs/fusions/background.")
        self.sl_morph, self.val_morph = self._labeled_slider(C, "Morph ellipse (odd)", 1, 61, int(self.DEFAULTS["morph_ks"]),
            "Open then close with an elliptical kernel (odd size).")
        self.sl_morph.setSingleStep(2); self.sl_morph.setPageStep(2)

        self.cb_invert = QtWidgets.QCheckBox("Invert polarity (dark objects)")
        self.cb_invert.setToolTip(
            "ON: detect dark objects on bright background (THRESH_BINARY_INV).\n"
            "OFF: detect bright objects on dark background (THRESH_BINARY)."
        )
        C.addWidget(self.cb_invert)

        C.addWidget(self._label("Fit shape"))
        self.cb_shape = QtWidgets.QComboBox()
        self.cb_shape.addItems(["Rectangle","Min Area Rectangle","Convex Hull","Ellipse"])
        self.cb_shape.setCurrentText(self.fit_shape)
        C.addWidget(self.cb_shape)

        self.cb_fast = QtWidgets.QCheckBox("Process at 0.75× scale (perf)")
        self.cb_fast.setChecked(self.fast_downscale)
        C.addWidget(self.cb_fast)

        self.sl_move, self.val_move = self._labeled_slider(
            C, "Moving threshold (px/s)", 0, 200, int(self.DEFAULTS["move_thresh"]),
            "Speed threshold in pixels/second to decide 'moving vs static' (used by analytics)."
        )
        self._install_value_editor(self.val_move, self.sl_move)

        C.addWidget(self._label("Background subtraction"))
        self.cb_bg = QtWidgets.QComboBox()
        self.cb_bg.addItems(["None","KNN","MOG2"])
        self.cb_bg.setCurrentText(self.bg_mode)
        C.addWidget(self.cb_bg)

        C.addWidget(self._label("Mask combine mode"))
        self.cb_combine = QtWidgets.QComboBox()
        self.cb_combine.addItems(["OR (Safe)","AND (Strict)","Replace (BG only)"])
        self.cb_combine.setCurrentText(self.bg_combine)
        C.addWidget(self.cb_combine)

        self.sl_bg_lr_label = self._label("BG learning rate")
        C.addWidget(self.sl_bg_lr_label)
        self.sl_bg_lr = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_bg_lr.setRange(0, 100)
        self.sl_bg_lr.setValue(int(round(self.bg_lr * 100)))
        C.addWidget(self.sl_bg_lr)
        self.lbl_bg_lr_val = ClickableLabel(f"{self.bg_lr:.3f}")
        self.lbl_bg_lr_val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        C.addWidget(self.lbl_bg_lr_val)

        # plotting toggle
        self.chk_plots = QtWidgets.QCheckBox("Show metrics (area / width / height)")
        self.chk_plots.setChecked(False)
        self.chk_plots.setToolTip("ON: charts visible and updated. OFF: metric computation disabled for faster tuning.")
        C.addWidget(self.chk_plots)

        # ---- NEW: Tracking & CSV controls ----
        track_box = QtWidgets.QGroupBox("Tracking & CSV (IDs)")
        TL = QtWidgets.QVBoxLayout(track_box)

        self.cb_record = QtWidgets.QCheckBox("Record CSV")
        self.cb_record.setToolTip("When ON, write per-ID CSVs and/or a combined CSV into ./results/.")
        TL.addWidget(self.cb_record)

        row_csv = QtWidgets.QHBoxLayout()
        self.cb_per_id = QtWidgets.QCheckBox("Per-ID CSV")
        self.cb_per_id.setChecked(True)
        self.cb_combined = QtWidgets.QCheckBox("Combined CSV")
        self.cb_combined.setChecked(False)
        row_csv.addWidget(self.cb_per_id); row_csv.addWidget(self.cb_combined); row_csv.addStretch(1)
        TL.addLayout(row_csv)

        row_mv = QtWidgets.QHBoxLayout()
        self.cb_only_moving = QtWidgets.QCheckBox("Only moving IDs")
        self.cb_only_moving.setChecked(True)
        row_mv.addWidget(self.cb_only_moving); row_mv.addStretch(1)
        TL.addLayout(row_mv)

        self.sl_static_win, self.val_static_win = self._labeled_slider(
            TL, "Motion window (frames)", 2, 60, 12,
            "Sliding window size to measure motion per track."
        )
        self.sl_min_motion, self.val_min_motion = self._labeled_slider(
            TL, "Min motion (px/step)", 0, 20, 2,
            "Minimum average step to consider a track as 'moving'."
        )
        self._install_value_editor(self.val_static_win, self.sl_static_win)
        self._install_value_editor(self.val_min_motion, self.sl_min_motion)

        C.addWidget(track_box)

        # info labels (kept)
        self.info_label = QtWidgets.QLabel("frame: 0 / 0 | detections: 0")
        self.info_label.setStyleSheet("color:#E8E8E8; padding-top:12px;")
        C.addWidget(self.info_label)

        self.status_inline = QtWidgets.QLabel("")
        self.status_inline.setStyleSheet("color:#E8E8E8; padding-top:4px;")
        C.addWidget(self.status_inline)
        C.addStretch(1)

        # right views + metrics row
        V = QtWidgets.QVBoxLayout()
        thumbs = QtWidgets.QHBoxLayout()
        self.lbl_bin   = self._pane("Binary")
        self.lbl_morph = self._pane("Morphed")
        self.lbl_cont  = self._pane("Contoured + IDs")
        thumbs.addWidget(self.lbl_bin["card"])
        thumbs.addWidget(self.lbl_morph["card"])
        thumbs.addWidget(self.lbl_cont["card"])
        V.addLayout(thumbs)

        self.metrics_box = QtWidgets.QGroupBox("Metrics (live)")
        M = QtWidgets.QHBoxLayout(self.metrics_box)
        if PG_AVAILABLE:
            try: pg.setConfigOptions(useOpenGL=False, antialias=False)
            except Exception: pass
            self.plot_area = pg.PlotWidget(title="Mean area (per embryo)")
            self.plot_w    = pg.PlotWidget(title="Mean width (per embryo)")
            self.plot_h    = pg.PlotWidget(title="Mean height (per embryo)")
            for pw in (self.plot_area, self.plot_w, self.plot_h):
                pw.setBackground('#1a1b1e'); pw.showGrid(x=True, y=True, alpha=0.2)
                pw.getAxis('left').setTextPen('#E8E8E8'); pw.getAxis('bottom').setTextPen('#E8E8E8')
                pw.getPlotItem().getAxis('left').setPen('#555'); pw.getPlotItem().getAxis('bottom').setPen('#555')
                pw.setFixedHeight(180); pw.setMinimumWidth(260)
            M.addWidget(self.plot_area, 1); M.addWidget(self.plot_w, 1); M.addWidget(self.plot_h, 1)
        else:
            M.addWidget(QtWidgets.QLabel("pyqtgraph not installed — no charts"))
        V.addWidget(self.metrics_box)
        self.metrics_box.setVisible(False)  # hidden by default

        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("color:#E8E8E8; padding:6px;")
        V.addWidget(self.status)

        #H.addLayout(C, 0); 
        
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(left_panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(360)   # ajuste si tu veux
        scroll.setMaximumWidth(420)   # ajuste si tu veux
        H.addWidget(scroll, 0)

        H.addLayout(V, 1)

        # wiring
        self.btn_open.clicked.connect(self.pick_video)
        self.btn_start.clicked.connect(self.start)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_reset.clicked.connect(self.reset_params)

        self.sl_fps.valueChanged.connect(lambda v: (self.reader.set_fps(v), self.val_fps.setText(str(v))))
        self.sl_thresh.valueChanged.connect(lambda v: (setattr(self,"gray_threshold",float(v)), self.val_thresh.setText(str(v))))
        self.sl_area_min.valueChanged.connect(lambda v: (setattr(self,"area_min",float(v)), self.val_area_min.setText(str(v))))
        self.sl_area_max.valueChanged.connect(lambda v: (setattr(self,"area_max",float(v)), self.val_area_max.setText(str(v))))
        self.sl_morph.valueChanged.connect(self._update_morph)
        self.sl_warmup.valueChanged.connect(lambda v: (setattr(self,"warmup_frames",int(v)), self.val_warmup.setText(str(v))))
        self.cb_shape.currentTextChanged.connect(lambda s: setattr(self,"fit_shape",s))
        self.cb_fast.toggled.connect(lambda b: setattr(self,"fast_downscale",b))
        self.cb_bg.currentTextChanged.connect(self._bg_mode_changed)
        self.cb_combine.currentTextChanged.connect(self._bg_combine_changed)
        self.sl_bg_lr.valueChanged.connect(self._bg_lr_changed)
        self.lbl_bg_lr_val.clicked.connect(self._edit_bg_lr_value)
        self.chk_plots.toggled.connect(self._toggle_plots)
        self.sl_move.valueChanged.connect(lambda v: (setattr(self, "move_thresh", int(v)), self.val_move.setText(str(v))))

        # NEW: tracking params bindings
        self.cb_record.toggled.connect(self._toggle_recording)
        self.cb_per_id.toggled.connect(lambda _: None)   # applied at start()
        self.cb_combined.toggled.connect(lambda _: None) # applied at start()
        self.cb_only_moving.toggled.connect(self._apply_motion_filter)
        self.sl_static_win.valueChanged.connect(self._apply_motion_filter)
        self.sl_min_motion.valueChanged.connect(self._apply_motion_filter)

        # value editors
        self._install_value_editor(self.val_fps, self.sl_fps)
        self._install_value_editor(self.val_thresh, self.sl_thresh)
        self._install_value_editor(self.val_area_min, self.sl_area_min)
        self._install_value_editor(self.val_area_max, self.sl_area_max)
        self._install_value_editor(self.val_morph, self.sl_morph, morph_odd=True)

    def _init_plots(self):
        if not PG_AVAILABLE:
            return
        for tr in self.tracks.values():
            for c in ("curve_area","curve_w","curve_h"):
                item = tr.get(c)
                if item is not None:
                    try: item.clear(); item.deleteLater()
                    except Exception: pass
        self.tracks.clear()
        self.next_track_id = 1
        self.color_cache.clear()
        for pw in (getattr(self, "plot_area", None),
                   getattr(self, "plot_w", None),
                   getattr(self, "plot_h", None)):
            if pw:
                pw.clear()
                pw.enableAutoRange('xy', True)

    # ---------- control handlers ----------
    def pick_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open video", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.video_path = path
            self.setWindowTitle(f"Embryo Detector (Qt) — {os.path.basename(path)}")
            cap = cv.VideoCapture(self.video_path)
            self.total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0) if cap.isOpened() else 0
            if cap: cap.release()
            self.info_label.setText(f"frame: 0 / {self.total_frames} | detections: 0")
            self.status_inline.setText("Video loaded. Click Start.")
            if self.enable_plots: self._init_plots()

    def start(self):
        if not self.video_path:
            self.status_inline.setText("Pick a video first."); return
        self.frame_index = 0
        self._warmup_finished_notified = False
        
        # Reset warm-up system
        reset_warmup()
        self.warmup_progress.setVisible(self.cb_enable_warmup.isChecked())
        if self.cb_enable_warmup.isChecked():
            self.warmup_progress.setMaximum(self.warmup_frames)
            self.warmup_progress.setValue(0)
            self.warmup_progress.setFormat("Warm-up: 0%")
        
        # (Re)create BG subtractor
        self._ensure_bg_subtractor()

        # ---- NEW: start a fresh tracking session (CSV folder ./results/) ----
        per_id = self.cb_per_id.isChecked()
        combined = self.cb_combined.isChecked()
        try:
            self.tracker.start_new_video(self.video_path, per_id_csv=per_id, enable_combined=combined)
        except Exception as e:
            self.status_inline.setText(f"Tracker init failed: {e}")
        # apply motion filter knobs
        self._apply_motion_filter()

        # recording state
        self.tracker.set_recording(self.cb_record.isChecked())

        # start reader
        self.reader.open(self.video_path)
        self.reader.set_fps(self.sl_fps.value())
        if not self.reader.isRunning():
            self.reader.start()
        else:
            self.reader.pause(False)
        self.status_inline.setText("Running… (warm-up actif)" if self.cb_enable_warmup.isChecked() else "Running…")

    def pause(self):
        if self.reader.isRunning():
            self.reader.pause(True); self.status_inline.setText("Paused.")

    def stop(self):
        if self.reader.isRunning():
            self.reader.stop(); self.reader.wait(500)
        self.status_inline.setText("Stopped.")
        self.warmup_progress.setVisible(False)
        # close CSV files cleanly
        try: self.tracker.close_all()
        except: pass
        if self.enable_plots: self._init_plots()

    def reset_params(self):
        self.sl_fps.setValue(int(self.DEFAULTS["fps"])); self.val_fps.setText(str(int(self.DEFAULTS["fps"])))
        self.sl_thresh.setValue(int(self.DEFAULTS["gray_threshold"])); self.val_thresh.setText(str(int(self.DEFAULTS["gray_threshold"])))
        self.sl_area_min.setValue(int(self.DEFAULTS["area_min"])); self.val_area_min.setText(str(int(self.DEFAULTS["area_min"])))
        self.sl_area_max.setValue(int(self.DEFAULTS["area_max"])); self.val_area_max.setText(str(int(self.DEFAULTS["area_max"])))
        self.sl_morph.setValue(int(self.DEFAULTS["morph_ks"])); self.val_morph.setText(str(int(self.DEFAULTS["morph_ks"])))
        self.sl_warmup.setValue(int(self.DEFAULTS["warmup_frames"])); self.val_warmup.setText(str(int(self.DEFAULTS["warmup_frames"])))
        
        self.gray_threshold = self.DEFAULTS["gray_threshold"]
        self.area_min = self.DEFAULTS["area_min"]
        self.area_max = self.DEFAULTS["area_max"]
        self.morph_ks = int(self.DEFAULTS["morph_ks"])
        self.warmup_frames = int(self.DEFAULTS["warmup_frames"])
        
        self.cb_invert.setChecked(False)
        self.cb_enable_warmup.setChecked(True)
        self.cb_shape.setCurrentText(self.DEFAULTS["fit_shape"])
        self.fast_downscale = self.DEFAULTS["fast_downscale"]; self.cb_fast.setChecked(self.fast_downscale)
        self.cb_bg.setCurrentText(self.DEFAULTS["bg_mode"]); self.bg_mode = self.DEFAULTS["bg_mode"]; self.bg_subtractor = None
        self.cb_combine.setCurrentText(self.DEFAULTS["bg_combine"]); self.bg_combine = self.DEFAULTS["bg_combine"]
        self.sl_bg_lr.setValue(int(round(self.DEFAULTS["bg_lr"] * 100))); self.bg_lr = self.DEFAULTS["bg_lr"]; self.lbl_bg_lr_val.setText(f"{self.bg_lr:.3f}")
        self.sl_move.setValue(int(self.DEFAULTS["move_thresh"])); self.val_move.setText(str(int(self.DEFAULTS["move_thresh"])))
        self.move_thresh = int(self.DEFAULTS["move_thresh"])

        # tracking defaults
        self.cb_record.setChecked(False)
        self.cb_per_id.setChecked(True)
        self.cb_combined.setChecked(False)
        self.cb_only_moving.setChecked(True)
        self.sl_static_win.setValue(12); self.val_static_win.setText("12")
        self.sl_min_motion.setValue(2); self.val_min_motion.setText("2")

        self.status_inline.setText("Parameters reset to defaults.")
        self.warmup_progress.setVisible(False)
        if self.enable_plots: self._init_plots()

    def _update_morph(self, v):
        if v % 2 == 0: v += 1
        self.morph_ks = int(v)
        self.sl_morph.blockSignals(True); self.sl_morph.setValue(v); self.sl_morph.blockSignals(False)
        self.val_morph.setText(str(v))

    def _bg_mode_changed(self, text):
        self.bg_mode = text
        self._ensure_bg_subtractor()

    def _bg_combine_changed(self, text):
        self.bg_combine = text

    def _bg_lr_changed(self, slider_val):
        self.bg_lr = float(slider_val) / 100.0
        self.lbl_bg_lr_val.setText(f"{self.bg_lr:.3f}")

    def _edit_bg_lr_value(self):
        val, ok = QtWidgets.QInputDialog.getDouble(self, "Set BG learning rate", "Enter value (0.000 — 1.000):",
                                                   self.bg_lr, 0.0, 1.0, 3)
        if ok:
            self.bg_lr = float(val); self.lbl_bg_lr_val.setText(f"{self.bg_lr:.3f}")
            self.sl_bg_lr.blockSignals(True); self.sl_bg_lr.setValue(int(round(self.bg_lr*100))); self.sl_bg_lr.blockSignals(False)

    def _toggle_plots(self, on: bool):
        self.enable_plots = bool(on) and PG_AVAILABLE
        self.metrics_box.setVisible(self.enable_plots)
        if not PG_AVAILABLE:
            self.status.setText("pyqtgraph not installed — metrics disabled.")
            return
        if self.enable_plots:
            self._init_plots()
            self._plot_refresh_enabled = True
            self._plot_timer.start()
            self._update_plots_timer_tick()
        else:
            self._plot_timer.stop()
            self._plot_refresh_enabled = False
            self._init_plots()

    def _ensure_bg_subtractor(self):
        if self.bg_mode == "KNN":
            self.bg_subtractor = cv.createBackgroundSubtractorKNN(history=300, dist2Threshold=400.0, detectShadows=True)
        elif self.bg_mode == "MOG2":
            self.bg_subtractor = cv.createBackgroundSubtractorMOG2(history=300, varThreshold=16.0, detectShadows=True)
        else:
            self.bg_subtractor = None

    # ---------- NEW: tracking helpers ----------
    def _apply_motion_filter(self, *_):
        try:
            static_win = max(2, int(self.sl_static_win.value()))
            min_motion = float(self.sl_min_motion.value())
            only_moving = bool(self.cb_only_moving.isChecked())
            self.tracker.set_motion_filter(static_window=static_win, min_motion_px=min_motion, only_moving=only_moving)
        except Exception:
            pass

    def _toggle_recording(self, on: bool):
        try:
            self.tracker.set_recording(bool(on))
            if on:
                self.status.setText("Recording CSV: ON (see ./results/)")
            else:
                self.status.setText("Recording CSV: OFF")
        except Exception as e:
            self.status.setText(f"Recording toggle failed: {e}")

    # ---------- plotting helpers (kept) ----------
    def _color_for_track(self, tid):
        if tid in self.color_cache:
            return self.color_cache[tid]
        c = pg.intColor((tid*3) % 64, hues=64) if PG_AVAILABLE else QtGui.QColor(0,255,0)
        pen = pg.mkPen(c, width=2) if PG_AVAILABLE else None
        self.color_cache[tid] = pen
        return pen

    def _ensure_track(self, tid):
        if tid in self.tracks or not PG_AVAILABLE:
            return
        pen = self._color_for_track(tid)
        curve_a = self.plot_area.plot([], [], pen=pen, name=f"id{tid}",
                                      symbol='o', symbolSize=5, symbolBrush=pen.color())
        curve_w = self.plot_w.plot([], [], pen=pen, name=f"id{tid}",
                                   symbol='o', symbolSize=5, symbolBrush=pen.color())
        curve_h = self.plot_h.plot([], [], pen=pen, name=f"id{tid}",
                                   symbol='o', symbolSize=5, symbolBrush=pen.color())
        self.tracks[tid] = dict(
            last_pt=None,
            t=deque(maxlen=5000),
            area=deque(maxlen=5000),
            w=deque(maxlen=5000),
            h=deque(maxlen=5000),
            curve_area=curve_a, curve_w=curve_w, curve_h=curve_h
        )

    def _refresh_track_curves(self, tr):
        if not PG_AVAILABLE:
            return
        tr["curve_area"].setData(list(tr["t"]), list(tr["area"]), connect='finite')
        tr["curve_w"].setData(list(tr["t"]), list(tr["w"]), connect='finite')
        tr["curve_h"].setData(list(tr["t"]), list(tr["h"]), connect='finite')

    def _append_track_point(self, tid, d):
        tr = self.tracks.get(tid)
        if not tr:
            return
        tr["last_pt"] = (d["cx"], d["cy"])
        tr["t"].append(self.frame_index)
        tr["area"].append(d["area"])
        tr["w"].append(d["w"])
        tr["h"].append(d["h"])
        self._refresh_track_curves(tr)
        if len(tr["t"]) == 1 and PG_AVAILABLE and self.enable_plots:
            try:
                for pw in (self.plot_area, self.plot_w, self.plot_h):
                    pw.enableAutoRange('xy', True)
                    pw.autoRange()
            except Exception:
                pass

    def _update_plots_timer_tick(self):
        if not (PG_AVAILABLE and self.enable_plots and self._plot_refresh_enabled):
            return
        try:
            for _, tr in self.tracks.items():
                self._refresh_track_curves(tr)
        except Exception:
            pass

    # pause during resize
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        if PG_AVAILABLE and self.enable_plots:
            self._plot_refresh_enabled = False
            self._resize_debounce.start()
        return super().resizeEvent(event)

    def _resume_plot_refresh(self):
        if PG_AVAILABLE and self.enable_plots:
            self._plot_refresh_enabled = True
    
    def _update_ui_labels(self):
        if self._pending_ui_data is None:
            return
        data = self._pending_ui_data
        self._pending_ui_data = None
        warmup_info = f" | warm-up: {data['warmup_progress']}%" if data['warmup_active'] else ""
        ids_info = f" | active IDs: {data.get('active_ids','-')}"
        self.info_label.setText(
            f"frame: {data['frame_index']} / {self.total_frames} | detections: {data['det_count']}{warmup_info}{ids_info}"
        )
        if data['warmup_active']:
            self.status_inline.setText("Running… (warm-up actif - filtres anti-parasites)")
        else:
            self.status_inline.setText("Running…")
        if self.cb_enable_warmup.isChecked():
            if data['warmup_active']:
                progress_value = data['warmup_progress_value']
                self.warmup_progress.setValue(progress_value)
                self.warmup_progress.setFormat(f"Warm-up: {progress_value}/{self.warmup_frames} frames")
            elif not data['warmup_finished_notified']:
                self.warmup_progress.setFormat("Warm-up: Terminé ✓")
                QtCore.QTimer.singleShot(2000, lambda: self.warmup_progress.setVisible(False))
                data['warmup_finished_notified'] = True

    # ---------- per frame ----------
    def on_frame(self, frame_bgr):
        if self.processing_busy: return
        self.processing_busy = True
        try:
            self.frame_index += 1
            if self.fast_downscale:
                h, w = frame_bgr.shape[:2]
                frame_bgr = cv.resize(frame_bgr, (int(w*0.75), int(h*0.75)), interpolation=cv.INTER_AREA)

            results = detect_larvae_contours(
                frame=frame_bgr,
                gray_threshold=self.gray_threshold,
                contour_area_range=(self.area_min, self.area_max),
                morph_ellipse_size=self.morph_ks,
                back_sub=self.bg_subtractor,
                bs_learning_rate=self.bg_lr,
                combine_mode=self.bg_combine,
                invert_polarity=self.cb_invert.isChecked(),
                enable_warmup=self.cb_enable_warmup.isChecked(),
                warmup_frames=self.warmup_frames,
            )
            
            contours = results.get("contours", [])
            det_count = len(contours)
            warmup_active = results.get("warmup_active", False)
            warmup_progress = results.get("warmup_progress", 1.0)

            # ---- NEW: feed detections to MultiTracker (IDs + CSV) ----
            try:
                self.tracker.update(self.frame_index, contours, frame_bgr.shape[:2])  # IDs, CSV recording handled internally
                active_ids = self.tracker.active_ids(only_moving=self.cb_only_moving.isChecked())
            except Exception:
                active_ids = []

            # Prepare UI update data (throttled)
            progress_value = int(warmup_progress * self.warmup_frames)
            self._pending_ui_data = {
                'frame_index': self.frame_index,
                'det_count': det_count,
                'warmup_active': warmup_active,
                'warmup_progress': int(warmup_progress * 100),
                'warmup_progress_value': progress_value,
                'warmup_finished_notified': self._warmup_finished_notified,
                'active_ids': len(active_ids),
            }
            if not self._ui_update_timer.isActive():
                self._ui_update_timer.start()

            if self.cb_enable_warmup.isChecked() and warmup_active:
                if not self.warmup_progress.isVisible():
                    self.warmup_progress.setVisible(True)

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

            # Visual warm-up ribbon
            if warmup_active and self.cb_enable_warmup.isChecked():
                h, w = contoured.shape[:2]
                overlay = contoured.copy()
                cv.rectangle(overlay, (10, 10), (w-10, 60), (0, 165, 255), -1)
                cv.addWeighted(contoured, 0.7, overlay, 0.3, 0, contoured)
                text = f"WARM-UP ACTIF - {int(warmup_progress*100)}% ({progress_value}/{self.warmup_frames})"
                cv.putText(contoured, text, (20, 35), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv.putText(contoured, "Filtres anti-parasites actifs", (20, 55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # ---- NEW: draw IDs on the contoured view (only current frame detections if desired) ----
            try:
                contoured = self.tracker.draw_ids(
                    contoured, only_current=True, frame_idx=self.frame_index,
                    only_moving=self.cb_only_moving.isChecked()
                )
            except Exception:
                pass

            bin_vis = results["binary_image"]; morph_vis = results["morphed_image"]
            if bin_vis.ndim==2:   bin_vis = cv.cvtColor(bin_vis, cv.COLOR_GRAY2BGR)
            if morph_vis.ndim==2: morph_vis = cv.cvtColor(morph_vis, cv.COLOR_GRAY2BGR)
            
            def _resize(img):
                h,w = img.shape[:2]; s = min(1.0, self.max_display_width/float(w))
                return cv.resize(img, (int(w*s), int(h*s)), interpolation=cv.INTER_AREA)
            
            for img, slot in ((_resize(bin_vis), self.lbl_bin["label"]),
                              (_resize(morph_vis), self.lbl_morph["label"]),
                              (_resize(contoured), self.lbl_cont["label"])):
                slot.setPixmap(qimage_to_pixmap(cv_to_qimage(img)))

            # Plotting (unchanged; only when warm-up finished)
            if self.enable_plots and PG_AVAILABLE and not warmup_active:
                dets = []
                for c in contours:
                    (cx, cy), (w, h), a = contour_centroid_w_h_area(c)
                    dets.append(dict(cx=cx, cy=cy, w=w, h=h, area=a))
                # simple assignment into local plotter (separate from MultiTracker)
                self._assign_tracks_to_plots(dets)
                
        finally:
            self.processing_busy = False

    # small adapter: keep your plotting workflow
    def _assign_tracks_to_plots(self, detections):
        if not PG_AVAILABLE:
            return
        if not detections:
            return
        active_ids = [tid for tid, tr in self.tracks.items() if tr.get("last_pt") is not None]
        track_pts  = {tid: np.array(self.tracks[tid]["last_pt"], dtype=float) for tid in active_ids}

        used_tracks = set()
        used_dets   = set()

        det_coords = [np.array([d["cx"], d["cy"]], dtype=float) for d in detections]
        det_order = list(range(len(det_coords)))
        if active_ids:
            nearest = []
            for j, dc in enumerate(det_coords):
                dists = [np.linalg.norm(dc - track_pts[tid]) for tid in active_ids]
                nearest.append((j, min(dists) if dists else np.inf))
            det_order = [j for j,_ in sorted(nearest, key=lambda t: t[1])]

        for j in det_order:
            if j in used_dets: continue
            dc = det_coords[j]
            best_tid, best_dist = None, np.inf
            for tid in active_ids:
                if tid in used_tracks: continue
                dist = float(np.linalg.norm(dc - track_pts[tid]))
                if dist < best_dist:
                    best_tid, best_dist = tid, dist
            if best_tid is not None and best_dist <= self.MATCH_MAX_DIST:
                self._append_track_point(best_tid, detections[j])
                used_tracks.add(best_tid); used_dets.add(j)
            else:
                if len(self.tracks) < self.MAX_TRACKS:
                    tid = self.next_track_id; self.next_track_id += 1
                    self._ensure_track(tid)
                    self._append_track_point(tid, detections[j])
                    used_tracks.add(tid); used_dets.add(j)

        for j, d in enumerate(detections):
            if j in used_dets: continue
            if len(self.tracks) < self.MAX_TRACKS:
                tid = self.next_track_id; self.next_track_id += 1
                self._ensure_track(tid)
                self._append_track_point(tid, d)

    def on_video_end(self):
        self.status_inline.setText("End of stream.")
        self.warmup_progress.setVisible(False)
        try: self.tracker.close_all()
        except: pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.aboutToQuit.connect(lambda: (w.reader.stop(), w.reader.wait(300), w.tracker.close_all()))
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
