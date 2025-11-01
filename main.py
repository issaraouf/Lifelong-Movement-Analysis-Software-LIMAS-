# -*- coding: utf-8 -*-
"""
================================================================================
Embryo / Larva / Adult Multi-Tracker — PySide6 UI (VERBOSE edition)
================================================================================

But
----
Application GUI (Qt) destinée à des utilisateurs non-ingénieurs pour analyser des
vidéos (embryons, larves, mouches adultes). L'UI expose des paramètres simples
et des "presets" qui appliquent des réglages cohérents pour 4 scénarios :

    • Embryo
    • Larve
    • Adulte (simple)
    • Multi-adultes

Pipeline (vue d'ensemble)
-------------------------
[Lecture vidéo]  -->  [Détection + Segmentation]  -->  [Assignation ID + Suivi]  -->  [CSV/Overlay]

  1) Lecture vidéo (thread séparé).
  2) Détection (contour.py) + BG (KNN/MOG2/None) + auto-freeze possible.
  3) Tracking (tracker.py) Greedy/Hungarian/Auto + hystérésis "moving".
  4) Overlay (contours + IDs) et **enregistrement MP4 live**.

Remarques
---------
• Version "verbeuse" (commentaires pédagogiques).
• L'UI appelle `contour.py` et `tracker.py` sans modifier leurs logiques.

Auteur : équipe R&D — 2025
"""

import os, sys, datetime
from typing import Optional, Dict, Any

import cv2 as cv
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# Détection + dessin (boîtes noires : on utilise l'API existante)
from contour import (
    detect_larvae_contours,
    fit_contours,
    reset_warmup,
    _draw_rectangle, _draw_min_area_rect, _draw_convex_hull, _draw_ellipse
)

# Tracking (boîte noire)
from tracker import MultiTracker, TrackParams

# Graphiques (optionnels)
try:
    import pyqtgraph as pg
    PG_AVAILABLE = True
except Exception:
    PG_AVAILABLE = False


# =============================================================================
# Utilitaires d’affichage (conversion OpenCV -> Qt)
# =============================================================================

def cv_to_qimage(img_bgr_or_gray: np.ndarray) -> QtGui.QImage:
    """Convertit une image OpenCV (BGR/GRAY) en QImage RGB888."""
    if img_bgr_or_gray is None:
        return QtGui.QImage()
    if img_bgr_or_gray.ndim == 2:
        img_rgb = cv.cvtColor(img_bgr_or_gray, cv.COLOR_GRAY2RGB)
    else:
        img_rgb = cv.cvtColor(img_bgr_or_gray, cv.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    return QtGui.QImage(img_rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)


class ClickableLabel(QtWidgets.QLabel):
    """QLabel cliquable pour saisir une valeur numérique exacte liée à un slider."""
    clicked = QtCore.Signal()
    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)


# =============================================================================
# Thread lecteur vidéo (séparé de l’UI)
# =============================================================================

class VideoReader(QtCore.QThread):
    """
    Lecture vidéo non bloquante via OpenCV.

    Signaux :
      • frameReady(np.ndarray BGR) : frame prête
      • ended()                    : fin de flux
    """
    frameReady = QtCore.Signal(object)
    ended = QtCore.Signal()

    def __init__(self, path: Optional[str] = None, parent=None, target_fps: int = 20):
        super().__init__(parent)
        self.path = path
        self._cap = None
        self._running = False
        self._paused = False
        self.target_fps = max(1, int(target_fps))
        self._pending_seek: Optional[int] = None

    # API contrôles depuis l’UI
    def open(self, path: str):         self.path = path
    def set_fps(self, fps: int):       self.target_fps = max(1, int(fps))
    def seek_to(self, frame_idx: int): self._pending_seek = max(0, int(frame_idx))
    def pause(self, p: bool = True):   self._paused = bool(p)
    def stop(self):                    self._running = False

    def run(self):
        if not self.path:
            self.ended.emit(); return

        cap = cv.VideoCapture(self.path)
        if not cap.isOpened():
            self.ended.emit(); return
        self._cap = cap
        self._running, self._paused = True, False

        while self._running:
            if self._paused and self._pending_seek is None:
                self.msleep(5); continue

            if self._pending_seek is not None:
                try:
                    cap.set(cv.CAP_PROP_POS_FRAMES, int(self._pending_seek))
                except Exception:
                    pass
                self._pending_seek = None

            ok, frame = cap.read()
            if not ok:
                self.ended.emit()
                break

            self.frameReady.emit(frame)

            if self._paused:
                self.msleep(5); continue
            self.msleep(int(1000 / self.target_fps))

        try:
            cap.release()
        except Exception:
            pass


# =============================================================================
# Fenêtre principale
# =============================================================================

class MainWindow(QtWidgets.QMainWindow):
    """Assemble le pipeline et fournit l’interface utilisateur."""

    DEFAULTS = dict(
        fps=20, gray_threshold=50.0,
        area_min=200.0, area_max=30000.0,
        morph_ks=23, fit_shape="Rectangle",
        fast_downscale=False,
        bg_mode="None", bg_combine="OR (Safe)", bg_lr=0.01,
        move_thresh=5,
        warmup_frames=30
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Embryo/Larva/Adult Tracker — Qt UI (verbose)")
        self.resize(1500, 920)

        # État runtime
        self.video_path: str = ""
        self.frame_index: int = 0
        self.total_frames: int = 0
        self.max_display_width: int = 540

        # Miroirs contrôles
        self.gray_threshold = float(self.DEFAULTS["gray_threshold"])
        self.area_min = float(self.DEFAULTS["area_min"])
        self.area_max = float(self.DEFAULTS["area_max"])
        self.morph_ks = int(self.DEFAULTS["morph_ks"])
        self.fit_shape = str(self.DEFAULTS["fit_shape"])
        self.fast_downscale = bool(self.DEFAULTS["fast_downscale"])
        self.bg_mode = str(self.DEFAULTS["bg_mode"])
        self.bg_combine = str(self.DEFAULTS["bg_combine"])
        self.bg_lr = float(self.DEFAULTS["bg_lr"])
        self.move_thresh = int(self.DEFAULTS["move_thresh"])
        self.warmup_frames = int(self.DEFAULTS["warmup_frames"])
        self.bg_subtractor = None

        # Tracker
        self.tracker = MultiTracker(TrackParams(
            max_dist=320.0, area_tolerance=0.60,
            ema_pos=0.40, ema_area=0.40, vel_ema=0.60,
            snap_limit_px=60.0, jump_threshold_px=120.0,
            max_miss=30, min_age_csv=12
        ))

        # Overlay recording (MP4)
        self.ov_writer = None
        self.ov_path = ""
        self.ov_size = None
        self.ov_fps = None

        # UI + thème + graphes
        self._build_ui()
        self._apply_dark_theme()
        self._init_plots()

        # Thread lecteur
        self.reader = VideoReader()
        self.reader.frameReady.connect(self.on_frame)
        self.reader.ended.connect(self.on_video_end)

        # UI de statut (dé-bounce)
        self._pending_ui_data: Optional[Dict[str, Any]] = None
        self._ui_update_timer = QtCore.QTimer(self)
        self._ui_update_timer.setSingleShot(True)
        self._ui_update_timer.setInterval(100)
        self._ui_update_timer.timeout.connect(self._update_ui_labels)

        self.setAcceptDrops(True)
        self._install_shortcuts()

    # ------------------------------ Thème sombre (cases bien visibles)
    def _apply_dark_theme(self):
        app = QtWidgets.QApplication.instance()
        app.setStyle("Fusion")

        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Window,        QtGui.QColor(30, 31, 34))
        pal.setColor(QtGui.QPalette.Base,          QtGui.QColor(20, 21, 23))
        pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(35, 36, 40))
        pal.setColor(QtGui.QPalette.Text,          QtGui.QColor(232, 232, 232))
        pal.setColor(QtGui.QPalette.WindowText,    QtGui.QColor(232, 232, 232))
        pal.setColor(QtGui.QPalette.Button,        QtGui.QColor(42, 45, 50))
        pal.setColor(QtGui.QPalette.ButtonText,    QtGui.QColor(232, 232, 232))
        pal.setColor(QtGui.QPalette.Highlight,     QtGui.QColor(74, 144, 226))
        app.setPalette(pal)

        self.setStyleSheet("""
            * { color:#E8E8E8; }
            QMainWindow { background:#1a1b1e; }
            QLabel { color:#E8E8E8; }

            QGroupBox {
                color:#E8E8E8;
                border:1px solid #444; border-radius:4px; margin-top:8px;
            }
            QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 4px; }

            QPushButton {
                background:#2a2d32; border:1px solid #555; padding:6px 10px; border-radius:4px;
            }
            QPushButton:hover { background:#33373d; }

            QCheckBox::indicator {
                width:18px; height:18px; border:1px solid #666; border-radius:3px;
                background:#222; margin-right:6px;
            }
            QCheckBox::indicator:checked { background:#4a90e2; border:1px solid #9cc3ff; }

            QSlider::groove:horizontal { height:6px; background:#333; margin:0 6px; border-radius:3px; }
            QSlider::handle:horizontal { background:#b0b0b0; width:14px; border-radius:7px; margin:-5px 0; }
            QSlider::sub-page:horizontal { background:#4a90e2; }

            QComboBox { background:#2d2d2d; border:1px solid #555; padding:4px 6px; }
            QComboBox QAbstractItemView { background:#2d2d2d; border:1px solid #555; }
            QScrollArea, QScrollArea > QWidget { background:#1a1b1e; border:0; }
        """)

    # ------------------------------ petits helpers UI
    def _label(self, txt: str) -> QtWidgets.QLabel:
        lab = QtWidgets.QLabel(txt)
        lab.setStyleSheet("margin-top:8px;")
        return lab

    def _pane(self, title: str):
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
        val_lbl.setMinimumWidth(72)
        val_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        header.addWidget(lab); header.addStretch(1); header.addWidget(val_lbl)
        parent_layout.addLayout(header)

        s = QtWidgets.QSlider(QtCore.Qt.Horizontal); s.setRange(mn, mx); s.setValue(val)
        lab.setToolTip(tooltip); s.setToolTip(tooltip)
        val_lbl.setToolTip("Clique pour entrer une valeur exacte.")
        parent_layout.addWidget(s)
        return s, val_lbl

    def _install_value_editor(self, value_label: ClickableLabel, slider: QtWidgets.QSlider, *, morph_odd: bool=False):
        def on_click():
            mn, mx = slider.minimum(), slider.maximum()
            val, ok = QtWidgets.QInputDialog.getInt(self, "Set value",
                                                    f"Enter a value [{mn}..{mx}]:",
                                                    slider.value(), mn, mx, 1)
            if ok:
                if morph_odd and (val % 2 == 0):
                    val += 1
                slider.setValue(val)
                value_label.setText(str(val))
        value_label.clicked.connect(on_click)

    # ------------------------------ construction de l’UI
    def _build_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        H = QtWidgets.QHBoxLayout(central)

        # Panneau gauche (scrollable)
        left = QtWidgets.QWidget()
        C = QtWidgets.QVBoxLayout(left)
        C.setContentsMargins(8,8,8,8); C.setSpacing(6)

        # Commandes globales
        row = QtWidgets.QHBoxLayout()
        self.btn_open  = QtWidgets.QPushButton("Open")
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_pause = QtWidgets.QPushButton("Pause"); self.btn_pause.setCheckable(True)
        self.btn_stop  = QtWidgets.QPushButton("Stop")
        self.btn_rec   = QtWidgets.QPushButton("Save overlay (MP4)"); self.btn_rec.setCheckable(True)
        row.addWidget(self.btn_open); row.addWidget(self.btn_start)
        row.addWidget(self.btn_pause); row.addWidget(self.btn_stop); row.addWidget(self.btn_rec)
        C.addLayout(row)

        self.btn_reset = QtWidgets.QPushButton("Reset parameters")
        C.addWidget(self.btn_reset)

        """ # Presets
        presets_box = QtWidgets.QGroupBox("Presets")
        PL = QtWidgets.QGridLayout(presets_box)
        btn_emb = QtWidgets.QPushButton("Embryo")
        btn_lar = QtWidgets.QPushButton("Larve")
        btn_as  = QtWidgets.QPushButton("Adulte (simple)")
        btn_am  = QtWidgets.QPushButton("Multi-adultes")
        PL.addWidget(btn_emb,0,0); PL.addWidget(btn_lar,0,1)
        PL.addWidget(btn_as,1,0);  PL.addWidget(btn_am,1,1)
        C.addWidget(presets_box)
        btn_emb.clicked.connect(lambda: self.apply_preset("embryo"))
        btn_lar.clicked.connect(lambda: self.apply_preset("larva"))
        btn_as.clicked.connect(lambda: self.apply_preset("adult_single"))
        btn_am.clicked.connect(lambda: self.apply_preset("adult_multi")) """

        # Warm-up
        warm = QtWidgets.QGroupBox("Anti-parasites (warm-up)")
        WL = QtWidgets.QVBoxLayout(warm)
        self.cb_enable_warmup = QtWidgets.QCheckBox("Activer le warm-up anti-parasites")
        self.cb_enable_warmup.setChecked(True)
        WL.addWidget(self.cb_enable_warmup)
        self.sl_warmup, self.val_warmup = self._labeled_slider(
            WL, "Durée warm-up (frames)", 10, 100, 30,
            "Pendant le warm-up, les filtres s'ajustent. Le BG peut apprendre avant gel."
        )
        self._install_value_editor(self.val_warmup, self.sl_warmup)
        self.warmup_progress = QtWidgets.QProgressBar()
        self.warmup_progress.setFormat("Warm-up: %p%"); self.warmup_progress.setVisible(False)
        WL.addWidget(self.warmup_progress)
        C.addWidget(warm)

        # Sliders de base
        self.sl_fps,    self.val_fps    = self._labeled_slider(C, "Reader FPS", 5, 60, 20)
        self.sl_thresh, self.val_thresh = self._labeled_slider(C, "Gray threshold", 0, 255, 50)
        self.sl_area_min, self.val_area_min = self._labeled_slider(C, "Area min", 0, 200000, 200)
        self.sl_area_max, self.val_area_max = self._labeled_slider(C, "Area max", 0, 300000, 30000)
        self.sl_morph, self.val_morph = self._labeled_slider(C, "Morph ellipse (odd)", 1, 61, 23)
        self.sl_morph.setSingleStep(2); self.sl_morph.setPageStep(2)

        self.cb_invert = QtWidgets.QCheckBox("Invert polarity (dark objects)")
        C.addWidget(self.cb_invert)

        C.addWidget(self._label("Fit shape"))
        self.cb_shape = QtWidgets.QComboBox()
        self.cb_shape.addItems(["Rectangle","Min Area Rectangle","Convex Hull","Ellipse"])
        self.cb_shape.setCurrentText("Rectangle")
        C.addWidget(self.cb_shape)

        self.cb_fast = QtWidgets.QCheckBox("Process at 0.75× scale (perf)")
        C.addWidget(self.cb_fast)

        self.sl_move, self.val_move = self._labeled_slider(
            C, "Moving threshold (px/s)", 0, 200, 5,
            "Seuil pour l'état 'moving' (export/affichage)."
        )

        # Background & Mask (+ auto-freeze + assignment)
        bg_box = QtWidgets.QGroupBox("Background & Mask")
        BGL = QtWidgets.QFormLayout(bg_box)

        self.cb_bg = QtWidgets.QComboBox(); self.cb_bg.addItems(["None", "KNN", "MOG2"]); self.cb_bg.setCurrentText("None")
        BGL.addRow("Background subtraction", self.cb_bg)

        self.cb_combine = QtWidgets.QComboBox()
        self.cb_combine.addItems(["OR (Safe)", "AND (Strict)", "Replace (BG only)"])
        self.cb_combine.setCurrentText("OR (Safe)")
        BGL.addRow("Mask combine mode", self.cb_combine)

        lr_row = QtWidgets.QHBoxLayout()
        self.sl_bg_lr = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sl_bg_lr.setRange(0, 100)
        self.sl_bg_lr.setValue(int(round(0.01 * 100)))
        self.lbl_bg_lr_val = ClickableLabel(f"{0.01:.3f}")
        self.lbl_bg_lr_val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter); self.lbl_bg_lr_val.setMinimumWidth(60)
        lr_row.addWidget(self.sl_bg_lr, 1); lr_row.addWidget(self.lbl_bg_lr_val)
        BGL.addRow("BG learning rate", lr_row)

        self.cb_auto_freeze_bg = QtWidgets.QCheckBox("Auto-freeze BG after warm-up")
        self.cb_auto_freeze_bg.setToolTip("Fige le LR ~0 après warm-up pour ne pas avaler un sujet immobile.")
        BGL.addRow(self.cb_auto_freeze_bg)

        self.cb_assign = QtWidgets.QComboBox()
        self.cb_assign.addItems(["Greedy (fast)", "Hungarian (stable)", "Auto (smart)"])
        self.cb_assign.setCurrentText("Greedy (fast)")
        BGL.addRow("Assignment", self.cb_assign)

        C.addWidget(bg_box)

        # Tracking & CSV
        track_box = QtWidgets.QGroupBox("Tracking & CSV (IDs)")
        TL = QtWidgets.QVBoxLayout(track_box)
        self.cb_record = QtWidgets.QCheckBox("Record CSV"); TL.addWidget(self.cb_record)

        row_csv = QtWidgets.QHBoxLayout()
        self.cb_per_id   = QtWidgets.QCheckBox("Per-ID CSV"); self.cb_per_id.setChecked(True)
        self.cb_combined = QtWidgets.QCheckBox("Combined CSV")
        row_csv.addWidget(self.cb_per_id); row_csv.addWidget(self.cb_combined); row_csv.addStretch(1)
        TL.addLayout(row_csv)

        self.cb_only_moving = QtWidgets.QCheckBox("Display and save only moving IDs")
        TL.addWidget(self.cb_only_moving)

        self.cb_single = QtWidgets.QCheckBox("Un seul individu (forcer ID=1)")
        TL.addWidget(self.cb_single)

        self.cb_req_motion = QtWidgets.QCheckBox("Créer un ID seulement si mouvement")
        TL.addWidget(self.cb_req_motion)

        self.sl_static_win, self.val_static_win = self._labeled_slider(
            TL, "Motion window (frames)", 2, 60, 12,
            "Fenêtre glissante pour 'moving/not moving'."
        )
        self.sl_min_motion, self.val_min_motion = self._labeled_slider(
            TL, "Min motion (px/step)", 0, 20, 2,
            "Pas moyen minimal pour être considéré 'moving'."
        )
        C.addWidget(track_box)

        # Infos
        self.info_label = QtWidgets.QLabel("frame: 0 / 0 | detections: 0")
        self.status_inline = QtWidgets.QLabel("")
        C.addWidget(self.info_label); C.addWidget(self.status_inline)
        C.addStretch(1)

        # Scroll container
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(left); scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scroll.setMinimumWidth(480)
        H.addWidget(scroll, 0)

        # Panneau droit (vignettes + seek + plots)
        V = QtWidgets.QVBoxLayout()
        thumbs = QtWidgets.QHBoxLayout()
        self.lbl_bin   = self._pane("Binary")
        self.lbl_morph = self._pane("Morphed")
        self.lbl_cont  = self._pane("Contoured + IDs")
        thumbs.addWidget(self.lbl_bin["card"]); thumbs.addWidget(self.lbl_morph["card"]); thumbs.addWidget(self.lbl_cont["card"])
        V.addLayout(thumbs)

        seek_row = QtWidgets.QHBoxLayout()
        seek_row.addWidget(QtWidgets.QLabel("Frame"))
        self.seek = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.seek.setRange(0, 0); self.seek.setSingleStep(1); self.seek.setPageStep(30)
        seek_row.addWidget(self.seek, 1)
        V.addLayout(seek_row)

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
            M.addWidget(self.plot_area,1); M.addWidget(self.plot_w,1); M.addWidget(self.plot_h,1)
        else:
            M.addWidget(QtWidgets.QLabel("pyqtgraph not installed — no charts"))
        V.addWidget(self.metrics_box); self.metrics_box.setVisible(False)

        self.chk_plots = QtWidgets.QCheckBox("Show metrics (area / width / height)")
        V.addWidget(self.chk_plots)

        self.status = QtWidgets.QLabel(""); self.status.setStyleSheet("padding:6px;")
        V.addWidget(self.status)
        H.addLayout(V, 1)

        # Connexions
        self.btn_open.clicked.connect(self.pick_video)
        self.btn_start.clicked.connect(self.start)
        self.btn_pause.toggled.connect(self._toggle_pause_button)   # toggle réel
        self.btn_stop.clicked.connect(self.stop)
        self.btn_reset.clicked.connect(self.reset_params)

        # Enregistrement overlay MP4 (live)
        self.btn_rec.toggled.connect(self._toggle_overlay_recording)

        # Sliders -> variables
        self.sl_fps.valueChanged.connect(lambda v: (self.reader.set_fps(v), self.val_fps.setText(str(v))))
        self.sl_thresh.valueChanged.connect(lambda v: (setattr(self, "gray_threshold", float(v)), self.val_thresh.setText(str(v))))
        self.sl_area_min.valueChanged.connect(lambda v: (setattr(self, "area_min", float(v)), self.val_area_min.setText(str(v))))
        self.sl_area_max.valueChanged.connect(lambda v: (setattr(self, "area_max", float(v)), self.val_area_max.setText(str(v))))
        self.sl_morph.valueChanged.connect(self._update_morph)
        self.sl_warmup.valueChanged.connect(lambda v: (setattr(self, "warmup_frames", int(v)), self.val_warmup.setText(str(v))))
        self.cb_shape.currentTextChanged.connect(lambda s: setattr(self, "fit_shape", s))
        self.cb_fast.toggled.connect(lambda b: setattr(self, "fast_downscale", b))

        # BG & mask
        self.cb_bg.currentTextChanged.connect(self._ensure_bg_subtractor)
        self.cb_combine.currentTextChanged.connect(lambda t: setattr(self, "bg_combine", t))
        self.sl_bg_lr.valueChanged.connect(self._bg_lr_changed)
        self.lbl_bg_lr_val.clicked.connect(self._edit_bg_lr_value)
        self.cb_auto_freeze_bg.toggled.connect(lambda _b: None)

        # Assignment -> tracker
        self.cb_assign.currentIndexChanged.connect(self._on_assign_change)

        # Plot on/off
        self.chk_plots.toggled.connect(self._toggle_plots)

        # Motion filter → tracker
        self.sl_move.valueChanged.connect(lambda v: (setattr(self, "move_thresh", int(v)), self.val_move.setText(str(v))))
        self.cb_record.toggled.connect(self._toggle_recording)
        self.cb_only_moving.toggled.connect(self._apply_motion_filter)
        self.sl_static_win.valueChanged.connect(self._apply_motion_filter)
        self.sl_min_motion.valueChanged.connect(self._apply_motion_filter)
        self.cb_single.toggled.connect(lambda b: self.tracker.set_single_target_mode(b))
        self.cb_req_motion.toggled.connect(lambda b: self.tracker.set_require_motion_to_spawn(b))

        # Saisies directes
        self._install_value_editor(self.val_fps, self.sl_fps)
        self._install_value_editor(self.val_thresh, self.sl_thresh)
        self._install_value_editor(self.val_area_min, self.sl_area_min)
        self._install_value_editor(self.val_area_max, self.sl_area_max)
        self._install_value_editor(self.val_morph, self.sl_morph, morph_odd=True)

        # Seek live (pause pendant le drag, reprise auto si Pause n'est pas ON)
        self.seek.sliderPressed.connect(lambda: self.reader.pause(True))
        def _seek_live():
            pos = int(self.seek.value())
            self.frame_index = pos
            if self.reader.isRunning():
                self.reader.seek_to(pos)
        self.seek.valueChanged.connect(_seek_live)
        def _seek_release():
            if self.reader.isRunning() and not self.btn_pause.isChecked():
                self.reader.pause(False)
                self.status_inline.setText("Running…")
        self.seek.sliderReleased.connect(_seek_release)

    # ------------------------------ plots init
    def _init_plots(self):
        if not PG_AVAILABLE: return
        for pw in (getattr(self, "plot_area", None),
                   getattr(self, "plot_w", None),
                   getattr(self, "plot_h", None)):
            if pw:
                pw.clear(); pw.enableAutoRange('xy', True)

    # ------------------------------ callbacks
    def _on_assign_change(self):
        text = self.cb_assign.currentText()
        method = "greedy" if text.startswith("Greedy") else ("hungarian" if text.startswith("Hungarian") else "auto")
        self.tracker.set_assignment_method(method)
        self.status_inline.setText(f"Assignment: {text}")

    def _toggle_pause_button(self, on: bool):
        self.reader.pause(on)
        self.btn_pause.setText("Resume" if on else "Pause")
        self.status_inline.setText("Paused." if on else "Running…")

    def pick_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open video", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if not path: return
        self._load_video_path(path)

    def _load_video_path(self, path: str):
        self.video_path = path
        self.setWindowTitle(f"Embryo/Larva/Adult Tracker — {os.path.basename(path)}")
        cap = cv.VideoCapture(self.video_path)
        self.total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0) if cap.isOpened() else 0
        try: cap.release()
        except Exception: pass
        self.info_label.setText(f"frame: 0 / {self.total_frames} | detections: 0")
        self.status_inline.setText("Video loaded. Click Start.")
        self.seek.setRange(0, max(0, self.total_frames - 1))
        self.seek.setValue(0)

    def start(self):
        if not self.video_path:
            self.status_inline.setText("Pick a video first."); return

        self.frame_index = 0
        reset_warmup()

        self.warmup_progress.setVisible(self.cb_enable_warmup.isChecked())
        if self.cb_enable_warmup.isChecked():
            self.warmup_progress.setMaximum(self.warmup_frames)
            self.warmup_progress.setValue(0)
            self.warmup_progress.setFormat("Warm-up: 0%")

        self._ensure_bg_subtractor()

        per_id = self.cb_per_id.isChecked()
        combined = self.cb_combined.isChecked()
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{os.path.splitext(os.path.basename(self.video_path))[0]}_{stamp}"
        self.tracker.start_new_video(self.video_path, per_id_csv=per_id,
                                     enable_combined=combined, session_name=session_name)

        self.tracker.set_single_target_mode(self.cb_single.isChecked())
        self.tracker.set_require_motion_to_spawn(self.cb_req_motion.isChecked())
        self._apply_motion_filter()
        self.tracker.set_recording(self.cb_record.isChecked())
        self._on_assign_change()

        self.btn_pause.setChecked(False); self.btn_pause.setText("Pause")

        self.reader.open(self.video_path)
        self.reader.set_fps(self.sl_fps.value())
        if not self.reader.isRunning(): self.reader.start()
        else: self.reader.pause(False)
        self.status_inline.setText("Running…")

    def stop(self):
        if self.reader.isRunning():
            self.reader.stop(); self.reader.wait(500)
        try: self.tracker.close_all()
        except Exception: pass
        self._close_overlay_writer()
        self.btn_rec.setChecked(False)

        self.btn_pause.setChecked(False); self.btn_pause.setText("Pause")
        self.warmup_progress.setVisible(False)
        self.status_inline.setText("Stopped.")

    def reset_params(self):
        self.sl_fps.setValue(int(self.DEFAULTS["fps"]))
        self.sl_thresh.setValue(int(self.DEFAULTS["gray_threshold"]))
        self.sl_area_min.setValue(int(self.DEFAULTS["area_min"]))
        self.sl_area_max.setValue(int(self.DEFAULTS["area_max"]))
        self.sl_morph.setValue(int(self.DEFAULTS["morph_ks"]))
        self.sl_warmup.setValue(int(self.DEFAULTS["warmup_frames"]))
        self.cb_invert.setChecked(False)
        self.cb_shape.setCurrentText(self.DEFAULTS["fit_shape"])
        self.cb_fast.setChecked(self.DEFAULTS["fast_downscale"])

        self.cb_bg.setCurrentText(self.DEFAULTS["bg_mode"]); self.bg_mode = self.DEFAULTS["bg_mode"]; self.bg_subtractor = None
        self.cb_combine.setCurrentText(self.DEFAULTS["bg_combine"]); self.bg_combine = self.DEFAULTS["bg_combine"]
        self.sl_bg_lr.setValue(int(round(self.DEFAULTS["bg_lr"] * 100))); self.bg_lr = self.DEFAULTS["bg_lr"]
        self.lbl_bg_lr_val.setText(f"{self.bg_lr:.3f}")
        self.cb_auto_freeze_bg.setChecked(False)

        self.sl_move.setValue(int(self.DEFAULTS["move_thresh"]))
        self.cb_record.setChecked(False); self.cb_per_id.setChecked(True); self.cb_combined.setChecked(False)
        self.cb_only_moving.setChecked(False); self.cb_single.setChecked(False); self.cb_req_motion.setChecked(False)
        self.cb_assign.setCurrentText("Greedy (fast)")
        self.sl_static_win.setValue(12); self.val_static_win.setText("12")
        self.sl_min_motion.setValue(2); self.val_min_motion.setText("2")
        self._init_plots()
        self.status_inline.setText("Parameters reset to defaults.")

    # ------------------------------ helpers de mise à jour UI
    def _update_morph(self, v: int):
        if v % 2 == 0: v += 1
        self.morph_ks = int(v)
        self.sl_morph.blockSignals(True); self.sl_morph.setValue(v); self.sl_morph.blockSignals(False)
        self.val_morph.setText(str(v))

    def _bg_lr_changed(self, slider_val: int):
        self.bg_lr = float(slider_val) / 100.0
        self.lbl_bg_lr_val.setText(f"{self.bg_lr:.3f}")

    def _edit_bg_lr_value(self):
        val, ok = QtWidgets.QInputDialog.getDouble(
            self, "Set BG learning rate", "Enter value (0.000 — 1.000):",
            self.bg_lr, 0.0, 1.0, 3
        )
        if ok:
            self.bg_lr = float(val); self.lbl_bg_lr_val.setText(f"{self.bg_lr:.3f}")
            self.sl_bg_lr.blockSignals(True)
            self.sl_bg_lr.setValue(int(round(self.bg_lr * 100)))
            self.sl_bg_lr.blockSignals(False)

    def _toggle_plots(self, on: bool):
        self.metrics_box.setVisible(bool(on) and PG_AVAILABLE)
        if on and not PG_AVAILABLE:
            self.status.setText("pyqtgraph not installed — metrics disabled.")

    def _ensure_bg_subtractor(self, *_):
        text = self.cb_bg.currentText() if hasattr(self, "cb_bg") else self.bg_mode
        self.bg_mode = text
        if self.bg_mode == "KNN":
            self.bg_subtractor = cv.createBackgroundSubtractorKNN(
                history=300, dist2Threshold=400.0, detectShadows=True
            )
        elif self.bg_mode == "MOG2":
            self.bg_subtractor = cv.createBackgroundSubtractorMOG2(
                history=300, varThreshold=16.0, detectShadows=True
            )
        else:
            self.bg_subtractor = None

    def _apply_motion_filter(self, *_):
        static_win = max(2, int(self.sl_static_win.value()))
        min_motion = float(self.sl_min_motion.value())
        only_moving = bool(self.cb_only_moving.isChecked())
        self.tracker.set_motion_filter(static_window=static_win,
                                       min_motion_px=min_motion,
                                       only_moving=only_moving)

    def _toggle_recording(self, on: bool):
        self.tracker.set_recording(bool(on))
        self.status.setText("Recording CSV: ON (see ./results/)" if on else "Recording CSV: OFF")

    # ------------------------------ Overlay recording (MP4 live)
    def _toggle_overlay_recording(self, on: bool):
        if on:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save overlay video", "", "MP4 (*.mp4)"
            )
            if not path:
                self.btn_rec.setChecked(False)
                return
            self.ov_path = path if path.lower().endswith(".mp4") else (path + ".mp4")
            self.ov_size = None
            self.ov_fps = float(self.sl_fps.value() or 20)
            self.status.setText(f"Overlay recording → {self.ov_path}")
        else:
            self._close_overlay_writer()
            self.status.setText("Overlay recording: OFF")

    def _open_overlay_writer_if_needed(self, frame_bgr: np.ndarray):
        if self.ov_writer is not None:
            return
        h, w = frame_bgr.shape[:2]
        self.ov_size = (w, h)
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        self.ov_writer = cv.VideoWriter(self.ov_path, fourcc, self.ov_fps, self.ov_size)
        if not self.ov_writer or not self.ov_writer.isOpened():
            self.status.setText("⚠️ Cannot open MP4 writer. Check codecs/permissions.")
            self.btn_rec.setChecked(False)
            self.ov_writer = None

    def _write_overlay_frame(self, frame_bgr: np.ndarray):
        if not self.btn_rec.isChecked():
            return
        if self.ov_path == "":
            return
        if self.ov_writer is None:
            self._open_overlay_writer_if_needed(frame_bgr)
            if self.ov_writer is None:
                return
        h, w = frame_bgr.shape[:2]
        if (w, h) != self.ov_size:
            frame_bgr = cv.resize(frame_bgr, self.ov_size, interpolation=cv.INTER_AREA)
        self.ov_writer.write(frame_bgr)

    def _close_overlay_writer(self):
        try:
            if self.ov_writer is not None:
                self.ov_writer.release()
        except Exception:
            pass
        self.ov_writer = None
        self.ov_path = ""
        self.ov_size = None

    # ------------------------------ Presets
    def apply_preset(self, kind: str):
        PRESETS = {
            "embryo": {
                "ui": dict(
                    fps=10, thresh=50, area_min=100, area_max=8000, morph=13,
                    invert=False, fit="Ellipse", fast=False,
                    bg="None", combine="OR (Safe)", bg_lr=0.01,
                    move_thresh=0, single=False, req_motion=False, only_moving=False,
                    motion_window=18, min_motion=0.5,
                    auto_freeze=False,
                ),
                "trk": dict(
                    max_dist=120, area_tol=0.5, ema_pos=0.35, ema_area=0.4,
                    vel_ema=0.4, snap_limit=20, jump_thr=40,
                    max_miss=80, min_age_csv=6,
                    confirm_px=4.0, confirm_win=8, confirm_to=300
                ),
                "assign": "Auto (smart)",
            },
            "larva": {
                "ui": dict(
                    fps=15, thresh=35, area_min=50, area_max=12000, morph=11,
                    invert=True, fit="Convex Hull", fast=False,
                    bg="KNN", combine="AND (Strict)", bg_lr=0.006,
                    move_thresh=1, single=False, req_motion=False, only_moving=False,
                    motion_window=18, min_motion=1.0,
                    auto_freeze=True,
                ),
                "trk": dict(
                    max_dist=200, area_tol=0.6, ema_pos=0.40, ema_area=0.40,
                    vel_ema=0.5, snap_limit=40, jump_thr=80,
                    max_miss=60, min_age_csv=10,
                    confirm_px=4.0, confirm_win=10, confirm_to=240
                ),
                "assign": "Auto (smart)",
            },
            "adult_single": {
                "ui": dict(
                    fps=20, thresh=40, area_min=20, area_max=3000, morph=7,
                    invert=True, fit="Rectangle", fast=False,
                    bg="None", combine="OR (Safe)", bg_lr=0.01,
                    move_thresh=2, single=True, req_motion=False, only_moving=False,
                    motion_window=10, min_motion=2.0,
                    auto_freeze=False,
                ),
                "trk": dict(
                    max_dist=320, area_tol=0.6, ema_pos=0.5, ema_area=0.4,
                    vel_ema=0.7, snap_limit=70, jump_thr=140,
                    max_miss=30, min_age_csv=8,
                    confirm_px=5.0, confirm_win=8, confirm_to=180
                ),
                "assign": "Greedy (fast)",
            },
            "adult_multi": {
                "ui": dict(
                    fps=20, thresh=35, area_min=20, area_max=3000, morph=7,
                    invert=True, fit="Rectangle", fast=False,
                    bg="KNN", combine="AND (Strict)", bg_lr=0.005,
                    move_thresh=2, single=False, req_motion=True, only_moving=True,
                    motion_window=12, min_motion=2.0,
                    auto_freeze=True,
                ),
                "trk": dict(
                    max_dist=320, area_tol=0.6, ema_pos=0.4, ema_area=0.4,
                    vel_ema=0.65, snap_limit=60, jump_thr=120,
                    max_miss=30, min_age_csv=12,
                    confirm_px=5.0, confirm_win=8, confirm_to=180
                ),
                "assign": "Hungarian (stable)",
            },
        }

        PDEF = PRESETS.get(kind, PRESETS["adult_single"])
        U, T = PDEF["ui"], PDEF["trk"]

        # UI
        self.sl_fps.setValue(int(U["fps"]))
        self.sl_thresh.setValue(int(U["thresh"]))
        self.sl_area_min.setValue(int(U["area_min"]))
        self.sl_area_max.setValue(int(U["area_max"]))
        morph = int(U["morph"]); self.sl_morph.setValue(morph if morph % 2 else morph + 1)
        self.cb_invert.setChecked(bool(U["invert"]))
        self.cb_shape.setCurrentText(U["fit"])
        self.cb_fast.setChecked(bool(U["fast"]))
        self.cb_bg.setCurrentText(U["bg"])
        self.cb_combine.setCurrentText(U["combine"])
        self.sl_bg_lr.setValue(int(round(float(U["bg_lr"]) * 100)))
        self.cb_auto_freeze_bg.setChecked(bool(U["auto_freeze"]))
        self.sl_move.setValue(int(U["move_thresh"]))
        self.cb_single.setChecked(bool(U["single"]))
        self.cb_req_motion.setChecked(bool(U["req_motion"]))
        self.cb_only_moving.setChecked(bool(U["only_moving"]))
        self.sl_static_win.setValue(int(U["motion_window"]))
        self.sl_min_motion.setValue(int(U["min_motion"]))
        self.cb_assign.setCurrentText(PDEF.get("assign", "Greedy (fast)"))
        self._on_assign_change()

        # Tracker
        P = self.tracker.params
        P.max_dist, P.area_tolerance = float(T["max_dist"]), float(T["area_tol"])
        P.ema_pos, P.ema_area, P.vel_ema = float(T["ema_pos"]), float(T["ema_area"]), float(T["vel_ema"])
        P.snap_limit_px, P.jump_threshold_px = float(T["snap_limit"]), float(T["jump_thr"])
        P.max_miss, P.min_age_csv = int(T["max_miss"]), int(T["min_age_csv"])
        self.tracker.confirm_motion_px = float(T["confirm_px"])
        self.tracker.confirm_window    = int(T["confirm_win"])
        self.tracker.confirm_timeout   = int(T["confirm_to"])
        self._apply_motion_filter()

        names = {"embryo":"Preset: Embryo","larva":"Preset: Larva",
                 "adult_single":"Preset: Adult (single)","adult_multi":"Preset: Multi-adults"}
        self.status_inline.setText(names.get(kind, "Preset applied"))

    # ------------------------------ Vidéo I/O
    def on_video_end(self):
        self.stop()

    def on_frame(self, frame_bgr: np.ndarray):
        self.frame_index += 1

        # Détection
        results = detect_larvae_contours(
            frame=frame_bgr,
            gray_threshold=self.gray_threshold,
            contour_area_range=(self.area_min, self.area_max),
            morph_ellipse_size=self.morph_ks,
            back_sub=self.bg_subtractor,
            bs_learning_rate=self.bg_lr,
            combine_mode=self.cb_combine.currentText(),
            invert_polarity=self.cb_invert.isChecked(),
            enable_warmup=self.cb_enable_warmup.isChecked(),
            warmup_frames=self.warmup_frames,
        )
        contours = results.get("contours", [])
        det_count = len(contours)
        warmup_active   = bool(results.get("warmup_active", False))
        warmup_progress = float(results.get("warmup_progress", 0.0))

        # Auto-freeze BG après warm-up
        if (self.cb_auto_freeze_bg.isChecked()
            and self.cb_enable_warmup.isChecked()
            and not warmup_active
            and self.bg_subtractor is not None):
            if self.bg_lr > 0.0001:
                self.bg_lr = 0.0001
                self.lbl_bg_lr_val.setText(f"{self.bg_lr:.3f}")
                self.sl_bg_lr.blockSignals(True)
                self.sl_bg_lr.setValue(int(round(self.bg_lr * 100)))
                self.sl_bg_lr.blockSignals(False)

        # Tracking
        self.tracker.update(self.frame_index, contours, frame_bgr.shape[:2])
        active_ids = self.tracker.active_ids(only_moving=self.cb_only_moving.isChecked())

        # Infos (10 Hz)
        self._pending_ui_data = dict(
            frame_index=self.frame_index, det_count=det_count,
            warmup_active=warmup_active, warmup_progress=int(warmup_progress*100),
            active_ids=len(active_ids)
        )
        if not self._ui_update_timer.isActive():
            self._ui_update_timer.start()

        # Dessin des contours
        fit_map = {
            "Rectangle": _draw_rectangle,
            "Min Area Rectangle": _draw_min_area_rect,
            "Convex Hull": _draw_convex_hull,
            "Ellipse": _draw_ellipse,
        }
        fit_fn = fit_map.get(self.cb_shape.currentText(), _draw_rectangle)
        contoured = frame_bgr.copy()
        if contours:
            fit_contours(contoured, contours, fit_fn)

        # Bande warm-up
        if warmup_active and self.cb_enable_warmup.isChecked():
            h, w = contoured.shape[:2]
            overlay = contoured.copy()
            cv.rectangle(overlay, (10, 10), (w - 10, 60), (0, 165, 255), -1)
            cv.addWeighted(contoured, 0.7, overlay, 0.3, 0, contoured)
            text = f"WARM-UP ACTIF - {int(warmup_progress * 100)}% ({int(warmup_progress * self.warmup_frames)}/{self.warmup_frames})"
            cv.putText(contoured, text, (20, 35), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv.putText(contoured, "Filtres anti-parasites actifs", (20, 55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # IDs
        if hasattr(self.tracker, "draw_ids"):
            contoured = self.tracker.draw_ids(
                contoured, only_current=False, frame_idx=self.frame_index,
                only_moving=self.cb_only_moving.isChecked()
            )

        # Écriture overlay MP4 (live)
        self._write_overlay_frame(contoured)

        # Vignettes (mise à l’échelle)
        bin_vis   = results["binary_image"]
        morph_vis = results["morphed_image"]
        if bin_vis.ndim == 2:   bin_vis   = cv.cvtColor(bin_vis,   cv.COLOR_GRAY2BGR)
        if morph_vis.ndim == 2: morph_vis = cv.cvtColor(morph_vis, cv.COLOR_GRAY2BGR)

        def _resize(img):
            h, w = img.shape[:2]
            s = min(1.0, self.max_display_width / float(w))
            if s != 1.0:
                return cv.resize(img, (int(w*s), int(h*s)), interpolation=cv.INTER_AREA)
            return img

        self.lbl_bin["label"].setPixmap(QtGui.QPixmap.fromImage(cv_to_qimage(_resize(bin_vis))))
        self.lbl_morph["label"].setPixmap(QtGui.QPixmap.fromImage(cv_to_qimage(_resize(morph_vis))))
        self.lbl_cont["label"].setPixmap(QtGui.QPixmap.fromImage(cv_to_qimage(_resize(contoured))))

        # Slider suit la lecture
        if self.seek.maximum() > 0:
            self.seek.blockSignals(True)
            self.seek.setValue(max(0, min(self.frame_index, self.seek.maximum())))
            self.seek.blockSignals(False)

    # ------------------------------ mise à jour statut
    def _update_ui_labels(self):
        d = self._pending_ui_data or {}
        fi = int(d.get("frame_index", 0))
        det = int(d.get("det_count", 0))
        self.info_label.setText(f"frame: {fi} / {self.total_frames} | detections: {det}")

        if d.get("warmup_active", False):
            self.warmup_progress.setVisible(True)
            self.warmup_progress.setValue(min(self.warmup_frames, fi))
        else:
            self.warmup_progress.setVisible(False)

        self.status.setText(f"Active IDs: {int(d.get('active_ids', 0))}")

    # ------------------------------ DnD + raccourcis
    def _install_shortcuts(self):
        QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Open), self, activated=self.pick_video)

        # Espace -> bascule Pause
        def _toggle_pause():
            self.btn_pause.setChecked(not self.btn_pause.isChecked())
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self, activated=_toggle_pause)

        # ← / → : step -1 / +1
        def _seek_step(delta):
            if not self.total_frames: return
            self.reader.pause(True)
            newpos = int(np.clip((self.seek.value() if self.seek.maximum()>0 else self.frame_index) + delta,
                                 0, max(0, self.total_frames - 1)))
            self.frame_index = newpos
            self.seek.setValue(newpos)
            if self.reader.isRunning(): self.reader.seek_to(newpos)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left),  self, activated=lambda: _seek_step(-1))
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, activated=lambda: _seek_step(+1))

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent) -> None:
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path and os.path.isfile(path):
                self._load_video_path(path)
                break


# =============================================================================
# Entrée programme
# =============================================================================

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
