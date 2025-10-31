import os, csv, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np
import cv2 as cv


@dataclass
class TrackParams:
    # Association gates
    max_dist: float = 300.0            # px, distance max entre dernière pos et nouvelle détection
    area_tolerance: float = 0.6        # +-60% de tolérance sur l’aire
    # Lissage
    ema_pos: float = 0.4
    ema_area: float = 0.4
    # Robustesse (vitesse/occlusions)
    max_miss: int = 12                 # garde la piste en vie pendant X frames manquées
    min_age_csv: int = 6               # n’ouvre le CSV qu’après X frames d’âge


class _Track:
    __slots__ = ("tid", "cx", "cy", "w", "h", "area", "ema_pos", "ema_area",
                 "age", "missed", "last_seen_frame", "pos_hist", "area_hist",
                 "moving", "csv_handle", "csv_writer", "csv_path")

    def __init__(self, tid: int, x: float, y: float, w: float, h: float, area: float, *,
                 ema_pos: float, ema_area: float, frame_idx: int):
        self.tid = tid
        self.cx, self.cy = float(x), float(y)
        self.w, self.h = float(w), float(h)
        self.area = float(area)
        self.ema_pos = float(ema_pos)
        self.ema_area = float(ema_area)
        self.age = 0
        self.missed = 0
        self.last_seen_frame = int(frame_idx)
        self.pos_hist = deque(maxlen=256)
        self.area_hist = deque(maxlen=256)
        self.moving = True
        self.csv_handle = None
        self.csv_writer = None
        self.csv_path = None
        self._push_hist(x, y, area)

    def _push_hist(self, x: float, y: float, area: float):
        self.pos_hist.append((float(x), float(y)))
        self.area_hist.append(float(area))

    def mean_step_px(self) -> float:
        if len(self.pos_hist) < 2:
            return 0.0
        dsum = 0.0; n = 0
        px, py = self.pos_hist[0]
        for (qx, qy) in list(self.pos_hist)[1:]:
            dsum += math.hypot(qx - px, qy - py)
            px, py = qx, qy
            n += 1
        return dsum / max(1, n)

    def open_csv_if_needed(self, base_folder: str, base_name: str):
        if self.csv_handle is None:
            os.makedirs(base_folder, exist_ok=True)
            self.csv_path = os.path.join(base_folder, f"{base_name}_id{self.tid}.csv")
            self.csv_handle = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_handle)
            self.csv_writer.writerow(["frame", "id", "cx", "cy", "w", "h", "area"])

    def close_csv(self):
        if self.csv_handle:
            try:
                self.csv_handle.flush()
                self.csv_handle.close()
            except Exception:
                pass
        self.csv_handle = None
        self.csv_writer = None


class MultiTracker:
    """
    API utilisée par main.py :
      - start_new_video(video_path, per_id_csv=True, enable_combined=False)
      - set_motion_filter(static_window, min_motion_px, only_moving)
      - set_recording(on: bool)
      - update(frame_idx, contours, frame_shape)
      - active_ids(only_moving: bool=False) -> List[int]
      - draw_ids(img_bgr, only_current, frame_idx, only_moving) -> img_bgr
      - close_all()
    """
    def __init__(self, params: TrackParams = TrackParams()):
        self.params = params
        self.tracks: Dict[int, _Track] = {}
        self.next_tid = 1

        # Enregistrement
        self.recording = False
        self.per_id_csv = True
        self.enable_combined = False
        self.combined_path: Optional[str] = None
        self.combined_handle = None
        self.combined_writer = None
        self.results_dir = "results"
        self.video_basename = "video"

        # Filtre “moving”
        self.static_window = 12
        self.min_motion_px = 2.0
        self.only_moving = False

    # ---------- session / IO ----------
    def start_new_video(self, video_path: str, *, per_id_csv=True, enable_combined=False):
        self.close_all()
        self.tracks.clear()
        self.next_tid = 1
        self.per_id_csv = bool(per_id_csv)
        self.enable_combined = bool(enable_combined)
        self.video_basename = os.path.splitext(os.path.basename(video_path))[0] or "video"
        self.results_dir = os.path.join(os.path.dirname(video_path) or ".", "results")
        os.makedirs(self.results_dir, exist_ok=True)

        # Combo: on l’ouvrira à la 1ère écriture
        self.combined_path = os.path.join(self.results_dir, f"{self.video_basename}_tracks.csv") if enable_combined else None
        self.combined_handle = None
        self.combined_writer = None

    def set_recording(self, on: bool):
        self.recording = bool(on)

    def set_motion_filter(self, *, static_window: int, min_motion_px: float, only_moving: bool):
        self.static_window = int(max(2, static_window))
        self.min_motion_px = float(max(0.0, min_motion_px))
        self.only_moving = bool(only_moving)

    def close_all(self):
        for t in list(self.tracks.values()):
            t.close_csv()
        if self.combined_handle:
            try:
                self.combined_handle.flush()
                self.combined_handle.close()
            except Exception:
                pass
        self.combined_handle = None
        self.combined_writer = None

    # ---------- helpers ----------
    @staticmethod
    def _contour_measure(c) -> Tuple[float, float, float, float, float]:
        # (cx, cy, w, h, area)
        m = cv.moments(c)
        if m["m00"] == 0:
            x, y, w, h = cv.boundingRect(c)
            return (x + w/2.0, y + h/2.0, w, h, float(cv.contourArea(c)))
        cx = m["m10"]/m["m00"]; cy = m["m01"]/m["m00"]
        x, y, w, h = cv.boundingRect(c)
        return (cx, cy, float(w), float(h), float(cv.contourArea(c)))

    def _ensure_combined(self):
        if self.enable_combined and self.combined_handle is None:
            os.makedirs(self.results_dir, exist_ok=True)
            self.combined_handle = open(self.combined_path, "w", newline="", encoding="utf-8")
            self.combined_writer = csv.writer(self.combined_handle)
            self.combined_writer.writerow(["frame", "id", "cx", "cy", "w", "h", "area"])

    # ---------- update principal ----------
    def update(self, frame_idx: int, contours: List[np.ndarray], frame_shape):
        """Met à jour avec les détections et écrit CSV si activé."""
        dets = [self._contour_measure(c) for c in contours]

        # Tracks et detections en matrices
        T_ids = list(self.tracks.keys())
        T_pos = np.array([(self.tracks[tid].cx, self.tracks[tid].cy) for tid in T_ids], dtype=np.float32) if T_ids else np.zeros((0,2), np.float32)
        D_pos = np.array([(d[0], d[1]) for d in dets], dtype=np.float32) if dets else np.zeros((0,2), np.float32)
        D_area = np.array([d[4] for d in dets], dtype=np.float32) if dets else np.zeros((0,), np.float32)

        assigned_tracks, assigned_dets = set(), set()

        if len(T_ids) and len(dets):
            # distance & compatibilité d’aire
            diff = T_pos[:,None,:] - D_pos[None,:,:]              # (T, D, 2)
            dist = np.sqrt(np.sum(diff**2, axis=2))               # (T, D)
            T_area = np.array([self.tracks[tid].area for tid in T_ids], dtype=np.float32)
            area_ratio = (D_area[None,:] / np.maximum(T_area[:,None], 1e-6))
            area_ok = (area_ratio > (1.0 - self.params.area_tolerance)) & (area_ratio < (1.0 + self.params.area_tolerance))

            # appariement glouton par distance croissante
            pairs = []
            for ti, tid in enumerate(T_ids):
                for di in range(len(dets)):
                    if area_ok[ti, di] and dist[ti, di] <= self.params.max_dist:
                        pairs.append((float(dist[ti, di]), ti, di))
            pairs.sort(key=lambda t: t[0])

            used_T, used_D = set(), set()
            for _, ti, di in pairs:
                if ti in used_T or di in used_D:
                    continue
                used_T.add(ti); used_D.add(di)
                assigned_tracks.add(T_ids[ti]); assigned_dets.add(di)
                # mise à jour de la piste
                t = self.tracks[T_ids[ti]]
                (cx, cy, w_det, h_det, a_det) = dets[di]
                alpha = self.params.ema_pos
                t.cx = (1-alpha)*t.cx + alpha*cx
                t.cy = (1-alpha)*t.cy + alpha*cy
                t.w = (1-alpha)*t.w + alpha*w_det
                t.h = (1-alpha)*t.h + alpha*h_det
                beta = self.params.ema_area
                t.area = (1-beta)*t.area + beta*a_det

                t._push_hist(t.cx, t.cy, t.area)
                t.age += 1
                t.missed = 0
                t.last_seen_frame = int(frame_idx)

        # Pistes non assignées : on les garde vivantes jusqu’à max_miss
        for tid in list(self.tracks.keys()):
            if tid in assigned_tracks:
                continue
            t = self.tracks[tid]
            t.missed += 1
            if t.missed > self.params.max_miss:
                t.close_csv()
                del self.tracks[tid]

        # Nouvelles détections non assignées → créer des pistes
        for di, d in enumerate(dets):
            if di in assigned_dets:
                continue
            (cx, cy, w_det, h_det, a_det) = d
            tid = self.next_tid; self.next_tid += 1
            self.tracks[tid] = _Track(tid, cx, cy, w_det, h_det, a_det,
                                      ema_pos=self.params.ema_pos, ema_area=self.params.ema_area, frame_idx=frame_idx)

        # Etat “moving” (fenêtre glissante)
        win = max(2, self.static_window)
        for t in self.tracks.values():
            n = len(t.pos_hist)
            if n >= 2:
                pts = list(t.pos_hist)[-win:]
                dsum = 0.0; m = 0
                px, py = pts[0]
                for (qx, qy) in pts[1:]:
                    dsum += math.hypot(qx - px, qy - py)
                    px, py = qx, qy
                    m += 1
                mean_step = dsum / max(1, m)
            else:
                mean_step = 0.0
            t.moving = (mean_step >= self.min_motion_px)

        # Écriture CSV
        if self.recording:
            self._write_csv(frame_idx)

    def _ensure_combined(self):
        if self.enable_combined and self.combined_handle is None:
            os.makedirs(self.results_dir, exist_ok=True)
            self.combined_handle = open(self.combined_path, "w", newline="", encoding="utf-8")
            self.combined_writer = csv.writer(self.combined_handle)
            self.combined_writer.writerow(["frame", "id", "cx", "cy", "w", "h", "area"])

    def _write_csv(self, frame_idx: int):
        self._ensure_combined()
        for t in self.tracks.values():
            # 1) si la piste N'A PAS été vue à cette frame, on saute
            if t.last_seen_frame != frame_idx:
                continue

            # 2) filtre "only moving"
            if self.only_moving and not t.moving:
                continue

            # 3) per-ID : ouverture retardée (si tu as ce mécanisme)
            if self.per_id_csv and t.age >= self.params.min_age_csv:
                if t.csv_handle is None:
                    t.open_csv_if_needed(self.results_dir, self.video_basename)
                t.csv_writer.writerow([frame_idx, t.tid,
                                    f"{t.cx:.3f}", f"{t.cy:.3f}",
                                    f"{t.w:.3f}", f"{t.h:.3f}", f"{t.area:.3f}"])

            # 4) combined CSV (optionnel)
            if self.enable_combined and self.combined_writer is not None:
                self.combined_writer.writerow([frame_idx, t.tid,
                                            f"{t.cx:.3f}", f"{t.cy:.3f}",
                                            f"{t.w:.3f}", f"{t.h:.3f}", f"{t.area:.3f}"])

    # ---------- helpers UI ----------
    def active_ids(self, *, only_moving: bool = False) -> List[int]:
        if not only_moving:
            return list(self.tracks.keys())
        return [tid for tid, t in self.tracks.items() if t.moving]

    def draw_ids(self, img_bgr, *, only_current: bool = False, frame_idx: int = None, only_moving: bool = False):
        for tid, t in self.tracks.items():
            if only_moving and not t.moving:
                continue
            if only_current and frame_idx is not None and (t.last_seen_frame != int(frame_idx)):
                continue
            p = (int(round(t.cx)), int(round(t.cy)))
            cv.drawMarker(img_bgr, p, (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=16, thickness=2)
            cv.putText(img_bgr, f"ID {tid}", (p[0]+8, p[1]-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv.LINE_AA)
        return img_bgr
