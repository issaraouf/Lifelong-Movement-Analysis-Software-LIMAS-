# tracker.py

"""
Multi-object tracker (IDs + CSV) :
- Association sélectionnable : 'greedy', 'hungarian', ou 'auto' (smart).
- Porte dynamique par distance + cohérence d'aire (gating).
- Modèle de mouvement : EMA position/vitesse + snap limit + jump brake.
- Mode "un seul individu" (forcer ID=1).
- Option "Créer un ID seulement si mouvement" (pistes provisoires + confirmation).
- Hystérésis moving/not moving : enter=min_motion, exit=ratio*min_motion (ratio=0.6 par défaut).
"""

from __future__ import annotations
import os, csv, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np
import cv2 as cv


# ------------------------------ Params --------------------------------------

@dataclass
class TrackParams:
    max_dist: float = 320.0
    area_tolerance: float = 0.6
    ema_pos: float = 0.4
    ema_area: float = 0.4
    vel_ema: float = 0.6
    snap_limit_px: float = 60.0
    jump_threshold_px: float = 120.0
    max_miss: int = 30
    min_age_csv: int = 12


# ------------------------------ Track ---------------------------------------

class _Track:
    __slots__ = ("tid", "cx", "cy", "w", "h", "area",
                 "vx", "vy", "ema_pos", "ema_area", "vel_ema",
                 "age", "missed", "last_seen_frame",
                 "pos_hist", "area_hist", "moving",
                 "last_mx", "last_my",
                 "confirmed", "spawn_x", "spawn_y", "first_frame",
                 "csv_handle", "csv_writer", "csv_path", 
                 "last_area")

    def __init__(self, tid: int, x: float, y: float, w: float, h: float, area: float, *,
                 ema_pos: float, ema_area: float, vel_ema: float, frame_idx: int):
        self.tid = tid
        self.cx, self.cy = float(x), float(y)
        self.w, self.h = float(w), float(h)
        self.area = float(area)

        self.vx, self.vy = 0.0, 0.0
        self.ema_pos  = float(ema_pos)
        self.ema_area = float(ema_area)
        self.vel_ema  = float(vel_ema)

        self.age = 0
        self.missed = 0
        self.last_seen_frame = int(frame_idx)

        self.pos_hist = deque(maxlen=256)
        self.area_hist = deque(maxlen=256)
        self.moving = True

        self.last_mx = None
        self.last_my = None

        self.confirmed = True
        self.spawn_x = float(x)
        self.spawn_y = float(y)
        self.first_frame = int(frame_idx)

        self.csv_handle = None
        self.csv_writer = None
        self.csv_path = None

        self.last_area = None

        self._push_hist(self.cx, self.cy, self.area)

    def _push_hist(self, x: float, y: float, area: float):
        self.pos_hist.append((float(x), float(y)))
        self.area_hist.append(float(area))

    def mean_step_px(self) -> float:
        if len(self.pos_hist) < 2:
            return 0.0
        dsum = 0.0; px, py = self.pos_hist[0]
        for (qx, qy) in list(self.pos_hist)[1:]:
            dsum += math.hypot(qx - px, qy - py)
            px, py = qx, qy
        return dsum / max(1, len(self.pos_hist) - 1)

    def predict_pos(self) -> Tuple[float, float]:
        return self.cx + self.vx, self.cy + self.vy

    def update_velocity(self, mx: float, my: float):
        dvx = mx - self.cx
        dvy = my - self.cy
        a = self.vel_ema
        self.vx = (1 - a) * self.vx + a * dvx
        self.vy = (1 - a) * self.vy + a * dvy

    def snap_towards(self, mx: float, my: float, *, snap_limit_px: float, alpha: float, jump_threshold_px: float):
        tx = (1 - alpha) * self.cx + alpha * mx
        ty = (1 - alpha) * self.cy + alpha * my
        dx, dy = tx - self.cx, ty - self.cy
        dist = math.hypot(dx, dy)
        limit = float(snap_limit_px)
        if dist > jump_threshold_px:
            limit *= 0.75
        if dist > limit and dist > 1e-6:
            s = limit / dist
            self.cx += dx * s
            self.cy += dy * s
        else:
            self.cx, self.cy = tx, ty

    def open_csv_if_needed(self, results_dir, video_basename):
        import os, csv
        if getattr(self, "csv_handle", None) is None:
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, f"{video_basename}_ID{self.tid}.csv")
            self.csv_handle = open(path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_handle)
            # ---- en-tête avec raw + filtered ----
            self.csv_writer.writerow(["frame","id","raw_x","raw_y","flt_x","flt_y","w","h","area", "delta_area"])


    def close_csv(self):
        if self.csv_handle:
            try:
                self.csv_handle.flush()
                self.csv_handle.close()
            except Exception:
                pass
        self.csv_handle = None
        self.csv_writer = None


# --------------------------- Hungarian (Munkres) -----------------------------

def _hungarian(cost: np.ndarray) -> List[Tuple[int,int]]:
    """
    Implémentation compacte du Hongrois (minimisation). Accepte matrice carrée.
    Retourne une liste de paires (row_i, col_j) d'affectation.
    Référence: adaptation Python de l'algorithme classique (O(n^3)).
    """
    cost = cost.copy().astype(float)
    n = cost.shape[0]

    # Etape 1: soustraction minima par ligne
    cost -= cost.min(axis=1, keepdims=True)
    # Etape 2: soustraction minima par colonne
    cost -= cost.min(axis=0, keepdims=True)

    # Masques
    starred = np.zeros_like(cost, dtype=bool)
    primed  = np.zeros_like(cost, dtype=bool)
    covered_rows = np.zeros(n, dtype=bool)
    covered_cols = np.zeros(n, dtype=bool)

    # Etape 3: étoile les zéros indépendants
    for i in range(n):
        for j in range(n):
            if (cost[i, j] == 0) and (not covered_rows[i]) and (not covered_cols[j]):
                starred[i, j] = True
                covered_rows[i] = True
                covered_cols[j] = True
    covered_rows[:] = False
    covered_cols[:] = False

    def cover_cols_with_starred():
        covered_cols[:] = starred.any(axis=0)

    cover_cols_with_starred()

    def find_a_zero():
        for i in range(n):
            if covered_rows[i]: continue
            for j in range(n):
                if not covered_cols[j] and cost[i, j] == 0:
                    return i, j
        return None

    def find_star_in_row(r):
        c = np.where(starred[r])[0]
        return (c[0] if len(c) else None)

    def find_star_in_col(c):
        r = np.where(starred[:, c])[0]
        return (r[0] if len(r) else None)

    def find_prime_in_row(r):
        c = np.where(primed[r])[0]
        return (c[0] if len(c) else None)

    def augment_path(path):
        for (r, c) in path:
            starred[r, c] = not starred[r, c]
            primed[r, c] = False

    def clear_primes():
        primed[:, :] = False

    # Boucle principale
    while True:
        if covered_cols.sum() == n:
            break
        z = find_a_zero()
        while z is None:
            # Etape 5: créer plus de zéros
            m = np.min(cost[~covered_rows][:, ~covered_cols])
            cost[~covered_rows, :] -= m
            cost[:,  covered_cols] += m
            z = find_a_zero()

        (r, c) = z
        primed[r, c] = True
        sc = find_star_in_row(r)
        if sc is not None:
            covered_rows[r] = True
            covered_cols[sc] = False
        else:
            # Etape 4: construire chemin augmentant
            path = [(r, c)]
            st = find_star_in_col(c)
            while st is not None:
                path.append((st, c))
                pr = find_prime_in_row(st)
                path.append((st, pr))
                c = pr
                st = find_star_in_col(c)
            augment_path(path)
            clear_primes()
            covered_rows[:] = False
            covered_cols[:] = False
            cover_cols_with_starred()

    return [(i, j) for i in range(n) for j in range(n) if starred[i, j]]


# --------------------------- MultiTracker ------------------------------------

class MultiTracker:
    def __init__(self, params: TrackParams = TrackParams()):
        self.params = params
        self.tracks: Dict[int, _Track] = {}
        self.next_tid = 1

        self.recording = False
        self.per_id_csv = True
        self.enable_combined = False
        self.combined_path: Optional[str] = None
        self.combined_handle = None
        self.combined_writer = None
        self.results_dir = "results"
        self.video_basename = "video"

        # motion filter + hysteresis
        self.static_window = 12
        self.min_motion_px = 2.0
        self.only_moving = False
        self.hyst_ratio_exit = 0.6
        self.motion_enter_px = self.min_motion_px
        self.motion_exit_px  = self.min_motion_px * self.hyst_ratio_exit

        # modes
        self.force_single_id = False
        self.require_motion_to_spawn = False
        self.confirm_motion_px = 5.0
        self.confirm_window = 8
        self.confirm_timeout = 180

        # assignment
        self.assignment_method = "greedy"   # 'greedy' | 'hungarian' | 'auto'
        self.area_penalty = 0.002           # poids pénalité aire (px)
        self.max_assign_cost = 1e9          # seuil "infini" (gating)

        self._debug = False

    # --- Utils debug --------------------------------------------------------
    def set_debug(self, on: bool = True):
        self._debug = bool(on)
    def _dbg(self, *a):
        if self._debug: print("[Tracker]", *a)

    # --- Session I/O --------------------------------------------------------
    def start_new_video(self, video_path: str, *, per_id_csv=True, enable_combined=False, session_name: Optional[str] = None):
        self.close_all()
        self.tracks.clear()
        self.next_tid = 1
        self.per_id_csv = bool(per_id_csv)
        self.enable_combined = bool(enable_combined)

        self.video_basename = os.path.splitext(os.path.basename(video_path))[0] or "video"
        base_dir = os.path.dirname(video_path) or "."
        results_root = os.path.join(base_dir, "results")
        self.results_dir = os.path.join(results_root, session_name) if session_name else results_root
        os.makedirs(self.results_dir, exist_ok=True)

        self.combined_path = os.path.join(self.results_dir, f"{self.video_basename}_tracks.csv") if enable_combined else None
        self.combined_handle = None
        self.combined_writer = None

        if self.enable_combined and self.combined_writer is not None:
            self.combined_writer.writerow(["frame","id","raw_x","raw_y","flt_x","flt_y","w","h","area", "delta_area"])

    def set_recording(self, on: bool):
        self.recording = bool(on)

    # --- Config -------------------------------------------------------------
    def set_assignment_method(self, method: str):
        method = (method or "").strip().lower()
        if method not in ("greedy","hungarian","auto"):
            method = "greedy"
        self.assignment_method = method
        self._dbg("Assignment method:", method)

    def set_motion_filter(self, *, static_window: int, min_motion_px: float, only_moving: bool):
        self.static_window = int(max(2, static_window))
        self.min_motion_px = float(max(0.0, min_motion_px))
        self.only_moving = bool(only_moving)
        self.motion_enter_px = self.min_motion_px
        self.motion_exit_px  = max(0.0, self.hyst_ratio_exit * self.min_motion_px)
        self._dbg(f"Motion filter: window={self.static_window}, enter={self.motion_enter_px:.2f}, exit={self.motion_exit_px:.2f}, only_moving={self.only_moving}")

    def set_hysteresis_ratio(self, ratio: float = 0.6):
        r = float(max(0.0, min(1.0, ratio)))
        self.hyst_ratio_exit = r
        self.motion_exit_px = max(0.0, self.hyst_ratio_exit * self.motion_enter_px)

    def set_single_target_mode(self, on: bool):
        self.force_single_id = bool(on)

    def set_require_motion_to_spawn(self, on: bool):
        self.require_motion_to_spawn = bool(on)

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

    # --- Helpers ------------------------------------------------------------
    @staticmethod
    def _contour_measure(c) -> Tuple[float, float, float, float, float]:
        area = float(cv.contourArea(c))
        if len(c) >= 5:
            (cx, cy), (MA, ma), _ = cv.fitEllipse(c)
            return (float(cx), float(cy), float(MA), float(ma), area)
        else:
            (cx, cy), (size_w, size_h), _ = cv.minAreaRect(c)
            return (float(cx), float(cy), float(size_w), float(size_h), area)

    def _ensure_combined(self):
        if self.enable_combined and self.combined_handle is None:
            os.makedirs(self.results_dir, exist_ok=True)
            self.combined_handle = open(self.combined_path, "w", newline="", encoding="utf-8")
            self.combined_writer = csv.writer(self.combined_handle)
            # header exactement comme les lignes écrites :
            self.combined_writer.writerow([
                "frame","id","raw_x","raw_y","flt_x","flt_y","w","h","area","delta_area"
            ])

    # ------------------------------ UPDATE ----------------------------------
    def update(self, frame_idx: int, contours: List[np.ndarray], frame_shape):
        dets = [self._contour_measure(c) for c in contours]

        # ========== Mode "ID=1" =============================================
        if self.force_single_id:
            for tid in list(self.tracks.keys()):
                if tid != 1:
                    self.tracks[tid].close_csv(); del self.tracks[tid]
            def _upd_from_det(t: _Track, d):
                (mx, my, w_det, h_det, a_det) = d
                t.update_velocity(mx, my)
                t.snap_towards(mx, my,
                               snap_limit_px=self.params.snap_limit_px,
                               alpha=self.params.ema_pos,
                               jump_threshold_px=self.params.jump_threshold_px)
                a = self.params.ema_pos; b = self.params.ema_area
                t.w  = (1-a)*t.w + a*w_det
                t.h  = (1-a)*t.h + a*h_det
                t.area = (1-b)*t.area + b*a_det
                t.last_mx, t.last_my = float(mx), float(my)
                t._push_hist(t.cx, t.cy, t.area)
                t.age += 1; t.missed = 0; t.last_seen_frame = int(frame_idx)
            if dets:
                if 1 in self.tracks:
                    t = self.tracks[1]
                    px, py = t.predict_pos()
                    gate = max(self.params.max_dist, 3.0 * t.mean_step_px())
                    near = [(i, d) for i, d in enumerate(dets) if math.hypot(d[0]-px, d[1]-py) <= gate]
                    di = (max(near, key=lambda t2: t2[1][4])[0] if near else int(np.argmax([d[4] for d in dets])))
                    _upd_from_det(t, dets[di])
                else:
                    (cx, cy, w, h, a) = max(dets, key=lambda d: d[4])
                    self.tracks[1] = _Track(1, cx, cy, w, h, a,
                                            ema_pos=self.params.ema_pos,
                                            ema_area=self.params.ema_area,
                                            vel_ema=self.params.vel_ema,
                                            frame_idx=frame_idx)
                    self.tracks[1].last_mx, self.tracks[1].last_my = float(cx), float(cy)
            else:
                if 1 in self.tracks:
                    t = self.tracks[1]
                    t.cx += t.vx; t.cy += t.vy
                    t._push_hist(t.cx, t.cy, t.area)
                    t.missed += 1
                    if t.missed > self.params.max_miss:
                        t.close_csv(); del self.tracks[1]
            self._update_moving_and_csv(frame_idx)
            return

        # ========== Mode normal =============================================
        T_ids = list(self.tracks.keys())
        if T_ids:
            pred_pos = np.array([self.tracks[tid].predict_pos() for tid in T_ids], dtype=np.float32)
            T_area  = np.array([self.tracks[tid].area for tid in T_ids], dtype=np.float32)
        else:
            pred_pos = np.zeros((0,2), np.float32)
            T_area   = np.zeros((0,), np.float32)

        D_pos  = np.array([(d[0], d[1]) for d in dets], dtype=np.float32) if dets else np.zeros((0,2), np.float32)
        D_area = np.array([d[4] for d in dets], dtype=np.float32) if dets else np.zeros((0,), np.float32)

        assigned_tracks, assigned_dets = set(), set()

        # --------- Choix d'assignation -------------------------------------
        method = self.assignment_method
        if method == "auto":
            # heuristique simple : si ≥2 pistes ET ≥2 détections → Hongrois
            method = "hungarian" if (len(T_ids) >= 2 and len(dets) >= 2) else "greedy"

        if method == "hybrid" and len(T_ids) and len(dets):
            nT, nD = len(T_ids), len(dets)
            # 1) Prépare portes dynamiques et tolérance d'aire
            gates = np.empty(nT, dtype=np.float32)
            tols  = np.empty(nT, dtype=np.float32)
            for ti, tid in enumerate(T_ids):
                speed = self.tracks[tid].mean_step_px()
                gates[ti] = max(self.params.max_dist, 3.0 * speed)
                tol = self.params.area_tolerance
                if speed > 10.0: tol = min(0.95, tol * 1.5)
                tols[ti] = tol

            # Distances piste<->détection
            diff  = pred_pos[:, None, :] - D_pos[None, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))

            # 2) Étape Greedy "sûre" (très proche ET unique)
            near_factor = 0.4  # paires vraiment évidentes
            used_T = set(); used_D = set()
            greedy_pairs = []
            # meilleur det pour chaque track
            bestD_for_T = np.argmin(dists, axis=1)
            # meilleur track pour chaque det
            bestT_for_D = np.argmin(dists, axis=0)

            for ti in range(nT):
                di = int(bestD_for_T[ti])
                d  = float(dists[ti, di])
                if d <= near_factor * float(gates[ti]) and bestT_for_D[di] == ti:
                    greedy_pairs.append((d, ti, di))

            # Verrouille les paires sûres (plus proche d’abord)
            greedy_pairs.sort(key=lambda t: t[0])
            for _, ti, di in greedy_pairs:
                if ti in used_T or di in used_D: 
                    continue
                used_T.add(ti); used_D.add(di)
                assigned_tracks.add(T_ids[ti]); assigned_dets.add(di)
                self._apply_measure(T_ids[ti], dets[di], frame_idx)

            # 3) Si tout est assigné, on s’arrête là
            if len(used_T) == nT or len(used_D) == nD:
                pass
            else:
                # 4) Étape Hungarian sur le sous-problème restant
                rem_T = [ti for ti in range(nT) if ti not in used_T]
                rem_D = [di for di in range(nD) if di not in used_D]
                if rem_T and rem_D:
                    # matrice de coût (gating + pénalité d'aire)
                    INF = self.max_assign_cost
                    cost = np.full((len(rem_T), len(rem_D)), INF, dtype=np.float64)
                    for r, ti in enumerate(rem_T):
                        a_min = (1.0 - tols[ti]) * max(T_area[ti], 1e-6)
                        a_max = (1.0 + tols[ti]) * max(T_area[ti], 1e-6)
                        gate  = float(gates[ti])
                        for c, di in enumerate(rem_D):
                            dist = float(dists[ti, di])
                            if dist <= 0.6 * gate:
                                area_pen = 0.0
                                cost[r, c] = dist + area_pen
                            else:
                                if (a_min <= D_area[di] <= a_max) and (dist <= gate):
                                    area_pen = self.area_penalty * abs(D_area[di]-T_area[ti]) / max(T_area[ti], 1e-6)
                                    cost[r, c] = dist + area_pen

                    # padding carré + Hongrois
                    n = max(cost.shape)
                    pad = np.full((n, n), INF, dtype=np.float64)
                    pad[:cost.shape[0], :cost.shape[1]] = cost
                    pairs = _hungarian(pad)
                    for (ri, cj) in pairs:
                        if ri < cost.shape[0] and cj < cost.shape[1]:
                            if pad[ri, cj] >= INF: 
                                continue
                            ti = rem_T[ri]; di = rem_D[cj]
                            assigned_tracks.add(T_ids[ti]); assigned_dets.add(di)
                            self._apply_measure(T_ids[ti], dets[di], frame_idx)

        if method == "hungarian" and len(T_ids) and len(dets):
            # matrice de coût avec gating + pénalité d'aire
            nT, nD = len(T_ids), len(dets)
            # calcul gates/tol par piste
            gates = np.empty(nT, dtype=np.float32)
            tol   = np.empty(nT, dtype=np.float32)
            for ti, tid in enumerate(T_ids):
                speed = self.tracks[tid].mean_step_px()
                gates[ti] = max(self.params.max_dist, 3.0 * speed)
                tol[ti]   = self.params.area_tolerance if speed <= 10.0 else min(0.95, self.params.area_tolerance * 1.5)

            cost = np.full((nT, nD), self.max_assign_cost, dtype=np.float64)
            for ti in range(nT):
                a_min = (1.0 - tol[ti]) * max(T_area[ti], 1e-6)
                a_max = (1.0 + tol[ti]) * max(T_area[ti], 1e-6)
                for di in range(nD):
                    dist = float(np.hypot(pred_pos[ti,0]-D_pos[di,0], pred_pos[ti,1]-D_pos[di,1]))
                    # passe courte : <= 0.6*gate -> ignorer l'aire (favorise rattrapage)
                    if dist <= 0.6 * gates[ti]:
                        area_pen = 0.0
                        cost[ti, di] = dist + area_pen
                    else:
                        if (a_min <= D_area[di] <= a_max) and (dist <= gates[ti]):
                            area_pen = self.area_penalty * abs(D_area[di] - T_area[ti]) / max(T_area[ti], 1e-6)
                            cost[ti, di] = dist + area_pen
                        # sinon: coût infini -> non assignable

            # pad en carré
            n = max(nT, nD)
            pad = np.full((n, n), self.max_assign_cost, dtype=np.float64)
            pad[:nT, :nD] = cost
            pairs = _hungarian(pad)

            # appliquer les paires valides
            for (ri, cj) in pairs:
                if ri < nT and cj < nD:
                    c = pad[ri, cj]
                    if c >= self.max_assign_cost:
                        continue
                    ti, di = ri, cj
                    assigned_tracks.add(T_ids[ti]); assigned_dets.add(di)
                    self._apply_measure(T_ids[ti], dets[di], frame_idx)
        else:
            # Greedy (comme avant)
            if len(T_ids) and len(dets):
                diff = pred_pos[:, None, :] - D_pos[None, :, :]
                dist = np.sqrt(np.sum(diff**2, axis=2))

                pairs = []
                if len(T_ids) == 1 and len(dets) == 1:
                    pairs.append((0.0, 0, 0))
                else:
                    dyn_gate, dyn_tol = [], []
                    for ti, tid in enumerate(T_ids):
                        speed = self.tracks[tid].mean_step_px()
                        gate  = max(self.params.max_dist, 3.0 * speed)
                        tol   = self.params.area_tolerance
                        if speed > 10.0:
                            tol = min(0.95, tol * 1.5)
                        dyn_gate.append(gate); dyn_tol.append(tol)
                    for ti, tid in enumerate(T_ids):
                        gate = float(dyn_gate[ti]); tol = float(dyn_tol[ti])
                        a_min = (1.0 - tol) * max(T_area[ti], 1e-6)
                        a_max = (1.0 + tol) * max(T_area[ti], 1e-6)
                        for di in range(len(dets)):
                            dval = float(dist[ti, di])
                            if dval <= 0.6 * gate:
                                pairs.append((dval, ti, di)); continue
                            if (a_min <= D_area[di] <= a_max) and (dval <= gate):
                                pairs.append((dval, ti, di))
                pairs.sort(key=lambda t: t[0])
                used_T, used_D = set(), set()
                for _, ti, di in pairs:
                    if ti in used_T or di in used_D:
                        continue
                    used_T.add(ti); used_D.add(di)
                    assigned_tracks.add(T_ids[ti]); assigned_dets.add(di)
                    self._apply_measure(T_ids[ti], dets[di], frame_idx)

        # coasting + TTL pour non assignés
        for tid in list(self.tracks.keys()):
            if tid in assigned_tracks: continue
            t = self.tracks[tid]
            t.cx += t.vx; t.cy += t.vy
            t._push_hist(t.cx, t.cy, t.area)
            t.missed += 1
            if t.missed > self.params.max_miss:
                t.close_csv(); del self.tracks[tid]

        # naissances pour les détections non assignées
        for di, d in enumerate(dets):
            if di in assigned_dets: continue
            (cx, cy, w_det, h_det, a_det) = d
            tid = self.next_tid; self.next_tid += 1
            t = _Track(tid, cx, cy, w_det, h_det, a_det,
                       ema_pos=self.params.ema_pos,
                       ema_area=self.params.ema_area,
                       vel_ema=self.params.vel_ema,
                       frame_idx=frame_idx)
            t.last_mx = float(cx); t.last_my = float(cy)
            if self.require_motion_to_spawn:
                t.confirmed = False
                t.spawn_x, t.spawn_y = t.cx, t.cy
                t.first_frame = frame_idx
            self.tracks[tid] = t

        # moving + confirmation + CSV
        self._update_moving_and_csv(frame_idx)

    # --- apply measure to track --------------------------------------------
    def _apply_measure(self, tid: int, det, frame_idx: int):
        (mx, my, w_det, h_det, a_det) = det
        t = self.tracks[tid]
        t.update_velocity(mx, my)
        t.snap_towards(mx, my,
                       snap_limit_px=self.params.snap_limit_px,
                       alpha=self.params.ema_pos,
                       jump_threshold_px=self.params.jump_threshold_px)
        a = self.params.ema_pos; b = self.params.ema_area
        t.w  = (1-a) * t.w  + a * w_det
        t.h  = (1-a) * t.h  + a * h_det
        t.area = (1-b) * t.area + b * a_det
        t.last_mx = float(mx); t.last_my = float(my)
        t._push_hist(t.cx, t.cy, t.area)
        t.age += 1; t.missed = 0; t.last_seen_frame = int(frame_idx)

    # --- moving/hysteresis + CSV -------------------------------------------
    def _update_moving_and_csv(self, frame_idx: int):
        win = max(2, self.static_window)
        for t in self.tracks.values():
            pts = list(t.pos_hist)[-win:]
            if len(pts) >= 2:
                dsum = 0.0; px, py = pts[0]
                for (qx, qy) in pts[1:]:
                    dsum += math.hypot(qx - px, qy - py); px, py = qx, qy
                mean_step = dsum / (len(pts) - 1)
            else:
                mean_step = 0.0
            if t.moving:
                t.moving = (mean_step >= self.motion_exit_px)
            else:
                t.moving = (mean_step >= self.motion_enter_px)

        # confirmation des pistes provisoires
        if self.require_motion_to_spawn:
            for tid in list(self.tracks.keys()):
                t = self.tracks[tid]
                if t.confirmed: continue
                age = frame_idx - t.first_frame
                moved = math.hypot(t.cx - t.spawn_x, t.cy - t.spawn_y)
                if age <= self.confirm_window and moved >= self.confirm_motion_px:
                    t.confirmed = True
                elif age > self.confirm_timeout:
                    t.close_csv(); del self.tracks[tid]

        # CSV
        if self.recording:
            self._write_csv(frame_idx)

    # --- CSV ----------------------------------------------------------------
    def _write_csv(self, frame_idx: int):
        self._ensure_combined()
        for t in self.tracks.values():
            if t.last_seen_frame != frame_idx:
                continue
            if self.only_moving and not t.moving:
                continue
            if self.require_motion_to_spawn and not t.confirmed:
                continue

            # positions RAW vs FILTERED
            mx = t.last_mx if getattr(t, "last_mx", None) is not None else t.cx
            my = t.last_my if getattr(t, "last_my", None) is not None else t.cy

            # delta_area en valeur absolue (1ère ligne vide)
            if t.last_area is None:
                delta_str = ""                           # pas de valeur pour la première ligne
            else:
                delta_str = f"{abs(t.area - t.last_area):.3f}"
            t.last_area = t.area

            # Per-ID CSV
            if self.per_id_csv and t.age >= self.params.min_age_csv:
                if t.csv_handle is None:
                    t.open_csv_if_needed(self.results_dir, self.video_basename)
                t.csv_writer.writerow([
                    frame_idx, t.tid,
                    f"{mx:.3f}", f"{my:.3f}",
                    f"{t.cx:.3f}", f"{t.cy:.3f}",
                    f"{t.w:.3f}", f"{t.h:.3f}", f"{t.area:.3f}",
                    delta_str
                ])

            # Combined CSV
            if self.enable_combined and self.combined_writer is not None:
                self.combined_writer.writerow([
                    frame_idx, t.tid,
                    f"{mx:.3f}", f"{my:.3f}",
                    f"{t.cx:.3f}", f"{t.cy:.3f}",
                    f"{t.w:.3f}", f"{t.h:.3f}", f"{t.area:.3f}",
                    delta_str
                ])


    # --- UI helpers ---------------------------------------------------------
    def active_ids(self, only_moving: bool = False):
        """
        Retourne la liste des IDs actifs.
        only_moving=True -> filtre sur l'état 'moving' (hystérésis interne).
        """
        return [tid for tid, t in self.tracks.items()
                if (not only_moving) or getattr(t, "moving", True)]

    def draw_ids(self, img_bgr, *, only_current: bool = False,
                frame_idx: int | None = None, only_moving: bool = False):
        """
        Dessine la croix + le label des pistes.

        only_current=True  -> ne dessine que les pistes VRAIMENT appariées
                            à une détection sur ce frame (pas les prédictions).
        only_moving=True   -> saute les pistes marquées 'not moving'.
        frame_idx          -> index courant (requis pour only_current).
        """
        # En mode "un seul individu", on dessine toujours la piste si elle existe
        if getattr(self, "force_single_id", False):
            only_moving = False

        cur_idx = int(frame_idx) if frame_idx is not None else None

        for tid, t in self.tracks.items():
            if only_moving and not getattr(t, "moving", True):
                continue
            if only_current and (cur_idx is not None) and (getattr(t, "last_seen_frame", None) != cur_idx):
                continue

            # Si on a eu une mesure sur ce frame, dessiner à (last_mx,last_my),
            # sinon dessiner à la position filtrée (cx,cy)
            if (cur_idx is not None) and (getattr(t, "last_seen_frame", None) == cur_idx) and (getattr(t, "last_mx", None) is not None):
                px, py = int(round(t.last_mx)), int(round(t.last_my))
            else:
                px, py = int(round(t.cx)), int(round(t.cy))

            color = (0, 255, 0) if getattr(t, "confirmed", False) else (160, 160, 160)
            cv.drawMarker(img_bgr, (px, py), color,
                        markerType=cv.MARKER_CROSS, markerSize=16, thickness=2)
            label = f"ID {tid}" if getattr(t, "confirmed", False) else f"?{tid}"
            cv.putText(img_bgr, label, (px + 8, py - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv.LINE_AA)

        return img_bgr
