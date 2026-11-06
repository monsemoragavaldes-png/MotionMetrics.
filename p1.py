"""
M.M. - MotionMetrics
Modos: Automático (un video) y Varios videos (un movimiento por video).
Ambos modos usan el MISMO motor de análisis (ángulo con signo + dirección del movimiento).
"""

import io
import math
import os
import re
import tempfile
import unicodedata
import urllib.request
from datetime import date
from fractions import Fraction

import av
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image

# ----------------------------------------------------------------------------
# 1. Configuración de Modelos y Datos Clínicos
# ----------------------------------------------------------------------------

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

BODY_PART_LABELS = {
    "hombro": "Hombro",
    "codo": "Codo",
    "cadera": "Cadera",
    "rodilla": "Rodilla",
    "tobillo": "Tobillo",
}

NORMATIVE_RANGES = {
    "hombro": {"flexion": (0, 180), "extension": (0, 60), "abduccion": (0, 180), "aduccion": (0, 30)},
    "codo": {"flexion": (0, 150), "extension": (0, 10)},
    "cadera": {"flexion": (0, 120), "extension": (0, 30), "abduccion": (0, 45), "aduccion": (0, 30)},
    "rodilla": {"flexion": (0, 135), "extension": (0, 10)},
    "tobillo": {"dorsiflexion": (0, 20), "plantiflexion": (0, 50)},
}

# lm = [punto proximal, vértice (articulación), punto distal] de MediaPipe Pose
MOVEMENTS = {
    "hombro": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "lm": {"left": [23, 11, 13], "right": [24, 12, 14]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "lm": {"left": [23, 11, 13], "right": [24, 12, 14]}},
        {"id": "abduccion", "label": "Abducción", "view": "frontal", "lm": {"left": [23, 11, 13], "right": [24, 12, 14]}},
        {"id": "aduccion", "label": "Aducción", "view": "frontal", "lm": {"left": [23, 11, 13], "right": [24, 12, 14]}},
    ],
    "codo": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "lm": {"left": [11, 13, 15], "right": [12, 14, 16]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "lm": {"left": [11, 13, 15], "right": [12, 14, 16]}},
    ],
    "cadera": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
        {"id": "abduccion", "label": "Abducción", "view": "frontal", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
        {"id": "aduccion", "label": "Aducción", "view": "frontal", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
    ],
    "rodilla": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "lm": {"left": [23, 25, 27], "right": [24, 26, 28]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "lm": {"left": [23, 25, 27], "right": [24, 26, 28]}},
    ],
    "tobillo": [
        {"id": "dorsiflexion", "label": "Dorsiflexión", "view": "lateral", "lm": {"left": [25, 27, 31], "right": [26, 28, 32]}},
        {"id": "plantiflexion", "label": "Plantiflexión", "view": "lateral", "lm": {"left": [25, 27, 31], "right": [26, 28, 32]}},
    ],
}

# Talón de cada lado: el tobillo se mide con el eje talón -> punta del pie
HEEL_IDX = {"left": 29, "right": 30}

# --- Parámetros ajustables (candidatos a optimizar en la fase piloto) ---
VISIBILITY_MIN = 0.65
SMOOTH_WINDOW = 5        # mediana móvil (más robusta a frames atípicos que el promedio)
DETECTION_CONF = 0.6
PRESENCE_CONF = 0.6
TRACKING_CONF = 0.7
FALLBACK_FPS = 30        # solo para leer/escribir el video, NO para velocidad
MAX_GAP_FACTOR = 2.5     # ignora velocidades donde hay huecos de datos grandes
MIN_USABLE_PCT = 70      # bajo este % de frames útiles se advierte

# --- Calidad del video de salida ---
OUTPUT_MAX_SIDE = 1280    # lado mayor máximo en píxeles (720p ~1280). Sube a 1920 para Full HD
OUTPUT_CRF = "20"         # 18 = casi sin pérdida, 23 = calidad media, 28 = baja (más liviano)
OUTPUT_PRESET = "veryfast"  # "ultrafast" procesa más rápido pero con peor compresión

# --- Detección de vista y dirección ---
VIEW_FRONTAL_MIN = 0.60   # ancho de hombros / largo de tronco: >= esto => vista frontal
VIEW_LATERAL_MAX = 0.30   # <= esto => vista lateral (entre ambos = oblicua, se descarta)
DIR_TOL = 3.0             # grados de tolerancia al decidir la dirección del movimiento
FACING_WINDOW = 15        # frames usados para estabilizar hacia dónde mira el paciente

# --- Privacidad: difuminado de rostro ---
FACE_LM_IDX = list(range(0, 11))  # nariz, ojos, orejas y boca (MediaPipe Pose 0-10)
FACE_BLUR_MARGIN = 1.7     # margen extra alrededor de los puntos faciales detectados
FACE_BLUR_MIN_RADIUS = 18  # radio mínimo en píxeles, para que no desaparezca si la cara es chica
FACE_HOLD_SEC = 0.5        # si se pierde la detección, se sigue difuminando la última posición este tiempo
                           # (pasado ese tiempo se difumina el frame completo)

SEGMENT_PALETTE = [
    "#1e78ff", "#32cd32", "#ff8c00", "#c71585", "#8a2be2", "#008b8b",
    "#d2691e", "#dc143c", "#556b2f", "#4682b4", "#b8860b", "#708090",
]

SIDE_LABELS = {"Izquierdo": "left", "Derecho": "right"}
COL_MOV, COL_SIDE = "Movimiento", "Lado"


def movement_label(bp, mov_id):
    m = next(m for m in MOVEMENTS[bp] if m["id"] == mov_id)
    return f"{BODY_PART_LABELS[bp]} - {m['label']}"


# "Hombro - Flexión" -> ("hombro", "flexion")
MOVE_LABEL_TO_KEY = {
    movement_label(bp, m["id"]): (bp, m["id"])
    for bp in MOVEMENTS for m in MOVEMENTS[bp]
}


def get_movement(bp, mov_id):
    return next(m for m in MOVEMENTS[bp] if m["id"] == mov_id)

# ----------------------------------------------------------------------------
# 2. Geometría, dibujo, privacidad y velocidad angular
# ----------------------------------------------------------------------------

def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def ascii_text(s):
    """OpenCV no dibuja tildes ni emojis: se limpia el texto del overlay."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def landmark_visibility(p):
    """Visibilidad del punto; si no viene informada se asume 1.0 (un 0 real NO se convierte en 1)."""
    v = getattr(p, "visibility", None)
    return 1.0 if v is None else v


def vec_angle(v1, v2):
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 == 0 or m2 == 0:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0))))


def angle_between(a, b, c):
    return vec_angle((a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1]))


def draw_scale(frame):
    """Factor de escala para que el dibujo se vea bien a cualquier resolución."""
    return min(3.0, max(0.9, min(frame.shape[:2]) / 720.0))


def put_text(frame, text, org, font_scale, color, sc):
    """Texto con contorno oscuro para que se lea sobre cualquier fondo."""
    th = max(2, int(2 * sc))
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale * sc, (20, 20, 20), th + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale * sc, color, th, cv2.LINE_AA)


def output_size(orig_w, orig_h):
    s = min(1.0, OUTPUT_MAX_SIDE / max(orig_w, orig_h))
    w, h = int(orig_w * s), int(orig_h * s)
    return w - (w % 2), h - (h % 2)


def draw_angle_arc(frame, p1, p2, p3, color, radius=35, thickness=2, scale=1.0):
    radius = int(radius * scale)
    thickness = max(2, int(thickness * scale))
    angle1 = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
    angle2 = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]))
    if angle1 < 0:
        angle1 += 360
    if angle2 < 0:
        angle2 += 360

    a_min, a_max = min(angle1, angle2), max(angle1, angle2)
    if a_max - a_min > 180:
        start_angle, end_angle = a_max, a_min + 360
    else:
        start_angle, end_angle = a_min, a_max

    cv2.ellipse(frame, (int(p2[0]), int(p2[1])), (radius, radius), 0, start_angle, end_angle, color, thickness)


def face_region(lm, w, h):
    """
    Región facial (ojos, nariz, orejas, boca) a difuminar, como (x0, y0, x1, y1) en píxeles.
    Usa el ancho de hombros como referencia de escala, más estable que la
    distancia entre ojos cuando el paciente mira de perfil.
    """
    xs = [lm[i].x * w for i in FACE_LM_IDX]
    ys = [lm[i].y * h for i in FACE_LM_IDX]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    shoulder_w = math.hypot((lm[11].x - lm[12].x) * w, (lm[11].y - lm[12].y) * h)
    face_span = max(max(xs) - min(xs), max(ys) - min(ys))
    radius = max(FACE_BLUR_MIN_RADIUS, shoulder_w * 0.5, face_span * FACE_BLUR_MARGIN)

    x0, y0 = max(0, int(cx - radius)), max(0, int(cy - radius))
    x1, y1 = min(w, int(cx + radius)), min(h, int(cy + radius))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return (x0, y0, x1, y1)


def apply_face_blur(frame, region):
    """Difumina de forma irreversible la región facial dentro del frame."""
    x0, y0, x1, y1 = region
    roi = frame[y0:y1, x0:x1]
    k = int(min(roi.shape[:2]) / 1.8)
    k = max(15, k + (1 - k % 2))  # impar y con un mínimo razonable
    blurred = cv2.GaussianBlur(roi, (k, k), 0)

    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    rh, rw = roi.shape[:2]
    cv2.ellipse(mask, (rw // 2, rh // 2), (rw // 2, rh // 2), 0, 0, 360, 255, -1)
    mask3 = cv2.merge([mask, mask, mask])
    frame[y0:y1, x0:x1] = np.where(mask3 > 0, blurred, roi)


def blur_whole_frame(frame):
    """Última defensa de privacidad: sin detección reciente no se puede ubicar el rostro."""
    h, w = frame.shape[:2]
    frame[:] = cv2.GaussianBlur(frame, (0, 0), max(h, w) / 12.0)


def pick_main_person(pose_landmarks_list):
    if len(pose_landmarks_list) <= 1:
        return pose_landmarks_list[0] if pose_landmarks_list else None
    best, best_area = None, -1
    for lm in pose_landmarks_list:
        xs, ys = [p.x for p in lm], [p.y for p in lm]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best, best_area = lm, area
    return best


def compute_angular_velocities(series, sample_period):
    """
    Velocidad angular por DIFERENCIA CENTRAL sobre una serie [(t, ángulo), ...]:
        w_i = |theta_{i+1} - theta_{i-1}| / (t_{i+1} - t_{i-1})

    Se descartan puntos con huecos de datos mayores a MAX_GAP_FACTOR veces el
    intervalo nominal (frames excluidos por baja confianza).
    """
    vels = []
    if sample_period is None:
        return vels
    nominal_dt = 2 * sample_period
    for i in range(1, len(series) - 1):
        dt = series[i + 1][0] - series[i - 1][0]
        if dt <= 0 or dt > MAX_GAP_FACTOR * nominal_dt:
            continue
        vels.append(abs(series[i + 1][1] - series[i - 1][1]) / dt)
    return vels


def frame_time(cap, frame_idx, video_fps, last_t):
    """
    Tiempo real del frame recién leído (en segundos). Los celulares suelen grabar con FPS
    variable, así que frame/fps acumula error y distorsiona la velocidad. Se usa el
    timestamp del contenedor y, si no es confiable, el cálculo frame/fps.
    Siempre devuelve valores estrictamente crecientes (lo exige MediaPipe en modo VIDEO).
    """
    fallback = frame_idx / video_fps
    ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    t = ms / 1000.0 if ms and ms > 0 else None
    if t is None or abs(t - fallback) > 2.0:
        t = fallback
    if t <= last_t:
        t = last_t + 0.001
    return t

# ----------------------------------------------------------------------------
# 3. Modelo MediaPipe
# ----------------------------------------------------------------------------

def download_model_if_needed():
    model_dir = os.path.join(tempfile.gettempdir(), "mediapipe_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "pose_landmarker_full.task")
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
    return model_path


def create_landmarker():
    model_path = download_model_if_needed()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=DETECTION_CONF,
        min_pose_presence_confidence=PRESENCE_CONF,
        min_tracking_confidence=TRACKING_CONF,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)

# ----------------------------------------------------------------------------
# 4. Motor de análisis (usado por los dos modos)
# ----------------------------------------------------------------------------
# (articulación, movimiento) -> (vista requerida, dirección +1 / -1)
# Muñeca y rotaciones de hombro/cadera no están disponibles: son difíciles de
# medir de forma confiable con un solo video 2D (dependen de la orientación de
# la palma o de posiciones 90-90 que no se pueden validar bien con una cámara).
AUTO_MOVEMENTS = {
    ("hombro", "flexion"): ("lateral", +1), ("hombro", "extension"): ("lateral", -1),
    ("hombro", "abduccion"): ("frontal", +1), ("hombro", "aduccion"): ("frontal", -1),
    ("codo", "flexion"): ("lateral", +1), ("codo", "extension"): ("lateral", -1),
    ("cadera", "flexion"): ("lateral", +1), ("cadera", "extension"): ("lateral", -1),
    ("cadera", "abduccion"): ("frontal", +1), ("cadera", "aduccion"): ("frontal", -1),
    ("rodilla", "flexion"): ("lateral", +1), ("rodilla", "extension"): ("lateral", -1),
    ("tobillo", "dorsiflexion"): ("lateral", +1), ("tobillo", "plantiflexion"): ("lateral", -1),
}
AUTO_JOINTS = ["hombro", "codo", "cadera", "rodilla", "tobillo"]


def dir_name(bp, view, direction):
    for (b, mid), (v, d) in AUTO_MOVEMENTS.items():
        if b == bp and v == view and d == direction:
            return get_movement(bp, mid)["label"]
    return ""


def detect_view(lm, w, h):
    """Vista según cuánto se separan los hombros respecto al largo del tronco."""
    sh = math.hypot((lm[11].x - lm[12].x) * w, (lm[11].y - lm[12].y) * h)
    mid_sh = ((lm[11].x + lm[12].x) / 2 * w, (lm[11].y + lm[12].y) / 2 * h)
    mid_hip = ((lm[23].x + lm[24].x) / 2 * w, (lm[23].y + lm[24].y) / 2 * h)
    torso = math.hypot(mid_sh[0] - mid_hip[0], mid_sh[1] - mid_hip[1])
    if torso <= 0:
        return None
    ratio = sh / torso
    if ratio >= VIEW_FRONTAL_MIN:
        return "frontal"
    if ratio <= VIEW_LATERAL_MAX:
        return "lateral"
    return None


def estimate_facing(lm, w):
    """+1 si el paciente mira hacia la derecha de la imagen, -1 a la izquierda."""
    toe = (lm[31].x + lm[32].x) / 2
    heel = (lm[29].x + lm[30].x) / 2
    d = (toe - heel) * w
    if abs(d) < 3:  # pies poco informativos: usar nariz vs hombros
        d = (lm[0].x - (lm[11].x + lm[12].x) / 2) * w
    if d == 0:
        return 0
    return 1 if d > 0 else -1


def signed_angle(bp, view, pts, f, mid_sh_x, mid_hip_x, heel=None):
    """
    Ángulo con signo. Positivo = flexión (o dorsiflexión) en vista lateral y
    abducción en vista frontal; negativo = extensión (plantiflexión) / aducción.

    Tobillo: si se entrega `heel`, se mide el ángulo entre la pierna (tobillo -> rodilla)
    y el eje del pie (talón -> punta). Medir rodilla-tobillo-punta sesga el resultado
    ~25-30° hacia plantiflexión, porque el tobillo (maléolo) queda sobre el piso.
    """
    a, b, c = pts

    if bp == "tobillo" and view == "lateral":
        if heel is not None:
            shank = (a[0] - b[0], a[1] - b[1])
            foot = (c[0] - heel[0], c[1] - heel[1])
            ang = vec_angle(shank, foot)
        else:
            ang = angle_between(a, b, c)
        return None if ang is None else 90.0 - ang

    raw = angle_between(a, b, c)
    if raw is None:
        return None

    if view == "lateral":
        mag = raw if bp == "hombro" else abs(180.0 - raw)
        if bp == "hombro":
            u = (a[0] - b[0], a[1] - b[1])      # tronco hacia abajo
        else:
            u = (b[0] - a[0], b[1] - a[1])      # segmento proximal
        wv = (c[0] - b[0], c[1] - b[1])          # segmento distal
        cross = u[0] * wv[1] - u[1] * wv[0]
        s = float(np.sign(cross)) * f
        if bp != "rodilla":  # en rodilla la flexión lleva la pierna hacia atrás
            s = -s
        return mag * (s if s != 0 else 1.0)

    # Vista frontal: abducción = alejarse de la línea media del cuerpo
    mag = raw if bp == "hombro" else abs(180.0 - raw)
    mid_x = mid_sh_x if bp == "hombro" else mid_hip_x
    d = c[0] - b[0]
    o = b[0] - mid_x
    return mag if d * o >= 0 else -mag


def build_auto_groups(selection, sides, color_offset=0):
    """
    selection: {articulación: [ids de movimiento]}; sides: ["left", "right"].
    Agrupa por (articulación, lado, vista): flexión y extensión comparten un
    único ángulo con signo.
    """
    groups, mov_count = [], color_offset
    for bp, mov_ids in selection.items():
        for side in sides:
            for view in ("lateral", "frontal"):
                movs = [m for m in mov_ids if AUTO_MOVEMENTS[(bp, m)][0] == view]
                if not movs:
                    continue
                ref = next(m for m in MOVEMENTS[bp] if m["view"] == view)
                side_label = "Izquierdo" if side == "left" else "Derecho"
                group = {
                    "gid": len(groups), "bp": bp, "view": view, "side": side,
                    "side_label": side_label, "lm_idx": ref["lm"][side],
                    "buffer": [], "movs": [],
                    "label": f"{BODY_PART_LABELS[bp]} {side_label[:3]}.",
                }
                for mid in movs:
                    color_hex = SEGMENT_PALETTE[mov_count % len(SEGMENT_PALETTE)]
                    mov_count += 1
                    group["movs"].append({
                        "mov_id": mid,
                        "dir": AUTO_MOVEMENTS[(bp, mid)][1],
                        "label": movement_label(bp, mid),
                        "color_hex": color_hex,
                        "color_bgr": hex_to_bgr(color_hex),
                        "series": [], "sampled": 0, "valid": 0,
                    })
                group["color_bgr"] = group["movs"][0]["color_bgr"]
                groups.append(group)
    return groups


def group_points(lm, g, w, h):
    """Puntos del grupo en píxeles. Para el tobillo agrega el talón (eje del pie)."""
    pts_norm = [lm[i] for i in g["lm_idx"]]
    pts_px = [(p.x * w, p.y * h) for p in pts_norm]
    heel_px = None
    if g["bp"] == "tobillo":
        hp = lm[HEEL_IDX[g["side"]]]
        pts_norm = pts_norm + [hp]
        heel_px = (hp.x * w, hp.y * h)
    return pts_norm, pts_px, heel_px


def analyze_frame_auto(frame, landmarker, groups, timestamp_ms, t_now, view_mode, state, blur_face=True):
    """Devuelve (frame dibujado, {gid: ángulo con signo suavizado o None})."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    record = {}
    if not result.pose_landmarks:
        # Sin detección: se corta el suavizado y NO se deja el rostro a la vista
        for g in groups:
            g["buffer"].clear()
        if blur_face:
            region = state.get("face_region")
            if region is not None and t_now - state["face_t"] <= FACE_HOLD_SEC:
                apply_face_blur(frame, region)
            else:
                blur_whole_frame(frame)
        return frame, record

    h, w = frame.shape[:2]
    sc = draw_scale(frame)
    lm = pick_main_person(result.pose_landmarks)

    if blur_face:
        region = face_region(lm, w, h)
        if region is not None:
            state["face_region"], state["face_t"] = region, t_now
            apply_face_blur(frame, region)

    for i1, i2 in POSE_CONNECTIONS:
        p1, p2 = lm[i1], lm[i2]
        cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (150, 150, 150), max(2, int(2 * sc)))

    view = view_mode if view_mode in ("lateral", "frontal") else detect_view(lm, w, h)
    put_text(frame, f"Vista: {view or 'oblicua (sin medir)'}", (int(20 * sc), int(34 * sc)), 0.7, (255, 255, 255), sc)

    # De frente los hombros se separan y la inclinación es medible; de perfil se superponen
    # (dx ~ 0) y el ángulo sale aleatorio, por eso el aviso solo se muestra en vista frontal.
    if view == "frontal":
        dx, dy = (lm[12].x - lm[11].x) * w, (lm[12].y - lm[11].y) * h
        tilt_angle = abs(math.degrees(math.atan2(dy, dx)))
        if 8 < tilt_angle < 172:
            put_text(frame, "Camara inclinada", (int(20 * sc), h - int(20 * sc)), 0.8, (0, 0, 255), sc)

    if view == "lateral":
        est = estimate_facing(lm, w)
        if est != 0:
            state["f_votes"] = (state["f_votes"] + [est])[-FACING_WINDOW:]
    f = 1 if sum(state["f_votes"]) >= 0 else -1

    mid_sh_x = (lm[11].x + lm[12].x) / 2 * w
    mid_hip_x = (lm[23].x + lm[24].x) / 2 * w

    k = 0
    for g in groups:
        if view is None or g["view"] != view:
            g["buffer"].clear()
            continue
        pts_norm, pts_px, heel_px = group_points(lm, g, w, h)
        low_conf = any(landmark_visibility(p) < VISIBILITY_MIN for p in pts_norm)
        color = (60, 70, 226) if low_conf else g["color_bgr"]

        signed = signed_angle(g["bp"], view, pts_px, f, mid_sh_x, mid_hip_x, heel_px)

        arc_pts = pts_px
        if heel_px is not None:  # el arco se dibuja entre la pierna y el eje del pie
            arc_pts = [pts_px[0], pts_px[1],
                       (pts_px[1][0] + pts_px[2][0] - heel_px[0], pts_px[1][1] + pts_px[2][1] - heel_px[1])]
        draw_angle_arc(frame, arc_pts[0], arc_pts[1], arc_pts[2], color, scale=sc)
        for p in pts_px:
            cv2.circle(frame, (int(p[0]), int(p[1])), max(5, int(6 * sc)), color, -1)

        smoothed = None
        if signed is not None and not low_conf:
            buf = g["buffer"]
            buf.append(signed)
            if len(buf) > SMOOTH_WINDOW:
                buf.pop(0)
            smoothed = float(np.median(buf))
            vx, vy = pts_px[1]
            put_text(frame, f"{abs(smoothed):.0f}", (int(vx + 15 * sc), int(vy + (-15 + 32 * k) * sc)), 1.0, color, sc)
        else:
            g["buffer"].clear()  # hueco de datos: no mezclar valores viejos con nuevos
        record[g["gid"]] = smoothed
        k += 1

    return frame, record


def open_writer(path, fps, w, h):
    container = av.open(path, mode="w")
    stream = container.add_stream("h264", rate=Fraction(fps).limit_denominator(1001))
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    stream.options = {"preset": OUTPUT_PRESET, "crf": OUTPUT_CRF}
    return container, stream


def process_video_auto(video_path, groups, target_fps, preview_placeholder, start_sec, end_sec, view_mode,
                       blur_face=True, message="⏳ Procesando evaluación... "):
    """
    Procesa el video UNA sola vez. Devuelve (bytes del video anotado o None, metadatos).
    Los tiempos son los reales de cada frame (robusto a FPS variable).
    """
    landmarker = create_landmarker()
    cap = cv2.VideoCapture(video_path)

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_known = bool(raw_fps and raw_fps > 0)
    video_fps = raw_fps if fps_known else FALLBACK_FPS
    frame_step = max(1, round(video_fps / target_fps))
    sample_period = (frame_step / video_fps) if fps_known else None

    start_frame, end_frame = int(start_sec * video_fps), int(end_sec * video_fps)
    total_frames = end_frame - start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_video_path = out_tfile.name
    out_tfile.close()

    container = stream = None
    new_w = new_h = None
    state = {"f_votes": [], "face_region": None, "face_t": -1.0}
    frames_processed, last_t, last_ms = 0, -1.0, -1
    wrote_any = False
    preview_placeholder.info(message)

    try:
        while frames_processed <= total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Las dimensiones se toman del frame real (los videos verticales de celular
            # pueden informar ancho/alto sin la rotación y salir deformados).
            if container is None:
                new_w, new_h = output_size(frame.shape[1], frame.shape[0])
                container, stream = open_writer(output_video_path, video_fps, new_w, new_h)

            t_now = frame_time(cap, start_frame + frames_processed, video_fps, last_t)
            last_t = t_now
            ts_ms = max(int(t_now * 1000), last_ms + 1)
            last_ms = ts_ms

            frame, record = analyze_frame_auto(
                frame, landmarker, groups, ts_ms, t_now, view_mode, state, blur_face=blur_face
            )

            if frames_processed % frame_step == 0:
                for g in groups:
                    if g["gid"] not in record:
                        continue  # la vista requerida no estaba presente en este frame
                    v = record[g["gid"]]
                    for mv in g["movs"]:
                        mv["sampled"] += 1
                        if v is None:
                            continue  # baja confianza
                        mv["valid"] += 1
                        value = mv["dir"] * v
                        if value >= -DIR_TOL:  # el paciente va en la dirección de este movimiento
                            mv["series"].append((t_now, value if value > 0 else 0.0))

            frame_to_write = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            av_frame = av.VideoFrame.from_ndarray(frame_to_write, format="bgr24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
            wrote_any = True
            frames_processed += 1

        if container is not None:
            for packet in stream.encode():
                container.mux(packet)
    except Exception:
        try:
            os.remove(output_video_path)
        except Exception:
            pass
        raise
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        cap.release()
        try:
            landmarker.close()
        except Exception:
            pass

    video_bytes = None
    if wrote_any:
        with open(output_video_path, "rb") as f:
            video_bytes = f.read()
    try:
        os.remove(output_video_path)
    except Exception:
        pass

    preview_placeholder.empty()
    return video_bytes, {"fps_known": fps_known, "sample_period": sample_period}


def summarize_auto(groups, sample_period, start_sec, end_sec):
    results = []
    for g in groups:
        for mv in g["movs"]:
            series = mv["series"]
            norm_min, norm_max = NORMATIVE_RANGES[g["bp"]][mv["mov_id"]]
            vels = compute_angular_velocities(series, sample_period)
            results.append({
                "title": mv["label"],
                "side": g["side_label"],
                "start": start_sec,
                "end": end_sec,
                "max": round(max(a for _, a in series), 1) if series else None,
                "norm_min": norm_min,
                "norm_max": norm_max,
                "vel": round(max(vels), 1) if vels else None,
                "vel_p95": round(float(np.percentile(vels, 95)), 1) if vels else None,
                "sampled": mv["sampled"],
                "valid": mv["valid"],
                "series": series,
                "color_hex": mv["color_hex"],
                "view": g["view"],
                "shade": False,
            })
    return results

# ----------------------------------------------------------------------------
# 5. Gráficos, RUN y PDF
# ----------------------------------------------------------------------------

def make_protocol_chart(results):
    """Un gráfico independiente por movimiento (cuadrícula de 2 columnas)."""
    n = max(1, len(results))
    ncols = 1 if n == 1 else 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 3.0 * nrows), squeeze=False)
    flat = axes.flatten()

    for ax, r in zip(flat, results):
        side_short = "Izq" if r["side"] == "Izquierdo" else "Der"
        ax.set_title(f"{r['title']} ({side_short})", fontsize=10, fontweight="bold", color=r["color_hex"])
        ax.grid(alpha=0.3)
        ax.set_xlabel("Tiempo (s)", fontsize=8)
        ax.set_ylabel("Ángulo (°)", fontsize=8)
        ax.tick_params(labelsize=8)

        top = r["norm_max"]
        if r["series"]:
            ts = [p[0] for p in r["series"]]
            angs = [p[1] for p in r["series"]]
            ax.plot(ts, angs, color=r["color_hex"], linewidth=2)
            ax.axhline(r["max"], color=r["color_hex"], linestyle="--", linewidth=1, alpha=0.7,
                       label=f"Máximo: {r['max']:.0f}°")
            top = max(top, r["max"])
        else:
            ax.text(0.5, 0.5, "Sin datos suficientes", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray")

        ax.axhline(r["norm_max"], color="gray", linestyle=":", linewidth=1,
                   label=f"Rango AAOS máx: {r['norm_max']}°")
        ax.set_ylim(0, max(10, top * 1.15))
        ax.legend(fontsize=7, loc="upper right", frameon=False)

    for ax in flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def fmt_vel(v):
    return "N/D" if v is None else f"{v:.1f} °/s"


def fmt_angle(v):
    return "N/D" if v is None else f"{v:.1f}°"


def run_valido(run):
    """Valida un RUN chileno con su dígito verificador (módulo 11)."""
    run = run.replace(".", "").replace("-", "").upper().strip()
    if len(run) < 2 or not run[:-1].isdigit():
        return False
    s, m = 0, 2
    for d in reversed(run[:-1]):
        s += int(d) * m
        m = 2 if m == 7 else m + 1
    r = 11 - s % 11
    return run[-1] == ("0" if r == 11 else "K" if r == 10 else str(r))


def format_run(run):
    clean = run.replace(".", "").replace("-", "").upper().strip()
    return f"{int(clean[:-1]):,}".replace(",", ".") + "-" + clean[-1]


_PDF_REPLACEMENTS = {
    "“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "...", "•": "-", "\u00a0": " ",
}


def pdf_safe(s):
    """Helvetica de FPDF solo admite latin-1: se reemplazan comillas/guiones tipográficos (p. ej. al pegar desde Word)."""
    s = "" if s is None else str(s)
    for a, b in _PDF_REPLACEMENTS.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


def patient_key(s):
    return s["run"] if s["run"] != "-" else s["patient"]


def build_pdf_report(sessions):
    """Informe de UN paciente: el llamador debe pasar solo las sesiones de ese paciente."""
    T = pdf_safe
    last = sessions[-1]
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 110, 86)
    pdf.cell(0, 12, "Informe de movilidad articular", ln=True)
    pdf.set_draw_color(15, 110, 86)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 100, 105)
    pdf.multi_cell(0, 6, T(f"Paciente: {last['patient']}   |   RUN: {last['run']}   |   Generado: {date.today().isoformat()}"))
    pdf.ln(4)

    widths = [70, 25, 30, 35, 30]
    headers = ["Movimiento", "Lado", "Máximo", "Rango AAOS", "Velocidad"]

    for s in sessions:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(28, 43, 48)
        pdf.cell(0, 9, T(f"Evaluación - {s['date']}  ({len(s['results'])} movimientos)"), ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(90, 100, 105)
        pdf.cell(0, 6, T(f"Evaluado por: {s.get('evaluador', '-')}"), ln=True)
        pdf.set_text_color(28, 43, 48)

        # El gráfico crece con la cantidad de movimientos: se limita para que quepa en una página
        iw, ih = Image.open(io.BytesIO(s["chart_png"])).size
        img_w = 160.0
        max_h = pdf.h - 45
        if img_w * ih / iw > max_h:
            img_w = max_h * iw / ih
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(s["chart_png"])
            tmp_img_path = tmp_img.name
        pdf.image(tmp_img_path, w=img_w)
        pdf.ln(2)
        try:
            os.remove(tmp_img_path)
        except Exception:
            pass

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(225, 245, 238)
        pdf.set_text_color(28, 43, 48)
        for wdt, htxt in zip(widths, headers):
            pdf.cell(wdt, 7, T(htxt), border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for r in s["results"]:
            row = [
                r["title"],
                r["side"],
                fmt_angle(r["max"]),
                f"{r['norm_min']}° - {r['norm_max']}°",
                fmt_vel(r.get("vel_p95")),
            ]
            for wdt, txt in zip(widths, row):
                pdf.cell(wdt, 7, T(txt), border=1)
            pdf.ln()
        pdf.ln(4)

        if s.get("diagnostico"):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(28, 43, 48)
            pdf.multi_cell(0, 6, T(f"Impresión Diagnóstica: {s['diagnostico']}"))

        if s.get("observaciones"):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(28, 43, 48)
            pdf.cell(0, 6, "Observaciones Clínicas:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, T(s["observaciones"]))

        pdf.ln(6)

    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1")
    return bytes(out)


def sessions_to_dataframe(sessions):
    rows = []
    for s in sessions:
        for r in s["results"]:
            rows.append({
                "Paciente": s["patient"],
                "RUN": s["run"],
                "Fecha": s["date"],
                "Evaluado por": s.get("evaluador", "-"),
                "Movimiento": r["title"],
                "Lado": r["side"],
                "Máximo (°)": r["max"],
                "Rango AAOS (°)": f"{r['norm_min']} - {r['norm_max']}",
                "Velocidad (°/s)": r.get("vel_p95"),
                "Diagnóstico": s.get("diagnostico", ""),
                "Observaciones": s.get("observaciones", ""),
            })
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------
# 6. UI Streamlit
# ----------------------------------------------------------------------------
st.set_page_config(page_title="M.M. - MotionMetrics", page_icon="📐", layout="centered")
st.title("M.M. - MotionMetrics 📐")
st.caption("Análisis cinemático, goniometría 2D y velocidad angular.")

MODE_AUTO = "Automático: un video, elijo qué evaluar"
MODE_MULTI = "Varios videos: un movimiento por video"
VIEW_MODES = {"Detectar automáticamente": "auto", "Siempre lateral (de perfil)": "lateral", "Siempre frontal (de frente)": "frontal"}
COL_FILE = "Archivo"
MIRROR_NOTE = (
    "Si grabaste con la **cámara frontal** del celular, el video puede quedar en espejo: "
    "izquierda y derecha se invierten. Usa la cámara trasera o desactiva el efecto espejo."
)


def save_video_to_disk(video_bytes):
    """Guarda el video procesado en disco y devuelve la ruta (evita acumular bytes en RAM)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        return tmp.name


def cleanup_multi_videos():
    """Borra los videos procesados de la corrida anterior (modo varios videos)."""
    for path in st.session_state.get("multi_videos", {}).values():
        try:
            os.remove(path)
        except Exception:
            pass
    st.session_state.pop("multi_videos", None)


def store_results(results, video_bytes=None, multi_videos=None):
    if all(r["valid"] == 0 for r in results):
        st.session_state.results = []
        st.error(
            "No hubo frames con confianza suficiente en ningún movimiento. "
            "Revisa la iluminación, el encuadre, la vista de cámara o que las articulaciones estén visibles."
        )
    else:
        st.success("✅ Procesamiento completado.")
        st.session_state.results = results
        st.session_state.pop("video_bytes", None)
        if multi_videos is None:
            cleanup_multi_videos()
        if video_bytes is not None:
            st.session_state.video_bytes = video_bytes
        if multi_videos is not None:
            st.session_state.multi_videos = multi_videos


for key, default in [("sessions", []), ("results", []), ("editor_version", 0), ("file_id", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

st.header("1. Privacidad y modo de evaluación")
blur_face = st.checkbox(
    "Difuminar el rostro del paciente en el video de salida", value=True,
    help="Difumina ojos, nariz, orejas y boca en todos los videos generados por la app. "
         "Si se pierde la detección del cuerpo, se sigue difuminando (y pasado medio segundo se difumina todo el frame).",
)
if not blur_face:
    st.warning(
        "El rostro del paciente quedará visible en el video de salida y en cualquier reporte o "
        "descarga que lo incluya. Actívalo salvo que tengas una razón clínica específica para no hacerlo."
    )

mode = st.radio("¿Cómo grabaste al paciente?", [MODE_AUTO, MODE_MULTI])
if mode == MODE_MULTI:
    st.caption("Un video distinto por cada movimiento (por ejemplo, grabaste cada movimiento por separado).")
else:
    st.caption("Un solo video con uno o varios movimientos consecutivos.")

target_fps = st.select_slider(
    "Frecuencia de muestreo (FPS)", options=[10, 15, 30], value=30,
    help="Cuántos puntos por segundo se usan para el gráfico y la velocidad angular. "
         "No acelera el procesamiento: el video se analiza completo de todas formas.",
)
preview_placeholder = st.empty()

# ======================================================================
# MODO VARIOS VIDEOS: un movimiento por archivo
# ======================================================================
if mode == MODE_MULTI:
    st.header("2. Videos")
    st.caption(MIRROR_NOTE)
    multi_files = st.file_uploader(
        "Sube uno o más videos (cada uno con un solo movimiento)",
        type=["mp4", "mov", "avi"], accept_multiple_files=True,
    )

    if multi_files:
        files_id = tuple(f"{f.name}-{f.size}" for f in multi_files)
        if st.session_state.get("multi_files_id") != files_id:
            rows = [{COL_FILE: f.name, COL_MOV: "Hombro - Flexión", COL_SIDE: "Derecho"} for f in multi_files]
            st.session_state.multi_table = pd.DataFrame(rows, columns=[COL_FILE, COL_MOV, COL_SIDE])
            st.session_state.multi_files_id = files_id
            st.session_state.editor_version += 1
            st.session_state.results = []
            st.session_state.pop("video_bytes", None)
            cleanup_multi_videos()

        st.write("Indica qué movimiento y lado corresponde a cada video:")
        edited_multi = st.data_editor(
            st.session_state.multi_table,
            key=f"multi_editor_{st.session_state.editor_version}",
            use_container_width=True, hide_index=True, num_rows="fixed",
            column_config={
                COL_FILE: st.column_config.TextColumn(COL_FILE, disabled=True),
                COL_MOV: st.column_config.SelectboxColumn(COL_MOV, options=list(MOVE_LABEL_TO_KEY.keys()), required=True, width="medium"),
                COL_SIDE: st.column_config.SelectboxColumn(COL_SIDE, options=list(SIDE_LABELS.keys()), required=True),
            },
        )

        if st.button("Procesar videos"):
            cleanup_multi_videos()
            all_results, all_videos = [], {}
            seen_titles = {}
            progress = st.progress(0.0)
            for i, row in edited_multi.reset_index(drop=True).iterrows():
                f = multi_files[i]  # por posición: soporta dos archivos con el mismo nombre
                bp, mov_id = MOVE_LABEL_TO_KEY[row[COL_MOV]]
                side_label = row[COL_SIDE]
                view_req = AUTO_MOVEMENTS[(bp, mov_id)][0]
                video_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(f.getbuffer())
                        video_path = tmp.name

                    cap_t = cv2.VideoCapture(video_path)
                    total_frames = int(cap_t.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps_raw = cap_t.get(cv2.CAP_PROP_FPS)
                    cap_t.release()
                    fps_known = bool(fps_raw and fps_raw > 0)
                    dur = total_frames / (fps_raw if fps_known else FALLBACK_FPS)
                    if dur <= 0:
                        st.warning(f"{f.name}: no se pudo leer el video, se omite.")
                        continue

                    # Mismo motor que el modo automático: ángulo con signo + dirección del movimiento
                    groups = build_auto_groups({bp: [mov_id]}, [SIDE_LABELS[side_label]], color_offset=i)
                    video_bytes, meta = process_video_auto(
                        video_path, groups, target_fps, preview_placeholder, 0.0, dur, view_req,
                        blur_face=blur_face,
                        message=f"⏳ Procesando: {movement_label(bp, mov_id)} ({side_label}) - {f.name}",
                    )
                    result = summarize_auto(groups, meta["sample_period"], 0.0, dur)[0]

                    # Dos videos del mismo movimiento y lado no se pisan entre sí
                    key = (result["title"], result["side"])
                    seen_titles[key] = seen_titles.get(key, 0) + 1
                    if seen_titles[key] > 1:
                        result["title"] = f"{result['title']} (video {seen_titles[key]})"
                    all_results.append(result)

                    if video_bytes:
                        label = f"{result['title']} ({side_label}) - {f.name}"
                        if label in all_videos:
                            label = f"{label} [{i + 1}]"
                        all_videos[label] = save_video_to_disk(video_bytes)
                    del video_bytes
                except Exception as e:
                    st.error(f"{f.name}: error al procesar el video ({e}).")
                finally:
                    if video_path:
                        try:
                            os.remove(video_path)
                        except Exception:
                            pass
                progress.progress((i + 1) / len(edited_multi))

            progress.empty()
            preview_placeholder.empty()
            if all_results:
                store_results(all_results, multi_videos=all_videos)

# ======================================================================
# MODO AUTOMÁTICO: un solo video
# ======================================================================
else:
    st.header("2. Video")
    st.caption(MIRROR_NOTE)
    uploaded_file = st.file_uploader("Sube el video del paciente", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"

        # Solo se escribe/lee el video cuando cambia el archivo (no en cada rerun)
        if st.session_state.file_id != file_id:
            old_path = st.session_state.get("tmp_path")
            if old_path:
                try:
                    os.remove(old_path)
                except Exception:
                    pass
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.getbuffer())
                st.session_state.tmp_path = tmp.name

            cap_temp = cv2.VideoCapture(st.session_state.tmp_path)
            total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_raw = cap_temp.get(cv2.CAP_PROP_FPS)
            cap_temp.release()

            st.session_state.fps_known = bool(fps_raw and fps_raw > 0)
            st.session_state.duration = total_frames / (fps_raw if st.session_state.fps_known else FALLBACK_FPS)
            st.session_state.file_id = file_id
            st.session_state.results = []
            st.session_state.pop("video_bytes", None)
            st.session_state.editor_version += 1

        duration = st.session_state.duration
        tmp_path = st.session_state.tmp_path

        if not st.session_state.fps_known:
            st.warning(
                "No fue posible determinar los FPS del video. El ROM puede analizarse, "
                "pero la velocidad angular no será informada."
            )

        with st.expander("Ver video original"):
            st.video(tmp_path)
        st.caption(f"Duración del video: {duration:.1f} s")

        st.header("3. Qué quieres evaluar")
        st.write(
            "Elige las articulaciones y los movimientos. La app separa cada movimiento por la "
            "dirección del ángulo y por la vista de cámara, sin que tengas que indicar tiempos."
        )
        joint_keys = st.multiselect(
            "Articulaciones", AUTO_JOINTS, default=["hombro", "codo"],
            format_func=lambda k: BODY_PART_LABELS[k],
        )
        selection = {}
        for bp in joint_keys:
            opts = [mid for (b, mid) in AUTO_MOVEMENTS if b == bp]
            chosen = st.multiselect(
                f"Movimientos de {BODY_PART_LABELS[bp].lower()}", opts, default=opts,
                key=f"auto_mov_{bp}",
                format_func=lambda mid, bp=bp: get_movement(bp, mid)["label"],
            )
            if chosen:
                selection[bp] = chosen

        ca, cb = st.columns(2)
        side_choice = ca.selectbox("Lado a evaluar", ["Izquierdo", "Derecho", "Ambos"], index=1)
        view_choice = cb.selectbox("Vista de cámara", list(VIEW_MODES.keys()))
        view_mode = VIEW_MODES[view_choice]

        start_sec, end_sec = st.slider(
            "Recorte del video (opcional)", 0.0, float(duration), (0.0, float(duration)), step=0.1
        )

        needed_views = sorted({AUTO_MOVEMENTS[(bp, m)][0] for bp in selection for m in selection[bp]})
        if view_mode == "auto" and len(needed_views) > 1:
            st.info(
                "Cada movimiento se mide solo en los frames donde la cámara ve la vista adecuada: "
                "**lateral** (de perfil) para flexión/extensión y **frontal** para abducción/aducción. "
                "El paciente debe girarse hacia la posición correcta para cada grupo de movimientos."
            )
        elif view_mode in ("lateral", "frontal"):
            missing = [movement_label(bp, m) for bp in selection for m in selection[bp]
                       if AUTO_MOVEMENTS[(bp, m)][0] != view_mode]
            if missing:
                st.warning(
                    f"Con la vista fija en {view_mode}, estos movimientos no se podrán medir: "
                    + ", ".join(missing)
                )

        if side_choice == "Ambos" and "lateral" in needed_views and view_mode in ("auto", "lateral"):
            st.warning(
                "De perfil, el brazo o la pierna del lado más lejano a la cámara queda tapado por el cuerpo y "
                "MediaPipe lo estima, a veces con buena visibilidad aparente. Para flexión/extensión "
                "es más confiable evaluar un lado por vez."
            )

        if st.button("Procesar evaluación", disabled=not selection):
            sides = ["left", "right"] if side_choice == "Ambos" else [SIDE_LABELS[side_choice]]
            groups = build_auto_groups(selection, sides)
            with st.spinner("Procesando cinemática y velocidad angular..."):
                video_bytes, meta = process_video_auto(
                    tmp_path, groups, target_fps, preview_placeholder, start_sec, end_sec, view_mode,
                    blur_face=blur_face,
                )
                results = summarize_auto(groups, meta["sample_period"], start_sec, end_sec)
            store_results(results, video_bytes=video_bytes)

# ----------------------------------------------------------------------
if st.session_state.results:
    results = st.session_state.results

    if "video_bytes" in st.session_state:
        st.video(st.session_state.video_bytes)
    elif "multi_videos" in st.session_state:
        st.caption("Video procesado por cada movimiento:")
        for label, vpath in st.session_state.multi_videos.items():
            with st.expander(label):
                st.video(vpath)

    st.header("4. Evolución temporal por movimiento")
    fig = make_protocol_chart(results)
    st.pyplot(fig)
    chart_png = fig_to_png_bytes(fig)

    st.header("5. Resultados clínicos")
    results_df = pd.DataFrame([{
        "Movimiento": r["title"],
        "Lado": r["side"],
        "Máximo": fmt_angle(r["max"]),
        "Rango AAOS": f"{r['norm_min']}° - {r['norm_max']}°",
        "Velocidad": fmt_vel(r["vel_p95"]),
    } for r in results])
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    for r in results:
        pct = 100 * r["valid"] / r["sampled"] if r["sampled"] else 0
        if r["sampled"] == 0 and r.get("view"):
            st.warning(
                f"{r['title']} ({r['side']}): no se detectó la vista {r['view']} en el video. "
                "Verifica que el paciente se gire o fija la vista manualmente."
            )
        elif r["valid"] == 0:
            st.warning(f"{r['title']} ({r['side']}): sin frames utilizables.")
        elif r["max"] is None:
            st.warning(f"{r['title']} ({r['side']}): no se detectó movimiento en esa dirección.")
        elif pct < MIN_USABLE_PCT:
            st.warning(f"{r['title']} ({r['side']}): solo {pct:.0f}% de frames utilizables; interpreta con cautela.")

    st.header("6. Registro de sesión y Observaciones")
    p1, p2, p3 = st.columns(3)
    patient_name = p1.text_input("Nombre del paciente")
    patient_run = p2.text_input("RUN", placeholder="12.345.678-9")
    test_date = p3.date_input("Fecha", value=date.today())
    evaluador = st.text_input("Evaluado por", placeholder="Nombre del profesional que evaluó al paciente")

    diagnostico = st.text_input("Impresión Diagnóstica (Opcional)", placeholder="Ej: Limitación funcional leve por probable tendinopatía...")
    observaciones = st.text_area("Observaciones Clínicas", placeholder="Ej: Paciente refiere dolor en los últimos 15° de flexión. Se observan compensaciones musculares...")

    if st.button("Guardar temporalmente"):
        run_in = patient_run.strip()
        if run_in and not run_valido(run_in):
            st.error("El RUN ingresado no es válido (revisa el número y el dígito verificador).")
        else:
            saved_results = [{k: v for k, v in r.items() if k not in ("series", "color_hex", "shade", "view")} for r in results]
            st.session_state.sessions.append({
                "run": format_run(run_in) if run_in else "-",
                "patient": patient_name.strip() or "Sin nombre",
                "date": str(test_date),
                "evaluador": evaluador.strip() or "-",
                "results": saved_results,
                "diagnostico": diagnostico,
                "observaciones": observaciones,
                "chart_png": chart_png,
            })
            st.success("Guardado en la sesión actual.")

if st.session_state.sessions:
    st.subheader("Historial (Se borrará al recargar la página)")
    sessions_df = sessions_to_dataframe(st.session_state.sessions)
    st.dataframe(sessions_df, use_container_width=True)

    # El informe PDF es de UN paciente (antes mezclaba a todos los pacientes guardados)
    labels = {}
    for s in st.session_state.sessions:
        labels[patient_key(s)] = f"{s['patient']} ({s['run']})"
    keys = list(labels.keys())
    last_key = patient_key(st.session_state.sessions[-1])
    sel_key = st.selectbox(
        "Paciente para el informe PDF", keys, index=keys.index(last_key), format_func=lambda k: labels[k]
    )
    pat_sessions = [s for s in st.session_state.sessions if patient_key(s) == sel_key]

    col_pdf, col_xlsx = st.columns(2)
    with col_pdf:
        try:
            pdf_bytes = build_pdf_report(pat_sessions)
            safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_text(pat_sessions[-1]["patient"])).strip("_") or "paciente"
            st.download_button("Descargar informe (PDF)", data=pdf_bytes, file_name=f"informe_{safe_name}.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"No se pudo generar el PDF: {e}")
    with col_xlsx:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            sessions_df.to_excel(writer, index=False, sheet_name="Pruebas")
        st.download_button("Descargar historial (Excel)", data=excel_buffer.getvalue(), file_name="historial_pruebas.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")