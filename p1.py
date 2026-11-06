"""
M.M. - MotionMetrics 
=====================================================================
"""
import io
import math
import os
import tempfile
import unicodedata
import urllib.request
from datetime import date

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
    "muneca": "Muñeca",
    "cadera": "Cadera",
    "rodilla": "Rodilla",
    "tobillo": "Tobillo",
}

NORMATIVE_RANGES = {
    "hombro": {"flexion": (0, 180), "extension": (0, 60), "abduccion": (0, 180), "aduccion": (0, 30), "rot_interna": (0, 70), "rot_externa": (0, 90)},
    "codo": {"flexion": (0, 150), "extension": (0, 10)},
    "muneca": {"flexion": (0, 80), "extension": (0, 70), "desviacion_radial": (0, 20), "desviacion_ulnar": (0, 30)},
    "cadera": {"flexion": (0, 120), "extension": (0, 30), "abduccion": (0, 45), "aduccion": (0, 30), "rot_interna": (0, 45), "rot_externa": (0, 45)},
    "rodilla": {"flexion": (0, 135), "extension": (0, 10)},
    "tobillo": {"dorsiflexion": (0, 20), "plantiflexion": (0, 50)},
}

MOVEMENTS = {
    "hombro": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "mode": "angle_0_rest", "lm": {"left": [23, 11, 13], "right": [24, 12, 14]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "mode": "angle_0_rest", "lm": {"left": [23, 11, 13], "right": [24, 12, 14]}},
        {"id": "abduccion", "label": "Abducción", "view": "frontal", "mode": "angle_0_rest", "lm": {"left": [23, 11, 13], "right": [24, 12, 14]}},
        {"id": "aduccion", "label": "Aducción", "view": "frontal", "mode": "angle_0_rest", "lm": {"left": [23, 11, 13], "right": [24, 12, 14]}},
        {"id": "rot_interna", "label": "Rotación interna", "view": "frontal", "mode": "vertical", "lm": {"left": [13, 15], "right": [14, 16]}},
        {"id": "rot_externa", "label": "Rotación externa", "view": "frontal", "mode": "vertical", "lm": {"left": [13, 15], "right": [14, 16]}},
    ],
    "codo": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [11, 13, 15], "right": [12, 14, 16]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [11, 13, 15], "right": [12, 14, 16]}},
    ],
    "muneca": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [13, 15, 19], "right": [14, 16, 20]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [13, 15, 19], "right": [14, 16, 20]}},
        {"id": "desviacion_radial", "label": "Desviación Radial", "view": "frontal", "mode": "angle_180_rest", "lm": {"left": [13, 15, 19], "right": [14, 16, 20]}},
        {"id": "desviacion_ulnar", "label": "Desviación Ulnar", "view": "frontal", "mode": "angle_180_rest", "lm": {"left": [13, 15, 19], "right": [14, 16, 20]}},
    ],
    "cadera": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
        {"id": "abduccion", "label": "Abducción", "view": "frontal", "mode": "angle_180_rest", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
        {"id": "aduccion", "label": "Aducción", "view": "frontal", "mode": "angle_180_rest", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
        {"id": "rot_interna", "label": "Rotación interna", "view": "frontal", "mode": "vertical", "lm": {"left": [25, 27], "right": [26, 28]}},
        {"id": "rot_externa", "label": "Rotación externa", "view": "frontal", "mode": "vertical", "lm": {"left": [25, 27], "right": [26, 28]}},
    ],
    "rodilla": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [23, 25, 27], "right": [24, 26, 28]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [23, 25, 27], "right": [24, 26, 28]}},
    ],
    "tobillo": [
        {"id": "dorsiflexion", "label": "Dorsiflexión", "view": "lateral", "mode": "angle_90_rest", "lm": {"left": [25, 27, 31], "right": [26, 28, 32]}},
        {"id": "plantiflexion", "label": "Plantiflexión", "view": "lateral", "mode": "angle_90_rest", "lm": {"left": [25, 27, 31], "right": [26, 28, 32]}},
    ],
}

# Protocolo predefinido: todos los movimientos de tren superior
UPPER_LIMB_PROTOCOL = [
    ("hombro", "flexion"), ("hombro", "extension"),
    ("hombro", "abduccion"), ("hombro", "aduccion"),
    ("hombro", "rot_interna"), ("hombro", "rot_externa"),
    ("codo", "flexion"), ("codo", "extension"),
    ("muneca", "flexion"), ("muneca", "extension"),
    ("muneca", "desviacion_radial"), ("muneca", "desviacion_ulnar"),
]

# --- Parámetros ajustables (candidatos a optimizar en la fase piloto) ---
VISIBILITY_MIN = 0.65
SMOOTH_WINDOW = 3
DETECTION_CONF = 0.6
PRESENCE_CONF = 0.6
TRACKING_CONF = 0.7
FALLBACK_FPS = 30        # solo para leer/escribir el video, NO para velocidad
MAX_GAP_FACTOR = 2.5     # ignora velocidades donde hay huecos de datos grandes
MIN_USABLE_PCT = 70      # bajo este % de frames útiles se advierte

# --- Calidad del video de salida ---
OUTPUT_MAX_SIDE = 960    # lado mayor máximo en píxeles (720p ~1280). Sube a 1920 para Full HD
OUTPUT_CRF = "26"         # 18 = casi sin pérdida, 23 = calidad media, 28 = baja (más liviano)
OUTPUT_PRESET = "veryfast"  # "ultrafast" procesa más rápido pero con peor compresión

# --- Límites recomendados para Streamlit Community Cloud ---
MAX_VIDEO_MB = 60
MAX_DURATION_SEC = 45

# --- Modo automático (detección de vista y dirección) ---
VIEW_FRONTAL_MIN = 0.60   # ancho de hombros / largo de tronco: >= esto => vista frontal
VIEW_LATERAL_MAX = 0.30   # <= esto => vista lateral (entre ambos = oblicua, se descarta)
DIR_TOL = 3.0             # grados de tolerancia al decidir la dirección del movimiento
FACING_WINDOW = 15        # frames usados para estabilizar hacia dónde mira el paciente

SEGMENT_PALETTE = [
    "#1e78ff", "#32cd32", "#ff8c00", "#c71585", "#8a2be2", "#008b8b",
    "#d2691e", "#dc143c", "#556b2f", "#4682b4", "#b8860b", "#708090",
]

SIDE_LABELS = {"Izquierdo": "left", "Derecho": "right"}
COL_MOV, COL_SIDE, COL_START, COL_END = "Movimiento", "Lado", "Inicio (s)", "Fin (s)"


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
# 2. Geometría, dibujo y velocidad angular
# ----------------------------------------------------------------------------

def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def ascii_text(s):
    """OpenCV no dibuja tildes ni emojis: se limpia el texto del overlay."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def angle_between(a, b, c):
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])
    mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if mag1 == 0 or mag2 == 0:
        return None
    cos_angle = np.clip(np.dot(v1, v2) / (mag1 * mag2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def angle_from_vertical(vertex, point):
    v = np.array([point[0] - vertex[0], point[1] - vertex[1]])
    mag = np.linalg.norm(v)
    if mag == 0:
        return None
    up = np.array([0, -1])
    cos_angle = np.clip(np.dot(v, up) / mag, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


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

# ----------------------------------------------------------------------------
# 3. Procesamiento y MediaPipe
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


def analyze_frame_segments(frame, landmarker, active_segments, timestamp_ms):
    """
    Analiza un frame para los tramos activos en ese instante.
    Devuelve (frame_dibujado, {idx_tramo: ángulo suavizado o None}).
    Un ángulo de baja confianza devuelve None (no entra a resultados ni suavizado).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    angles_out = {seg["idx"]: None for seg in active_segments}
    if not result.pose_landmarks:
        return frame, angles_out

    h, w = frame.shape[:2]
    sc = draw_scale(frame)
    lm = pick_main_person(result.pose_landmarks)

    p_shoulder_l, p_shoulder_r = lm[11], lm[12]
    dx, dy = (p_shoulder_r.x - p_shoulder_l.x) * w, (p_shoulder_r.y - p_shoulder_l.y) * h
    tilt_angle = abs(math.degrees(math.atan2(dy, dx)))
    if 8 < tilt_angle < 172:
        put_text(frame, "Camara inclinada", (int(20 * sc), h - int(20 * sc)), 0.8, (0, 0, 255), sc)

    for i1, i2 in POSE_CONNECTIONS:
        p1, p2 = lm[i1], lm[i2]
        cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (150, 150, 150), max(2, int(2 * sc)))

    for k, seg in enumerate(active_segments):
        movement, side, smooth_buffer = seg["mov"], seg["side"], seg["buffer"]
        base_color = seg["color_bgr"]
        pts_norm = [lm[i] for i in movement["lm"][side]]
        pts_px = [(p.x * w, p.y * h) for p in pts_norm]
        low_conf = any((getattr(p, "visibility", 1.0) or 1.0) < VISIBILITY_MIN for p in pts_norm)
        color = (60, 70, 226) if low_conf else base_color

        # Etiqueta del movimiento en curso (arriba a la izquierda)
        put_text(frame, ascii_text(seg["label"]), (int(20 * sc), int((34 + 30 * k) * sc)), 0.7, color, sc)

        if movement["mode"] == "vertical":
            vertex, point = pts_px
            angle = angle_from_vertical(vertex, point)
            if angle is not None:
                ref_pt = (vertex[0], vertex[1] + 50 * sc)
                draw_angle_arc(frame, point, vertex, ref_pt, color, scale=sc)
        else:
            a, b, c = pts_px
            vertex = b
            raw_angle = angle_between(a, b, c)
            angle = None
            if raw_angle is not None:
                if movement["mode"] == "angle_0_rest":
                    angle = raw_angle
                elif movement["mode"] == "angle_180_rest":
                    angle = abs(180.0 - raw_angle)
                elif movement["mode"] == "angle_90_rest":
                    angle = abs(raw_angle - 90.0)
                draw_angle_arc(frame, a, b, c, color, scale=sc)

        for p in pts_px:
            cv2.circle(frame, (int(p[0]), int(p[1])), max(5, int(6 * sc)), color, -1)

        if angle is not None and not low_conf:
            smooth_buffer.append(angle)
            if len(smooth_buffer) > SMOOTH_WINDOW:
                smooth_buffer.pop(0)
            smoothed = sum(smooth_buffer) / len(smooth_buffer)
            text_org = (int(vertex[0] + 15 * sc), int(vertex[1] + (-15 + 32 * k) * sc))
            put_text(frame, f"{smoothed:.0f}", text_org, 1.0, color, sc)
            angles_out[seg["idx"]] = smoothed

    return frame, angles_out


def process_video_protocol(video_path, segments, target_fps, preview_placeholder):
    """
    Procesa el video UNA sola vez. Cada tramo acumula su propia serie
    [(t, ángulo)] únicamente con frames válidos dentro de su ventana.
    """
    landmarker = create_landmarker()
    cap = cv2.VideoCapture(video_path)

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_known = bool(raw_fps and raw_fps > 0)
    video_fps = raw_fps if fps_known else FALLBACK_FPS

    frame_step = max(1, round(video_fps / target_fps))
    sample_period = (frame_step / video_fps) if fps_known else None

    # Solo se procesa desde el primer inicio hasta el último fin de los tramos
    start_sec = min(s["start"] for s in segments)
    end_sec = max(s["end"] for s in segments)
    start_frame, end_frame = int(start_sec * video_fps), int(end_sec * video_fps)
    total_frames = end_frame - start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    orig_w, orig_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_w, new_h = output_size(orig_w, orig_h)

    out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_video_path = out_tfile.name
    out_tfile.close()

    container = av.open(output_video_path, mode="w")
    output_fps = max(1, int(round(video_fps / frame_step)))
    stream = container.add_stream("h264", rate=output_fps)
    stream.width, stream.height, stream.pix_fmt = new_w, new_h, "yuv420p"
    stream.options = {"preset": OUTPUT_PRESET, "crf": OUTPUT_CRF}

    for seg in segments:
        seg["buffer"], seg["series"], seg["sampled"] = [], [], 0

    frames_processed = 0
    preview_placeholder.info("⏳ Procesando protocolo... ")

    while frames_processed <= total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t_now = (start_frame + frames_processed) / video_fps
        current_ms = int(t_now * 1000)
        active = [s for s in segments if s["start"] <= t_now <= s["end"]]

        # En Cloud solo se analiza y codifica el frame cuando corresponde al muestreo elegido.
        # Así 10 FPS sobre un video de 30 FPS reduce aproximadamente a 1/3 las inferencias.
        if frames_processed % frame_step == 0:
            if active:
                frame, angles = analyze_frame_segments(frame, landmarker, active, current_ms)
                for seg in active:
                    seg["sampled"] += 1
                    a = angles.get(seg["idx"])
                    if a is not None:
                        seg["series"].append((t_now, a))

            frame_to_write = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            av_frame = av.VideoFrame.from_ndarray(frame_to_write, format="bgr24")
            for packet in stream.encode(av_frame):
                container.mux(packet)

        frames_processed += 1

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    cap.release()
    try:
        landmarker.close()
    except Exception:
        pass

    preview_placeholder.empty()
    return output_video_path, {"fps_known": fps_known, "sample_period": sample_period}


def summarize_segments(segments, sample_period):
    """Convierte cada tramo procesado en un diccionario de resultados."""
    results = []
    for seg in segments:
        series = seg["series"]
        norm_min, norm_max = NORMATIVE_RANGES[seg["bp"]][seg["mov_id"]]
        vels = compute_angular_velocities(series, sample_period)
        results.append({
            "title": seg["label"],
            "side": seg["side_label"],
            "start": seg["start"],
            "end": seg["end"],
            "max": round(max(a for _, a in series), 1) if series else None,
            "norm_min": norm_min,
            "norm_max": norm_max,
            "vel": round(max(vels), 1) if vels else None,
            "vel_p95": round(float(np.percentile(vels, 95)), 1) if vels else None,
            "sampled": seg["sampled"],
            "valid": len(series),
            "series": series,
            "color_hex": seg["color_hex"],
        })
    return results

# ----------------------------------------------------------------------------
# 4. Protocolo (tabla de tramos)
# ----------------------------------------------------------------------------

def make_protocol_df(keys, side_label, duration):
    """Reparte los movimientos en tramos iguales a lo largo del video."""
    n = max(1, len(keys))
    seg_len = duration / n
    rows = []
    for i, (bp, mov_id) in enumerate(keys):
        rows.append({
            COL_MOV: movement_label(bp, mov_id),
            COL_SIDE: side_label,
            COL_START: round(i * seg_len, 1),
            COL_END: round(min((i + 1) * seg_len, duration), 1),
        })
    return pd.DataFrame(rows, columns=[COL_MOV, COL_SIDE, COL_START, COL_END])


def parse_protocol(df, duration):
    """Valida la tabla y devuelve (tramos, errores)."""
    segments, errors = [], []
    for i, row in df.reset_index(drop=True).iterrows():
        vals = [row.get(COL_MOV), row.get(COL_SIDE), row.get(COL_START), row.get(COL_END)]
        if all(pd.isna(v) for v in vals):
            continue  # fila vacía
        if any(pd.isna(v) for v in vals):
            errors.append(f"Fila {i + 1}: completa movimiento, lado, inicio y fin.")
            continue
        mov_label, side_label, start, end = vals[0], vals[1], float(vals[2]), float(vals[3])
        if mov_label not in MOVE_LABEL_TO_KEY:
            errors.append(f"Fila {i + 1}: movimiento no reconocido.")
            continue
        if start >= end:
            errors.append(f"Fila {i + 1} ({mov_label}): el inicio debe ser menor que el fin.")
            continue
        if end > duration + 0.05:
            errors.append(f"Fila {i + 1} ({mov_label}): el fin supera la duración del video ({duration:.1f} s).")
            continue

        bp, mov_id = MOVE_LABEL_TO_KEY[mov_label]
        idx = len(segments)
        color_hex = SEGMENT_PALETTE[idx % len(SEGMENT_PALETTE)]
        segments.append({
            "idx": idx,
            "bp": bp,
            "mov_id": mov_id,
            "mov": get_movement(bp, mov_id),
            "side": SIDE_LABELS[side_label],
            "side_label": side_label,
            "label": mov_label,
            "start": start,
            "end": end,
            "color_hex": color_hex,
            "color_bgr": hex_to_bgr(color_hex),
        })
    return segments, errors


# ----------------------------------------------------------------------------
# 4b. Modo automático: el usuario elige QUÉ evaluar, sin definir tiempos
# ----------------------------------------------------------------------------
# Movimientos que se pueden separar automáticamente usando el signo del ángulo.
# (articulación, movimiento) -> (vista requerida, dirección +1 / -1)
# Muñeca y rotaciones NO están: dependen de la orientación de la palma / posición
# 90-90 y se evalúan en el modo "por tramos".
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


def signed_angle(bp, view, pts, f, mid_sh_x, mid_hip_x):
    """
    Ángulo con signo. Positivo = flexión (o dorsiflexión) en vista lateral y
    abducción en vista frontal; negativo = extensión (plantiflexión) / aducción.
    """
    a, b, c = pts
    raw = angle_between(a, b, c)
    if raw is None:
        return None

    if view == "lateral":
        if bp == "tobillo":
            return 90.0 - raw
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


def build_auto_groups(selection, sides):
    """
    selection: {articulación: [ids de movimiento]}; sides: ["left", "right"].
    Agrupa por (articulación, lado, vista): flexión y extensión comparten un
    único ángulo con signo.
    """
    groups, mov_count = [], 0
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


def analyze_frame_auto(frame, landmarker, groups, timestamp_ms, view_mode, state):
    """Devuelve (frame dibujado, {gid: ángulo con signo suavizado o None})."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    record = {}
    if not result.pose_landmarks:
        return frame, record

    h, w = frame.shape[:2]
    sc = draw_scale(frame)
    lm = pick_main_person(result.pose_landmarks)

    dx, dy = (lm[12].x - lm[11].x) * w, (lm[12].y - lm[11].y) * h
    tilt_angle = abs(math.degrees(math.atan2(dy, dx)))
    if 8 < tilt_angle < 172:
        put_text(frame, "Camara inclinada", (int(20 * sc), h - int(20 * sc)), 0.8, (0, 0, 255), sc)

    for i1, i2 in POSE_CONNECTIONS:
        p1, p2 = lm[i1], lm[i2]
        cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (150, 150, 150), max(2, int(2 * sc)))

    view = view_mode if view_mode in ("lateral", "frontal") else detect_view(lm, w, h)
    put_text(frame, f"Vista: {view or 'oblicua (sin medir)'}", (int(20 * sc), int(34 * sc)), 0.7, (255, 255, 255), sc)

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
            continue
        pts_norm = [lm[i] for i in g["lm_idx"]]
        pts_px = [(p.x * w, p.y * h) for p in pts_norm]
        low_conf = any((getattr(p, "visibility", 1.0) or 1.0) < VISIBILITY_MIN for p in pts_norm)
        color = (60, 70, 226) if low_conf else g["color_bgr"]

        signed = signed_angle(g["bp"], view, pts_px, f, mid_sh_x, mid_hip_x)
        draw_angle_arc(frame, pts_px[0], pts_px[1], pts_px[2], color, scale=sc)
        for p in pts_px:
            cv2.circle(frame, (int(p[0]), int(p[1])), max(5, int(6 * sc)), color, -1)

        smoothed = None
        if signed is not None and not low_conf:
            g["buffer"].append(signed)
            if len(g["buffer"]) > SMOOTH_WINDOW:
                g["buffer"].pop(0)
            smoothed = sum(g["buffer"]) / len(g["buffer"])
            name = ascii_text(dir_name(g["bp"], view, 1 if smoothed >= 0 else -1))
            vx, vy = pts_px[1]
            put_text(frame, f"{abs(smoothed):.0f} {name}", (int(vx + 15 * sc), int(vy + (-15 + 32 * k) * sc)), 0.9, color, sc)
        record[g["gid"]] = smoothed
        k += 1

    return frame, record


def process_video_auto(video_path, groups, target_fps, preview_placeholder, start_sec, end_sec, view_mode):
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

    orig_w, orig_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_w, new_h = output_size(orig_w, orig_h)

    out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_video_path = out_tfile.name
    out_tfile.close()

    container = av.open(output_video_path, mode="w")
    output_fps = max(1, int(round(video_fps / frame_step)))
    stream = container.add_stream("h264", rate=output_fps)
    stream.width, stream.height, stream.pix_fmt = new_w, new_h, "yuv420p"
    stream.options = {"preset": OUTPUT_PRESET, "crf": OUTPUT_CRF}

    state = {"f_votes": []}
    frames_processed = 0
    preview_placeholder.info("⏳ Procesando evaluación... ")

    while frames_processed <= total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t_now = (start_frame + frames_processed) / video_fps

        # MediaPipe solo se ejecuta en los frames correspondientes al FPS de análisis.
        if frames_processed % frame_step == 0:
            frame, record = analyze_frame_auto(
                frame, landmarker, groups, int(t_now * 1000), view_mode, state
            )

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
                        mv["series"].append((t_now, max(value, 0.0)))

            frame_to_write = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            av_frame = av.VideoFrame.from_ndarray(frame_to_write, format="bgr24")
            for packet in stream.encode(av_frame):
                container.mux(packet)

        frames_processed += 1

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    cap.release()
    try:
        landmarker.close()
    except Exception:
        pass

    # El video procesado se conserva como archivo temporal y no se carga completo en RAM.
    except Exception:
        pass

    preview_placeholder.empty()
    return output_video_path, {"fps_known": fps_known, "sample_period": sample_period}


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
# 5. Gráficos y PDF
# ----------------------------------------------------------------------------

def make_protocol_chart(results):
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        if not r["series"]:
            continue
        ts = [p[0] for p in r["series"]]
        angs = [p[1] for p in r["series"]]
        side_short = "Izq" if r["side"] == "Izquierdo" else "Der"
        ax.plot(ts, angs, color=r["color_hex"], linewidth=2, label=f"{r['title']} ({side_short})")
        if r.get("shade", True):
            ax.axvspan(r["start"], r["end"], color=r["color_hex"], alpha=0.05)

    ax.set_ylim(0, 180)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Ángulo (°)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=7, frameon=False)
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


def build_pdf_report(sessions):
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
    pdf.multi_cell(0, 6, f"Paciente: {last['patient']}   |   RUN: {last['run']}   |   Generado: {date.today().isoformat()}")
    pdf.ln(4)

    widths = [70, 25, 30, 35, 30]
    headers = ["Movimiento", "Lado", "Máximo", "Rango AAOS", "Velocidad"]

    for s in sessions:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(28, 43, 48)
        pdf.cell(0, 9, f"Evaluación - {s['date']}  ({len(s['results'])} movimientos)", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(90, 100, 105)
        pdf.cell(0, 6, f"Evaluado por: {s.get('evaluador', '-')}", ln=True)
        pdf.set_text_color(28, 43, 48)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(s["chart_png"])
            tmp_img_path = tmp_img.name
        pdf.image(tmp_img_path, w=160)
        pdf.ln(2)
        try:
            os.remove(tmp_img_path)
        except Exception:
            pass

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(225, 245, 238)
        pdf.set_text_color(28, 43, 48)
        for wdt, htxt in zip(widths, headers):
            pdf.cell(wdt, 7, htxt, border=1, fill=True)
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
                pdf.cell(wdt, 7, txt, border=1)
            pdf.ln()
        pdf.ln(4)

        if s.get("diagnostico"):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(28, 43, 48)
            pdf.cell(0, 6, f"Impresión Diagnóstica: {s['diagnostico']}", ln=True)

        if s.get("observaciones"):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(28, 43, 48)
            pdf.cell(0, 6, "Observaciones Clínicas:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, s["observaciones"])

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
st.caption("Análisis cinemático, goniometría 2D y velocidad angular. Varios movimientos en un solo video.")

for key, default in [("sessions", []), ("results", []), ("editor_version", 0), ("protocol_df", None), ("file_id", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

MODE_AUTO = "Automático: elijo qué evaluar"
MODE_SEGMENTS = "Por tramos: defino los tiempos"
VIEW_MODES = {"Detectar automáticamente": "auto", "Siempre lateral (de perfil)": "lateral", "Siempre frontal (de frente)": "frontal"}


def store_results(results, video_path):
    if all(r["valid"] == 0 for r in results):
        st.session_state.results = []
        try:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass
        st.error(
            "No hubo frames con confianza suficiente en ningún movimiento. "
            "Revisa la iluminación, el encuadre, la vista de cámara o que las articulaciones estén visibles."
        )
    else:
        old_processed = st.session_state.get("processed_video_path")
        if old_processed and old_processed != video_path:
            try:
                os.remove(old_processed)
            except Exception:
                pass
        st.success("✅ Procesamiento completado.")
        st.session_state.results = results
        st.session_state.processed_video_path = video_path


st.header("1. Video")
target_fps = st.select_slider(
    "Frecuencia de análisis (FPS)", options=[10, 15, 30], value=10
)
st.caption(
    f"Para mayor estabilidad en la nube: máximo {MAX_DURATION_SEC} s y {MAX_VIDEO_MB} MB. "
    "Se recomienda MP4/H.264, 720p y movimientos lentos."
)
uploaded_file = st.file_uploader("Sube el video del paciente", type=["mp4", "mov", "avi"])
preview_placeholder = st.empty()

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_VIDEO_MB:
        st.error(
            f"El video pesa {file_size_mb:.1f} MB. Para esta versión en la nube, "
            f"utiliza un archivo de máximo {MAX_VIDEO_MB} MB."
        )
        st.stop()

    file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    # Solo se escribe/lee el video cuando cambia el archivo (no en cada rerun)
    if st.session_state.file_id != file_id:
        old_path = st.session_state.get("tmp_path")
        if old_path:
            try:
                os.remove(old_path)
            except Exception:
                pass
        old_processed = st.session_state.get("processed_video_path")
        if old_processed:
            try:
                os.remove(old_processed)
            except Exception:
                pass
            st.session_state.pop("processed_video_path", None)

        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.tmp_path = tmp.name

        cap_temp = cv2.VideoCapture(st.session_state.tmp_path)
        total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_raw = cap_temp.get(cv2.CAP_PROP_FPS)
        cap_temp.release()

        st.session_state.fps_known = bool(fps_raw and fps_raw > 0)
        st.session_state.duration = total_frames / (fps_raw if st.session_state.fps_known else FALLBACK_FPS)
        st.session_state.file_id = file_id
        st.session_state.results = []
        st.session_state.protocol_df = make_protocol_df([("hombro", "flexion")], "Derecho", st.session_state.duration)
        st.session_state.editor_version += 1

    duration = st.session_state.duration
    tmp_path = st.session_state.tmp_path

    if not st.session_state.fps_known:
        st.warning(
            "No fue posible determinar los FPS del video. El ROM puede analizarse, "
            "pero la velocidad angular no será informada."
        )

    if duration > MAX_DURATION_SEC:
        st.error(
            f"El video dura {duration:.1f} segundos. Para esta versión en la nube, "
            f"utiliza videos de máximo {MAX_DURATION_SEC} segundos."
        )
        st.stop()

    with st.expander("Ver video original"):
        st.video(tmp_path)
    st.caption(f"Duración del video: {duration:.1f} s")

    st.header("2. Qué quieres evaluar")
    mode = st.radio("Modo de evaluación", [MODE_AUTO, MODE_SEGMENTS], horizontal=True)

    # ==================================================================
    # MODO AUTOMÁTICO
    # ==================================================================
    if mode == MODE_AUTO:
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
        st.caption("Muñeca y rotaciones de hombro/cadera no están en este modo: usa el modo «Por tramos».")

        if st.button("Procesar evaluación", disabled=not selection):
            sides = ["left", "right"] if side_choice == "Ambos" else [SIDE_LABELS[side_choice]]
            groups = build_auto_groups(selection, sides)
            with st.spinner("Procesando cinemática y velocidad angular..."):
                processed_video_path, meta = process_video_auto(
                    tmp_path, groups, target_fps, preview_placeholder, start_sec, end_sec, view_mode
                )
                results = summarize_auto(groups, meta["sample_period"], start_sec, end_sec)
            store_results(results, processed_video_path)

    # ==================================================================
    # MODO POR TRAMOS
    # ==================================================================
    else:
        st.write(
            "Define en qué segundo empieza y termina cada movimiento. "
            "Puedes agregar filas, borrarlas o cargar el protocolo completo de tren superior."
        )

        cpre1, cpre2, cpre3 = st.columns([2, 2, 1])
        preset_side = cpre1.selectbox("Lado del protocolo", list(SIDE_LABELS.keys()), index=1)
        if cpre2.button("Cargar protocolo de tren superior"):
            st.session_state.protocol_df = make_protocol_df(UPPER_LIMB_PROTOCOL, preset_side, duration)
            st.session_state.editor_version += 1
            st.rerun()
        if cpre3.button("Limpiar"):
            st.session_state.protocol_df = make_protocol_df([], preset_side, duration).iloc[0:0]
            st.session_state.editor_version += 1
            st.rerun()

        st.caption(
            "El protocolo de tren superior reparte los tiempos en partes iguales: "
            "ajusta Inicio y Fin de cada fila según tu video."
        )

        edited_df = st.data_editor(
            st.session_state.protocol_df,
            key=f"protocol_editor_{st.session_state.editor_version}",
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                COL_MOV: st.column_config.SelectboxColumn(COL_MOV, options=list(MOVE_LABEL_TO_KEY.keys()), required=True, width="medium"),
                COL_SIDE: st.column_config.SelectboxColumn(COL_SIDE, options=list(SIDE_LABELS.keys()), required=True),
                COL_START: st.column_config.NumberColumn(COL_START, min_value=0.0, step=0.1, format="%.1f"),
                COL_END: st.column_config.NumberColumn(COL_END, min_value=0.0, step=0.1, format="%.1f"),
            },
        )

        segments, errors = parse_protocol(edited_df, duration)
        for err in errors:
            st.error(err)

        views_needed = sorted({seg["mov"]["view"] for seg in segments})
        if len(views_needed) > 1:
            st.info(
                "Este protocolo mezcla movimientos de vista **lateral** y **frontal**. Con una sola cámara "
                "fija, el paciente debe girar (de frente a de perfil) entre los tramos correspondientes, "
                "y cada tramo debe comenzar cuando ya esté en la posición correcta."
            )

        if st.button("Procesar protocolo", disabled=(len(segments) == 0 or len(errors) > 0)):
            with st.spinner("Procesando cinemática y velocidad angular..."):
                processed_video_path, meta = process_video_protocol(tmp_path, segments, target_fps, preview_placeholder)
                results = summarize_segments(segments, meta["sample_period"])
            store_results(results, processed_video_path)

# ----------------------------------------------------------------------
if st.session_state.results:
    results = st.session_state.results

    if "processed_video_path" in st.session_state:
        st.video(st.session_state.processed_video_path)

    st.header("3. Evolución temporal por movimiento")
    fig = make_protocol_chart(results)
    st.pyplot(fig)
    chart_png = fig_to_png_bytes(fig)

    st.header("4. Resultados clínicos")
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
        elif pct < MIN_USABLE_PCT:
            st.warning(f"{r['title']} ({r['side']}): solo {pct:.0f}% de frames utilizables; interpreta con cautela.")
        elif r["max"] is None:
            st.warning(f"{r['title']} ({r['side']}): no se detectó movimiento en esa dirección.")

    st.header("5. Registro de sesión y Observaciones")
    p1, p2, p3 = st.columns(3)
    patient_name = p1.text_input("Nombre del paciente")
    patient_run = p2.text_input("RUN", placeholder="12.345.678-9")
    test_date = p3.date_input("Fecha", value=date.today())
    evaluador = st.text_input("Evaluado por", placeholder="Nombre del profesional que evaluó al paciente")

    diagnostico = st.text_input("Impresión Diagnóstica (Opcional)", placeholder="Ej: Limitación funcional leve por probable tendinopatía...")
    observaciones = st.text_area("Observaciones Clínicas", placeholder="Ej: Paciente refiere dolor en los últimos 15° de flexión. Se observan compensaciones musculares...")

    if st.button("Guardar temporalmente"):
        saved_results = [{k: v for k, v in r.items() if k not in ("series", "color_hex", "shade", "view")} for r in results]
        st.session_state.sessions.append({
            "run": patient_run or "-",
            "patient": patient_name or "Sin nombre",
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

    col_pdf, col_xlsx = st.columns(2)
    with col_pdf:
        if st.button("Generar informe (PDF)"):
            pdf_bytes = build_pdf_report(st.session_state.sessions)
            safe_filename = st.session_state.sessions[-1]["patient"].replace(" ", "_").replace("-", "_")
            st.download_button("Descargar informe (PDF)", data=pdf_bytes, file_name=f"informe_{safe_filename}.pdf", mime="application/pdf")
    with col_xlsx:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            sessions_df.to_excel(writer, index=False, sheet_name="Pruebas")
        st.download_button("Descargar historial (Excel)", data=excel_buffer.getvalue(), file_name="historial_pruebas.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")