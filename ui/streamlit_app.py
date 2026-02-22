# ui/streamlit_app.py
from pathlib import Path
import streamlit as st
import cv2
import numpy as np

from calibration.io import load_calibration
from perception.detect import detect_colored_objects, annotate
from perception.geometry import uv_to_XY
from robot.mock import MockRobot
# From lab PC, switch to the real robot:
# from robot.mg400 import MG400

st.set_page_config(page_title="Vision-Guided Pick & Place", layout="wide")

# -----------------------------
# Sidebar controls (operator UI)
# -----------------------------
st.sidebar.header("Controls")

mode = st.sidebar.radio("Mode", ["Plan", "Execute"], index=0)
color = st.sidebar.selectbox("Color", ["any","red","yellow","orange","green","blue","purple"], index=0)
shape = st.sidebar.selectbox("Shape (optional)", ["any","circle","other"], index=0)

calib_path = st.sidebar.text_input("Calibration file", "calibration/calibration.json")

# Choose image OR camera
use_camera = st.sidebar.checkbox("Use camera", value=False)
camera_index = st.sidebar.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
image_path = st.sidebar.text_input("Image file (when not using camera)", "data/images/calib_image.jpeg")

btn_capture = st.sidebar.button("Capture / Refresh")
btn_detect  = st.sidebar.button("Detect target")
btn_pick    = st.sidebar.button("Run pick (Execute mode only)", disabled=(mode!="Execute"))

# -----------------------------
# Main area layout
# -----------------------------
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Camera / Scene")
with col_right:
    st.subheader("Overlay & Results")

# -----------------------------
# State helpers
# -----------------------------
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "last_detections" not in st.session_state:
    st.session_state.last_detections = []
if "overlay" not in st.session_state:
    st.session_state.overlay = None

# -----------------------------
# Load/capture image
# -----------------------------
def get_scene_frame():
    """Return BGR frame and a string label of source."""
    if use_camera:
        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            st.error("Camera capture failed.")
            return None, "[camera]"
        return frame, f"[camera {camera_index}]"
    else:
        if not Path(image_path).exists():
            st.error(f"Image not found: {image_path}")
            return None, image_path
        frame = cv2.imread(image_path)
        if frame is None:
            st.error(f"Failed to load image: {image_path}")
        return frame, image_path

# Show current scene
frame_src = "[none]"
if btn_capture or st.session_state.last_frame is None:
    st.session_state.last_frame, frame_src = get_scene_frame()
else:
    frame_src = "(cached)"

with col_left:
    if st.session_state.last_frame is not None:
        st.image(cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_BGR2RGB),
                 caption=f"Scene: {frame_src}")
    else:
        st.info("No scene available. Click 'Capture / Refresh'.")

# -----------------------------
# Detection pipeline
# -----------------------------
def run_detection():
    # Load calibration
    if not Path(calib_path).exists():
        st.error("Calibration file not found. Run CLI calibrate first.")
        return None, None, []

    try:
        H, meta = load_calibration(calib_path)
    except Exception as e:
        st.error(f"Failed to load calibration: {e}")
        return None, None, []

    img = st.session_state.last_frame
    if img is None:
        st.warning("No frame to analyze. Capture first.")
        return None, None, []

    c = None if color=="any" else color
    dets = detect_colored_objects(img, color=c, return_masks=False)

    # Optional shape filter
    if shape != "any":
        dets = [d for d in dets if d.get("shape")==shape]

    # Map UV->XY and prepare overlay
    for d in dets:
        X, Y = uv_to_XY(H, d['uv'])
        d['XY'] = (X, Y)

    vis = annotate(img, dets, H=H, uv_to_XY=uv_to_XY)
    return H, vis, dets

# Detect button
if btn_detect:
    H, vis, dets = run_detection()
    st.session_state.overlay = vis
    st.session_state.last_detections = dets

# Show overlay & results
with col_right:
    if st.session_state.overlay is not None:
        st.image(cv2.cvtColor(st.session_state.overlay, cv2.COLOR_BGR2RGB),
                 caption="Overlay")
    else:
        st.info("No overlay yet. Click 'Detect target'.")

    # Results list
    dets = st.session_state.last_detections or []
    if len(dets)==0:
        st.warning("No target found.")
    else:
        st.success(f"Found {len(dets)} target(s).")
        # Show compact table with (X,Y) and color/shape
        rows = []
        for d in dets:
            (u,v) = d['uv']
            (X,Y) = d['XY']
            rows.append({
                "u": u, "v": v,
                "X(mm)": round(X,1), "Y(mm)": round(Y,1),
                "color": d.get("color",""),
                "shape": d.get("shape","")
            })
        st.dataframe(rows, use_container_width=True)

# -----------------------------
# Execute (Run pick) - mock by default
# -----------------------------
if btn_pick:
    if mode != "Execute":
        st.error("Switch mode to Execute to run pick.")
    else:
        dets = st.session_state.last_detections or []
        if len(dets)==0:
            st.warning("No detections to pick. Click 'Detect target' first.")
        else:
            st.info("Running pick sequence (MOCK). On lab PC, switch to MG400 in code.")
            robot = MockRobot()
            # For real robot in lab:
            # robot = MG400(ip='192.168.1.6', port=29999, v=30, a=30, z_safe=120.0, z_pick=60.0)

            robot.connect()

            # Optional: per-color bins (edit)
            place_map = {
                'red':    (200, 100),
                'yellow': (220, 100),
                'orange': (240, 100),
                'green':  (260, 100),
                'blue':   (280, 100),
                'purple': (300, 100),
            }
            default_drop = (200, 150)

            for d in dets:
                X, Y = d['XY']
                robot.movej(X, Y, 120)   # approach
                robot.movej(X, Y, 60)    # pick height (tune)
                robot.suction_on()
                robot.movej(X, Y, 120)   # lift

                px, py = place_map.get(d.get('color'), default_drop)
                robot.movej(px, py, 120)
                robot.movej(px, py, 60)
                robot.suction_off()
                robot.movej(px, py, 120)

            robot.disconnect()
            st.success("Pick sequence completed (MOCK).")