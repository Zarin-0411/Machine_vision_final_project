import os
import sys
import io
import platform
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from PIL import Image

import streamlit as st

# -------------------------------
# Import your project modules
# -------------------------------
# Expecting this file at repo root. Ensure repo root in sys.path.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Your packages
from robot.robot_control import robot_connect, robot_disconnect, pick_one
# We won't call detect_objects() directly because it captures internally.
# Instead, we replicate its logic on the frame we capture here.
# from perception.detect_color import detect_objects

# -------------------------------
# Constants / Paths
# -------------------------------
CALIB_PATH = REPO_ROOT / "calibration" / "calibration.json"
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_PATH = OUTPUT_DIR / "annotated_result.jpg"

APP_TITLE = "🤖 Tile Pick & Place — Operator UI"

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")

# -------------------------------
# Session state initialization
# -------------------------------
ss = st.session_state
ss.setdefault("mode", "Plan")             # "Plan" | "Execute"
ss.setdefault("cam_source", "USB index")  # "USB index" | "RTSP URL"
ss.setdefault("usb_index", 1)             # default camera index like your code
ss.setdefault("rtsp_url", "")
ss.setdefault("last_frame_bgr", None)     # numpy array BGR
ss.setdefault("detections", [])           # list of dicts: {pixel:(u,v), robot:(X,Y)}
ss.setdefault("selected_idx", 0)
ss.setdefault("robot_connected", False)
ss.setdefault("calib_ok", False)
ss.setdefault("H", None)                  # 3x3 homography (float64)

# -------------------------------
# Utilities
# -------------------------------
def load_homography(calib_path: Path) -> Optional[np.ndarray]:
    if not calib_path.exists():
        return None
    try:
        import json
        with open(calib_path, "r") as f:
            data = json.load(f)
        H = np.array(data["H"], dtype=np.float64)
        if H.shape != (3, 3):
            return None
        return H
    except Exception:
        return None


def uv_to_xy(uv: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """Apply planar homography (pixels -> workspace mm)."""
    u, v = uv
    vec = np.array([u, v, 1.0], dtype=np.float64)
    XYW = H @ vec
    W = XYW[2]
    if abs(W) < 1e-12:
        raise ValueError("Degenerate homography (W≈0).")
    return float(XYW[0] / W), float(XYW[1] / W)


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def capture_single_frame(source_kind: str, usb_index: int = 1, rtsp_url: str = "") -> Optional[np.ndarray]:
    """
    Capture a single BGR frame.
    - For USB, try CAP_DSHOW + MJPG 640x480 (Windows), fallback otherwise.
    - For RTSP, open given URL and grab one frame.
    Returns None on failure.
    """
    cap = None
    try:
        if source_kind == "USB index":
            if platform.system().lower().startswith("win"):
                cap = cv2.VideoCapture(int(usb_index), cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(int(usb_index))

            # Try MJPG and 640x480 like your detector
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
                pass
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            # RTSP/HTTP URL
            cap = cv2.VideoCapture(rtsp_url)

        # Warmup a few frames for stability
        ok, frame = False, None
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            cv2.waitKey(30)

        if not ok or frame is None:
            return None
        return frame

    finally:
        if cap is not None:
            cap.release()


def detect_on_frame(frame_bgr: np.ndarray, H: Optional[np.ndarray]) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """
    Replicates perception/detect_color.py logic on the provided frame:
      - HSV -> mask (non-white colors)
      - Morph closing/opening + blur
      - External contours, area >= 1500
      - Centroid -> (u,v) -> (X,Y) via H (if available)
      - Draw centers/contours and label robot coords
      - Sort results by robot X
    Returns (results, mask, annotated_bgr)
    """
    results = []

    # HSV conversion
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Broad non-white mask (same as your code)
    lower = np.array([0, 60, 40])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup (same as your code)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Contours (external)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = frame_bgr.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:  # noise filter identical to your code
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        # Pixel center
        u, v = cx, cy

        # Robot coords (if H available)
        if H is not None:
            try:
                X, Y = uv_to_xy((u, v), H)
                robot_xy = (float(X), float(Y))
            except Exception:
                robot_xy = None
        else:
            robot_xy = None

        # Draw visuals (same look & feel as yours)
        cv2.circle(annotated, (int(cx), int(cy)), 7, (0, 0, 255), -1)      # red center
        cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)             # green contour

        label = f"({robot_xy[0]:.1f},{robot_xy[1]:.1f})" if robot_xy else "(no H)"
        cv2.putText(annotated, label, (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        results.append({"pixel": (u, v), "robot": robot_xy})

    # Sort left-to-right by robot X (as in your code); if robot is None, push to end
    results.sort(key=lambda r: (1e12 if r["robot"] is None else r["robot"][0]))

    return results, mask, annotated


def save_annotated(annotated_bgr: np.ndarray, out_path: Path) -> None:
    try:
        cv2.imwrite(str(out_path), annotated_bgr)
    except Exception:
        pass


# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙️ Controls")

ss["mode"] = st.sidebar.radio("Mode", ["Plan", "Execute"], horizontal=True, index=(0 if ss["mode"] == "Plan" else 1))

st.sidebar.markdown("### Camera")
ss["cam_source"] = st.sidebar.radio("Source", ["USB index", "RTSP URL"], index=(0 if ss["cam_source"] == "USB index" else 1))
if ss["cam_source"] == "USB index":
    ss["usb_index"] = st.sidebar.number_input("USB camera index", min_value=0, step=1, value=int(ss["usb_index"]))
else:
    ss["rtsp_url"] = st.sidebar.text_input("RTSP/HTTP URL", value=ss["rtsp_url"], placeholder="rtsp://user:pass@host/stream")

st.sidebar.markdown("---")
st.sidebar.markdown("### Calibration")
H_loaded = load_homography(CALIB_PATH)
ss["H"] = H_loaded
ss["calib_ok"] = H_loaded is not None
if ss["calib_ok"]:
    st.sidebar.success("Calibration: **Loaded**")
else:
    st.sidebar.warning("Calibration: **Missing** (`calibration/calibration.json`)")

st.sidebar.markdown("---")
st.sidebar.caption("Tips")
st.sidebar.info(
    "• Use **Plan** to detect and verify.\n"
    "• Switch to **Execute** to **Connect robot** and **Run pick**.\n"
    "• Annotated image is also saved to `output/annotated_result.jpg`."
)

# -------------------------------
# Main layout
# -------------------------------
left, right = st.columns([7, 5], gap="large")

with left:
    st.title(APP_TITLE)

    # Buttons row
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    # Capture
    if c1.button("📷 Capture / Refresh", use_container_width=True):
        frame = capture_single_frame(ss["cam_source"], ss["usb_index"], ss["rtsp_url"])
        if frame is None:
            st.error("Failed to capture. Check camera source.")
        else:
            ss["last_frame_bgr"] = frame
            ss["detections"] = []
            st.success("Frame captured.")

    # Detect
    if c2.button("🔍 Detect target", use_container_width=True):
        if ss["last_frame_bgr"] is None:
            st.warning("No frame yet. Click **Capture / Refresh** first.")
        else:
            results, mask, annotated = detect_on_frame(ss["last_frame_bgr"], ss["H"])
            ss["detections"] = results
            if annotated is not None:
                save_annotated(annotated, ANNOTATED_PATH)
            if len(results) == 0:
                st.warning("No targets found.")
            else:
                st.success(f"Detected {len(results)} target(s).")

    # Connect / Disconnect (Execute mode)
    connect_disabled = (ss["mode"] != "Execute")
    if c3.button(("🔌 Connect robot" if not ss["robot_connected"] else "🔌 Reconnect"), use_container_width=True, disabled=connect_disabled):
        try:
            robot_connect()
            ss["robot_connected"] = True
            st.success("Robot connected and ready.")
        except Exception as e:
            ss["robot_connected"] = False
            st.error(f"Robot connection failed: {e}")

    if c4.button("⏏ Disconnect robot", use_container_width=True, disabled=(ss["mode"] != "Execute" or not ss["robot_connected"])):
        try:
            robot_disconnect()
            ss["robot_connected"] = False
            st.info("Robot disconnected.")
        except Exception as e:
            st.error(f"Disconnect failed: {e}")

    # Run pick row
    run_pick_disabled = (ss["mode"] != "Execute" or not ss["robot_connected"] or len(ss["detections"]) == 0 or not ss["calib_ok"])
    if st.button("🤝 Run pick", use_container_width=True, disabled=run_pick_disabled):
        if not ss["calib_ok"]:
            st.warning("Calibration not loaded. Cannot execute.")
        elif not ss["robot_connected"]:
            st.warning("Robot not connected. Connect first.")
        elif len(ss["detections"]) == 0:
            st.warning("No detection to pick. Detect target first.")
        else:
            idx = min(ss["selected_idx"], len(ss["detections"]) - 1)
            target = ss["detections"][idx]["robot"]
            if target is None:
                st.error("Selected target has no calibrated coordinates (H missing/invalid).")
            else:
                try:
                    st.info(f"Executing pick at X={target[0]:.2f}, Y={target[1]:.2f} (mm)...")
                    pick_one(target)
                    st.success("Pick & place completed successfully.")
                except Exception as e:
                    st.error(f"Pick failed: {e}")

    # Image display
    st.markdown("### View")
    image_to_show = None
    caption = None

    if ANNOTATED_PATH.exists() and len(ss["detections"]) > 0:
        # Prefer annotated if we just detected
        annotated_bgr = cv2.imread(str(ANNOTATED_PATH))
        if annotated_bgr is not None:
            image_to_show = bgr_to_pil(annotated_bgr)
            caption = "Annotated result"
    elif ss["last_frame_bgr"] is not None:
        image_to_show = bgr_to_pil(ss["last_frame_bgr"])
        caption = "Last captured frame"

    if image_to_show is not None:
        st.image(image_to_show, caption=caption, use_column_width=True)
    else:
        st.info("Capture a frame to begin.")

with right:
    st.markdown("### Targets")
    dets = ss["detections"]

    if len(dets) == 0:
        st.caption("No detections yet.")
    else:
        # Build a compact table of detections
        rows = []
        for i, d in enumerate(dets):
            u, v = d["pixel"]
            xy = d["robot"]
            if xy is None:
                rows.append({
                    "Index": i,
                    "Pixel (u,v)": f"({u:.1f}, {v:.1f})",
                    "Robot (X,Y) [mm]": "N/A (no H)"
                })
            else:
                rows.append({
                    "Index": i,
                    "Pixel (u,v)": f"({u:.1f}, {v:.1f})",
                    "Robot (X,Y) [mm]": f"({xy[0]:.1f}, {xy[1]:.1f})"
                })

        # Selection + table
        ss["selected_idx"] = st.number_input("Select target index", min_value=0, max_value=max(0, len(dets)-1), step=1, value=min(ss["selected_idx"], len(dets)-1))
        st.dataframe(rows, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Status")
    st.write("**Mode:**", ss["mode"])
    st.write("**Calibration:**", "Loaded ✅" if ss["calib_ok"] else "Missing ⚠️")
    st.write("**Robot:**", "Connected ✅" if ss["robot_connected"] else "Disconnected ⛔")
    if ss["cam_source"] == "USB index":
        st.write("**Camera:**", f"USB index {ss['usb_index']}")
    else:
        st.write("**Camera:**", f"RTSP URL ({'set' if ss['rtsp_url'] else 'not set'})")

    st.markdown("---")
    st.markdown("### Notes")
    st.caption(
        "- Detection pipeline mirrors `perception/detect_color.py` (HSV mask, morphology, contours, centroid, H).\n"
        "- Coordinates (X, Y) are in **mm** as defined by your homography.\n"
        "- Annotated result saved to `output/annotated_result.jpg` like your original script.\n"
        "- Robot cycle is executed via your `robot/robot_control.py::pick_one()`.\n"
        "- For safety, **Run pick** is disabled unless in **Execute**, robot connected, calibration loaded, and a target is detected."
    )
