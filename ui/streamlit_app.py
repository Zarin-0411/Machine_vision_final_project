import io
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import cv2

# =========================
# CONFIG & SESSION SETUP
# =========================
st.set_page_config(page_title="Factory Pick UI", page_icon="🤖", layout="wide")

if "last_image" not in st.session_state:
    st.session_state.last_image = None  # PIL Image
if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None  # PIL Image
if "detection" not in st.session_state:
    st.session_state.detection = None  # dict with bbox, center_px, center_xy
if "message" not in st.session_state:
    st.session_state.message = ("info", "Welcome! Capture an image to begin.")
if "mode" not in st.session_state:
    st.session_state.mode = "Plan"  # or "Execute"

# =========================
# UTILS
# =========================
def set_message(level: str, text: str):
    """level in {'success','warning','error','info'}"""
    st.session_state.message = (level, text)

def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def annotate(image_pil: Image.Image, bbox=None, center_px=None, label=None):
    """Draw bbox and center dot on image."""
    draw = ImageDraw.Draw(image_pil)
    # Choose a simple default font; Streamlit hosting often lacks system fonts
    # For production, embed a specific TTF for consistent rendering.
    font = None
    if bbox is not None:
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)
    if center_px is not None:
        cx, cy = center_px
        r = 6
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0), width=3)
    if label:
        # background box for readability
        draw.rectangle([10, 10, 10 + 8 * len(label), 40], fill=(0, 0, 0, 127))
        draw.text((14, 14), label, fill=(255, 255, 255), font=font)
    return image_pil

# =========================
# CAMERA CAPTURE PLACEHOLDER(S)
# =========================
def capture_from_browser(camera_input_bytes) -> Image.Image:
    """Convert st.camera_input bytes to PIL Image."""
    if camera_input_bytes is None:
        return None
    return Image.open(io.BytesIO(camera_input_bytes.getvalue())).convert("RGB")

def capture_from_ip_camera(url: str) -> Image.Image:
    """
    Example for future integration with an IP camera.
    Replace this with your actual stream/capture pipeline.
    """
    # Placeholder: not implemented for offline demo
    # Typical pattern:
    # cap = cv2.VideoCapture(url)
    # ret, frame = cap.read()
    # cap.release()
    # if ret:
    #     return cv2_to_pil(frame)
    return None

def generate_placeholder_image(w=960, h=540) -> Image.Image:
    """Generate a synthetic scene for demo: colored shapes on a background."""
    img = Image.new("RGB", (w, h), color=(30, 30, 35))
    draw = ImageDraw.Draw(img)
    # Red rectangle
    draw.rectangle([120, 150, 280, 300], fill=(220, 30, 30))
    # Green circle
    cx, cy, r = 500, 260, 70
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(30, 220, 60))
    # Blue square
    draw.rectangle([720, 180, 820, 280], fill=(30, 80, 220))
    return img

# =========================
# SIMPLE DETECTOR (replace with your model)
# =========================
def simple_color_shape_detector(img_pil: Image.Image, color: str, shape: str):
    """
    Returns a dict with:
      - bbox: (x, y, w, h) in pixels
      - center_px: (cx, cy)
      - center_xy: (X, Y) in same-length units after calibration (mocked here)
      - confidence: [0..1]
      - label: readable label
    This demo does naive color thresholding (HSV) + largest contour.
    """
    img_bgr = pil_to_cv2(img_pil)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Color ranges (HSV) — tune for your lighting
    ranges = {
        "red":   [((0, 140, 70), (10, 255, 255)), ((170, 140, 70), (180, 255, 255))],
        "green": [((35, 60, 60), (85, 255, 255))],
        "blue":  [((95, 80, 60), (130, 255, 255))],
        "any":   None
    }

    if color.lower() not in ranges:
        color = "any"

    if color == "any":
        # union of broad ranges
        masks = []
        for k in ("red", "green", "blue"):
            for (lo, hi) in ranges[k]:
                masks.append(cv2.inRange(hsv, np.array(lo), np.array(hi)))
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)
    else:
        masks = []
        for (lo, hi) in ranges[color.lower()]:
            masks.append(cv2.inRange(hsv, np.array(lo), np.array(hi)))
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)

    # Morph cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Optional shape filter (very naive demo):
    # - circle: contour circularity near 1
    # - square/rectangle: approx polygon with ~4 vertices
    # - any: skip shape filtering
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:  # ignore small artifacts
            continue
        x, y, w, h = cv2.boundingRect(cnt)

        score = float(area)
        if shape != "any":
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if shape == "circle":
                # prefer circular contours
                score *= (1.0 - abs(circularity - 1.0))  # closer to 1 is better
            elif shape == "square":
                # prefer ~4 vertices and near-square ratio
                if len(approx) == 4:
                    ratio = w / float(h)
                    score *= (1.0 - abs(1.0 - ratio))  # closer to 1 is better
                else:
                    score *= 0.1
            elif shape == "rectangle":
                # prefer ~4 vertices, any ratio OK
                if len(approx) != 4:
                    score *= 0.2
            else:
                pass

        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        return None

    x, y, w, h = best
    cx, cy = int(x + w / 2), int(y + h / 2)

    # Mock pixel->world conversion (replace with your calibration)
    # Assume 1 pixel = 0.5 mm and image origin at top-left
    px_to_mm = 0.5
    X = cx * px_to_mm  # mm
    Y = cy * px_to_mm  # mm

    result = {
        "bbox": (int(x), int(y), int(w), int(h)),
        "center_px": (int(cx), int(cy)),
        "center_xy": (float(X), float(Y)),
        "confidence": min(1.0, best_score / 100000.0),
        "label": f"{color.capitalize()} {shape.capitalize() if shape!='any' else ''}".strip()
    }
    return result

# =========================
# ROBOT PICK PLACEHOLDER
# =========================
def run_robot_pick(target_xy, detection_payload):
    """
    Replace with your robot/control API call.
    This function should:
      - move robot to (X,Y[,Z])
      - actuate gripper
      - return success/failure and diagnostic info
    """
    time.sleep(0.8)  # simulate execution time
    # Demo: succeed if detection confidence is decent
    ok = detection_payload and detection_payload.get("confidence", 0) > 0.1
    return ok, {"cycle_time_s": 0.8, "note": "demo path"}

# =========================
# SIDEBAR: CONTROLS
# =========================
st.sidebar.title("⚙️ Controls")

st.session_state.mode = st.sidebar.radio("Mode", options=["Plan", "Execute"], index=0, horizontal=True)

color = st.sidebar.selectbox("Color (optional)", options=["any", "red", "green", "blue"], index=0)
shape = st.sidebar.selectbox("Shape (optional)", options=["any", "circle", "square", "rectangle"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Camera source")

cam_option = st.sidebar.radio(
    "Select capture source",
    options=["Browser camera", "Factory camera (IP)", "Placeholder scene"],
    index=2
)

ip_url = None
if cam_option == "Factory camera (IP)":
    ip_url = st.sidebar.text_input("RTSP/HTTP URL", value="", placeholder="rtsp://user:pass@x.x.x.x/stream")

st.sidebar.markdown("---")
st.sidebar.caption("Tips")
st.sidebar.info("• Use **Plan** to detect/validate.\n• Switch to **Execute** to enable **Run pick**.", icon="ℹ️")

# =========================
# MAIN LAYOUT
# =========================
col_left, col_right = st.columns([7, 5], gap="large")

with col_left:
    st.title("🤖 Factory Operator UI")

    # Status message
    level, msg = st.session_state.message
    getattr(st, level)(msg)

    # --- Capture / Refresh ---
    st.subheader("Camera")
    cam_input_widget = None
    if cam_option == "Browser camera":
        cam_input_widget = st.camera_input("Capture", label_visibility="collapsed")

    cols_btn = st.columns([1, 1, 2, 2, 2])
    capture_clicked = cols_btn[0].button("📷 Capture / Refresh", use_container_width=True)
    detect_clicked = cols_btn[1].button("🔍 Detect target", use_container_width=True)
    run_pick_clicked = cols_btn[2].button("🤝 Run pick", use_container_width=True, disabled=(st.session_state.mode != "Execute"))

    # Handle capture
    if capture_clicked:
        img = None
        if cam_option == "Browser camera":
            if cam_input_widget is not None:
                img = capture_from_browser(cam_input_widget)
        elif cam_option == "Factory camera (IP)":
            if ip_url:
                img = capture_from_ip_camera(ip_url)
            else:
                set_message("warning", "Provide a valid IP camera URL.")
        else:
            img = generate_placeholder_image()

        if img is not None:
            st.session_state.last_image = img
            st.session_state.annotated_image = None
            st.session_state.detection = None
            set_message("success", "Image captured.")
        else:
            set_message("error", "Failed to capture image. Check source and try again.")

    # Handle detection
    if detect_clicked:
        if st.session_state.last_image is None:
            set_message("warning", "No image available. Click **Capture / Refresh** first.")
        else:
            det = simple_color_shape_detector(st.session_state.last_image, color=color, shape=shape)
            if det is None:
                st.session_state.annotated_image = st.session_state.last_image.copy()
                set_message("warning", "No target found with current filters.")
            else:
                annotated = annotate(
                    st.session_state.last_image.copy(),
                    bbox=det["bbox"],
                    center_px=det["center_px"],
                    label=f'{det["label"]} | conf={det["confidence"]:.2f}'
                )
                st.session_state.detection = det
                st.session_state.annotated_image = annotated
                cx, cy = det["center_px"]
                X, Y = det["center_xy"]
                set_message("success", f"Target detected at pixels ({cx}, {cy}) → coords ({X:.1f}, {Y:.1f}) mm.")

    # Handle run pick
    if run_pick_clicked:
        if st.session_state.mode != "Execute":
            set_message("warning", "Switch to **Execute** mode to run the pick.")
        elif st.session_state.detection is None:
            set_message("warning", "No valid detection. Click **Detect target** first.")
        else:
            ok, info = run_robot_pick(st.session_state.detection["center_xy"], st.session_state.detection)
            if ok:
                set_message("success", f"Pick SUCCESS. Cycle ~{info.get('cycle_time_s', 0):.2f}s.")
            else:
                set_message("error", "Pick FAILED. Verify detection, robot reach, and gripper state.")

    # Display image (last or annotated)
    st.subheader("View")
    img_to_show = st.session_state.annotated_image or st.session_state.last_image
    if img_to_show is not None:
        st.image(img_to_show, caption="Annotated image" if st.session_state.annotated_image else "Last captured image", use_column_width=True)
    else:
        st.info("No image yet. Capture to begin.")

with col_right:
    st.subheader("Target Data")
    if st.session_state.detection:
        det = st.session_state.detection
        x, y, w, h = det["bbox"]
        cx, cy = det["center_px"]
        X, Y = det["center_xy"]
        st.metric("Mode", st.session_state.mode)
        st.write("**Label**:", det["label"])
        st.write("**Bounding box (px)**:", f"(x={x}, y={y}, w={w}, h={h})")
        st.write("**Center (px)**:", f"({cx}, {cy})")
        st.write("**Coordinates (mm)**:", f"X={X:.1f}, Y={Y:.1f}")
        st.write("**Confidence**:", f"{det['confidence']:.2f}")
    else:
        st.caption("No detection yet.")

    st.markdown("---")
    st.subheader("Operator Notes")
    st.text_area("Add notes (optional):", key="op_notes", height=120)
    st.caption("These notes are local to this session. Persist to DB if needed.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Demo detector • Replace with your camera + robot APIs.")
