from pathlib import Path
import json
import cv2
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None  # pandas is optional; we'll fall back to built-in streamlit tables
try:
    import streamlit as st
except ImportError:
    st = None
from robot import robot_control
def _pixel_to_robot(u, v, H):
    """Apply a 3x3 homography *H* to pixel coordinates (u,v) and return
    robot coordinates (x,y)."""
    p = np.array([u, v, 1.0], dtype=np.float64).reshape(3, 1)
    pr = H @ p
    pr = pr / pr[2]
    return float(pr[0][0]), float(pr[1][0])
class Detector:
    # colour thresholds in HSV (lower/upper pairs). red uses two ranges.
    _COLOR_RANGES = {
        "red": [([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])],
        "green": [([36, 100, 100], [86, 255, 255])],
        "blue": [([94, 80, 2], [126, 255, 255])],
    }
    def find_objects(self, image, color_name="any", shape_type="any"):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # build mask according to requested colour
        mask = None
        if color_name == "any":
            # combine all colours
            m = None
            for ranges in self._COLOR_RANGES.values():
                for (low, high) in ranges:
                    curr = cv2.inRange(hsv, np.array(low), np.array(high))
                    m = curr if m is None else cv2.bitwise_or(m, curr)
            mask = m
        else:
            ranges = self._COLOR_RANGES.get(color_name, [])
            for (low, high) in ranges:
                curr = cv2.inRange(hsv, np.array(low), np.array(high))
                mask = curr if mask is None else cv2.bitwise_or(mask, curr)

        if mask is None:
            return []

        # clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # ignore small noise
                continue

            # centroid
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

            # shape classification
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                detected_shape = "square"
            elif len(approx) > 4:
                detected_shape = "circle"
            else:
                detected_shape = "unknown"

            if shape_type != "any" and detected_shape != shape_type:
                continue

            results.append({
                "pixel_center": (cx, cy),
                "color": color_name if color_name != "any" else "unknown",
                "Shape": detected_shape,
            })

        return results


ROOT = Path(__file__).resolve().parent
# primary default image (older code used 'outputs', some modules write to 'output')
DEFAULT_IMAGE = ROOT / "outputs" / "camera_detection.png"
ALT_DEFAULT_IMAGE = ROOT / "output" / "annotated_result.jpg"
CALIBRATION_CANDIDATES = [
    ROOT / "calibration.json",
    ROOT / "callibration.json",
    ROOT / "calibration" / "calibration.json",
    ROOT / "calibration" / "callibration.json",
]


def _load_image(uploaded_file, captured_image):
    if uploaded_file is not None:
        # UploadedFile.read() consumes the stream; getvalue() is stable across reruns.
        data = np.frombuffer(uploaded_file.getvalue(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    if captured_image is not None:
        return captured_image

    if DEFAULT_IMAGE.exists():
        return cv2.imread(str(DEFAULT_IMAGE))

    # fallback to alternate output path used by some tools
    if ALT_DEFAULT_IMAGE.exists():
        return cv2.imread(str(ALT_DEFAULT_IMAGE))

    return None


def _load_homography(calibration_upload):
    if calibration_upload is not None:
        try:
            # UploadedFile.read() consumes the stream; getvalue() is stable across reruns.
            data = json.loads(calibration_upload.getvalue().decode("utf-8"))
            H = data.get("homography") or data.get("homography_matrix") or data.get("H")
            if H is None:
                return None, "Uploaded calibration JSON missing 'homography', 'homography_matrix' or 'H'"
            return np.array(H, dtype=np.float64), "Loaded calibration from uploaded file"
        except Exception as e:
            return None, f"Failed to read uploaded calibration: {e}"

    for path in CALIBRATION_CANDIDATES:
        if not path.exists():
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            H = data.get("homography") or data.get("homography_matrix") or data.get("H")
            if H is None:
                return None, f"Calibration file found at {path.name}, but no homography/H key"
            return np.array(H, dtype=np.float64), f"Loaded calibration from {path}"
        except Exception as e:
            return None, f"Failed to load calibration from {path}: {e}"

    return None, "Calibration file not found"


def _to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _build_rows(detected_objects, H):
    rows = []
    for idx, obj in enumerate(detected_objects, start=1):
        # support the legacy format returned by ``perception.detect_color`` as
        # well as the original UI design
        if "pixel_center" in obj:
            u, v = obj["pixel_center"]
        elif "pixel" in obj:
            u, v = obj["pixel"]
        else:
            continue

        rx, ry = None, None
        if H is not None:
            try:
                rx, ry = _pixel_to_robot(u, v, H)
            except Exception:
                rx, ry = None, None

        rows.append(
            {
                "id": idx,
                "shape": obj.get("Shape") or obj.get("shape") or "unknown",
                "color": obj.get("color", "unknown"),
                "pixel_u": u,
                "pixel_v": v,
                "robot_x": rx,
                "robot_y": ry,
            }
        )

    return rows


def _annotate_image(image_bgr, rows):
    annotated = image_bgr.copy()

    for row in rows:
        u = int(row["pixel_u"])
        v = int(row["pixel_v"])
        cv2.circle(annotated, (u, v), 7, (0, 0, 255), 2)

        label = f"#{row['id']} {row['shape']}"
        if row["robot_x"] is not None and row["robot_y"] is not None:
            label += f" ({row['robot_x']:.1f}, {row['robot_y']:.1f})"

        cv2.putText(
            annotated,
            label,
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def _ensure_state():
    if "robot" not in st.session_state:
        st.session_state.robot = None
    if "detections" not in st.session_state:
        st.session_state.detections = []
    if "captured_image" not in st.session_state:
        st.session_state.captured_image = None


def _connect_robot(ip):
    if st.session_state.robot is not None:
        return
    # override the hardcoded IP in robot_control then call the helper
    robot_control.ROBOT_IP = ip
    robot_control.robot_connect()
    st.session_state.robot = True


def _disconnect_robot():
    if not st.session_state.robot:
        return
    try:
        robot_control.robot_disconnect()
    finally:
        st.session_state.robot = None


def main():
    st.set_page_config(page_title="Dobot MG400 Pick-and-Place", layout="wide")
    _ensure_state()

    st.title("Dobot MG400 Vision Pick-and-Place")

    with st.sidebar:
        st.subheader("Detection")
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        calibration_upload = st.file_uploader("Upload calibration JSON (optional)", type=["json"])
        camera_index = st.number_input("Camera Index", value=0, step=1, min_value=0)
        capture_clicked = st.button("Capture From Camera")

        if capture_clicked:
            with st.spinner("Capturing frame from camera..."):
                try:
                    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        st.error("Camera capture failed. No image was saved.")
                    else:
                        # save to the default path for consistency with earlier code
                        DEFAULT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(DEFAULT_IMAGE), frame)
                        st.session_state.captured_image = frame
                        st.session_state.detections = []
                        st.success("Captured image from camera")
                except Exception as e:
                    st.error(f"Camera capture failed: {e}")

        color_name = st.selectbox("Color", ["any", "red", "green", "blue"])
        shape_type = st.selectbox("Shape", ["any", "circle", "square"])

        st.subheader("Robot")
        robot_ip = st.text_input("Robot IP", value="192.168.1.6")
        st.caption("Drop position is configured in robot/robot_control.py (BOX_X/BOX_Y/BOX_DROP_Z).")

        col1, col2 = st.columns(2)
        with col1:
            connect_clicked = st.button("Connect")
        with col2:
            disconnect_clicked = st.button("Disconnect")

        if connect_clicked:
            with st.spinner("Connecting to robot..."):
                try:
                    _connect_robot(robot_ip)
                    st.success("Robot connected")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

        if disconnect_clicked:
            with st.spinner("Disconnecting robot..."):
                try:
                    _disconnect_robot()
                    st.success("Robot disconnected")
                except Exception as e:
                    st.error(f"Disconnect failed: {e}")

    image = _load_image(uploaded_file, st.session_state.captured_image)
    if image is None:
        st.warning(
            "Upload an image, click Capture From Camera, or place outputs/camera_detection.png or output/annotated_result.jpg in the project root output folder."
        )
        return

    H, calibration_message = _load_homography(calibration_upload)
    if H is None:
        st.warning(f"Calibration unavailable: {calibration_message}")
    else:
        st.success(calibration_message)

    detector = Detector()

# If the user uploaded a file, auto-run detection so the UI becomes interactive
# without requiring an extra button click.
    if uploaded_file is not None and not st.session_state.detections:
        with st.spinner("Detecting uploaded image..."):
            try:
                detections = detector.find_objects(image, color_name=color_name, shape_type=shape_type)
                st.session_state.detections = _build_rows(detections, H)
                st.success("Auto-detection completed")
            except Exception as e:
                st.error(f"Auto-detection failed: {e}")

    if st.button("Detect Objects", type="primary"):
        detections = detector.find_objects(image, color_name=color_name, shape_type=shape_type)
        st.session_state.detections = _build_rows(detections, H)

    detections = st.session_state.detections

    left, right = st.columns([3, 2])

    with left:
        if detections:
            annotated = _annotate_image(image, detections)
            st.image(_to_rgb(annotated), caption="Detections", use_container_width=True)
        else:
            st.image(_to_rgb(image), caption="Input Image", use_container_width=True)

    with right:
        st.subheader("Detected Objects")

        if not detections:
            st.write("Run detection to list objects.")
            return

        if pd is not None:
            df = pd.DataFrame(detections)
            st.dataframe(df, use_container_width=True)
        else:
            # lightweight fallback; ``st.table`` works with list-of-dicts
            st.table(detections)

        selectable_ids = [row["id"] for row in detections if row["robot_x"] is not None and row["robot_y"] is not None]
        if not selectable_ids:
            st.warning("No calibrated robot coordinates available for detected objects.")
            return

        selected_id = st.selectbox("Select object to pick", selectable_ids)

        connected = bool(st.session_state.robot)

        pick_col1, pick_col2 = st.columns(2)

        with pick_col1:
            if st.button("Pick Selected", use_container_width=True):
                if not connected:
                    st.error("Connect robot first")
                else:
                    row = next((r for r in detections if r["id"] == selected_id), None)
                    if row is None:
                        st.error("Selected object not found")
                    elif row["robot_x"] is None or row["robot_y"] is None:
                        st.error("Selected object has no robot coordinates")
                    else:
                        with st.spinner("Executing pick and place..."):
                            try:
                                robot_control.pick_one((row["robot_x"], row["robot_y"]))
                                st.success(f"Picked object #{selected_id}")
                            except Exception as e:
                                st.error(f"Pick failed: {e}")

        with pick_col2:
            if st.button("Pick All", use_container_width=True):
                if not connected:
                    st.error("Connect robot first")
                else:
                    with st.spinner("Executing pick and place for all objects..."):
                        try:
                            count = 0
                            for row in detections:
                                if row["robot_x"] is None or row["robot_y"] is None:
                                    continue
                                robot_control.pick_one((row["robot_x"], row["robot_y"]))
                                count += 1
                            st.success(f"Completed pick-and-place for {count} object(s)")
                        except Exception as e:
                            st.error(f"Pick-all failed: {e}")


def _cli_main():
    """Minimal CLI when Streamlit isn't installed."""
    import argparse

    parser = argparse.ArgumentParser(description="Fallback CLI for app.py")
    parser.add_argument("command", choices=["detect"], help="Only detection supported")
    parser.add_argument("--mode", choices=["plan", "execute"], default="plan")
    args = parser.parse_args()

    if args.command == "detect":
        image = None
        if DEFAULT_IMAGE.exists():
            image = cv2.imread(str(DEFAULT_IMAGE))
        if image is None:
            print("No input image found. See README or use the Streamlit UI.")
            return

        H, msg = _load_homography(None)
        print(msg)
        detector = Detector()
        detections = detector.find_objects(image)
        rows = _build_rows(detections, H)
        if not rows:
            print("No objects detected.")
            return

        for r in rows:
            print(r)

        if args.mode == "execute":
            confirm = input("Robot will move. Continue? (y/n): ")
            if confirm.lower() != "y":
                print("Execution cancelled")
                return
            robot_control.robot_connect()
            for r in rows:
                if r["robot_x"] is None or r["robot_y"] is None:
                    continue
                robot_control.pick_one((r["robot_x"], r["robot_y"]))
            robot_control.robot_disconnect()


if __name__ == "__main__":
    if st is None:
        print("Streamlit not found; switching to CLI mode")
        _cli_main()
    else:

        main()
