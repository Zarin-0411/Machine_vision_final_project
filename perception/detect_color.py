import cv2
import numpy as np
import json
import os


# =========================
# Load calibration matrix safely
# =========================
def load_calibration():

    base_dir = os.path.dirname(os.path.dirname(__file__))
    calib_path = os.path.join(base_dir, "calibration", "calibration.json")

    if not os.path.exists(calib_path):
        print("❌ calibration.json not found")
        return None

    with open(calib_path, "r") as f:
        data = json.load(f)

    H = np.array(data["H"], dtype=np.float64)

    print("Calibration matrix loaded")
    return H


# =========================
# Convert pixel -> robot coordinates (HIGH PRECISION)
# =========================
def pixel_to_robot(u, v, H):

    p = np.array([u, v, 1.0], dtype=np.float64).reshape(3, 1)

    pr = H @ p
    pr = pr / pr[2]

    X = pr[0][0]
    Y = pr[1][0]

    return float(X), float(Y)


# =========================
# Detect colored tiles with precise center
# =========================
def detect_objects(show_windows=True):

    H = load_calibration()

    if H is None:
        return []

    results = []

    # =========================
    # Camera Initialization
    # =========================
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*'MJPG'))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera opened:", cap.isOpened())

    if not cap.isOpened():
        print("❌ Camera not detected")
        return []

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to capture image")
        return []

    original_frame = frame.copy()

    # =========================
    # Convert to HSV
    # =========================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect colored tiles (ignore white background)
    lower = np.array([0, 60, 40])
    upper = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # =========================
    # Clean mask (improves center accuracy)
    # =========================
    kernel = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # =========================
    # Find contours
    # =========================
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"Contours found: {len(contours)}")

    for cnt in contours:

        area = cv2.contourArea(cnt)

        # Filter noise
        if area < 1500:
            continue

        # =========================
        # Compute TRUE center using centroid
        # =========================
        M = cv2.moments(cnt)

        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Convert to robot coordinates
        robot_x, robot_y = pixel_to_robot(cx, cy, H)

        results.append({
            "pixel": (float(cx), float(cy)),
            "robot": (robot_x, robot_y)
        })

        # =========================
        # Draw center (RED DOT)
        # =========================
        cv2.circle(frame,
                   (int(cx), int(cy)),
                   7,
                   (0, 0, 255),
                   -1)

        # Draw contour (GREEN)
        cv2.drawContours(frame,
                         [cnt],
                         -1,
                         (0, 255, 0),
                         2)

        # Draw robot coordinate text
        text = f"({robot_x:.1f},{robot_y:.1f})"

        cv2.putText(frame,
                    text,
                    (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2)

    # =========================
    # Sort objects left-to-right (optional improvement)
    # =========================
    results = sorted(results, key=lambda obj: obj["robot"][0])

    # =========================
    # Save annotated image
    # =========================
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "output")

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir,
                               "annotated_result.jpg")

    cv2.imwrite(output_path, frame)

    print(f"\nAnnotated image saved to:")
    print(output_path)

    # =========================
    # Show debug windows
    # =========================
    if show_windows:

        cv2.imshow("Detected Objects", frame)
        cv2.imshow("Mask", mask)

        print("\nPress any key to close windows")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


# =========================
# Standalone test
# =========================
if __name__ == "__main__":

    objs = detect_objects(show_windows=True)

    print(f"\nObjects detected: {len(objs)}\n")

    for i, obj in enumerate(objs):

        print(f"Object {i+1}")
        print(f"Pixel: {obj['pixel']}")
        print(f"Robot: ({obj['robot'][0]:.2f}, "
              f"{obj['robot'][1]:.2f})\n")