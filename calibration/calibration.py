import cv2
import numpy as np
import json
import os

# Global lists to store points
image_points = []
robot_points = []

# ---------------------------------------------------
# Mouse callback to collect pixel coordinates
# ---------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global image_points

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked pixel: ({x}, {y})")
        image_points.append([x, y])


# ---------------------------------------------------
# Main calibration function
# ---------------------------------------------------
def run_calibration():

    global image_points, robot_points

    # Reset lists (important if rerun)
    image_points = []
    robot_points = []

    # Open camera
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    print("Camera opened:", cap.isOpened())

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    # Create window and attach mouse callback
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)

    print("\nINSTRUCTIONS:")
    print("Click 4 points in this EXACT order:")
    print("1. Top-left corner")
    print("2. Top-right corner")
    print("3. Bottom-right corner")
    print("4. Bottom-left corner")
    print("Then press 'q'\n")

    # Show live camera feed
    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Cannot read camera frame")
            break

        # Draw clicked points
        for i, p in enumerate(image_points):

            cv2.circle(frame, tuple(p), 6, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"P{i+1}",
                (p[0] + 10, p[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Calibration", frame)

        # Press q to finish
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Check if enough points
    if len(image_points) != 4:
        print("ERROR: You must click EXACTLY 4 points")
        print(f"You clicked {len(image_points)} points")
        return

    # ---------------------------------------------------
    # Enter robot coordinates
    # ---------------------------------------------------

    print("\nNow enter robot coordinates in SAME ORDER")

    for i, p in enumerate(image_points):

        print(f"\nPoint {i+1} pixel: {p}")

        rx = float(input("Enter Robot X: "))
        ry = float(input("Enter Robot Y: "))

        robot_points.append([rx, ry])

    # Convert to numpy arrays
    img_pts = np.array(image_points, dtype=np.float32)
    rob_pts = np.array(robot_points, dtype=np.float32)

    # ---------------------------------------------------
    # Compute homography
    # ---------------------------------------------------

    H, status = cv2.findHomography(img_pts, rob_pts)

    if H is None:
        print("ERROR: Homography computation failed")
        return

    print("\nHomography matrix computed successfully")
    print(H)

    # ---------------------------------------------------
    # Save calibration safely using absolute path
    # ---------------------------------------------------

    # Get directory of THIS script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    save_path = os.path.join(script_dir, "calibration.json")

    calibration_data = {
        "H": H.tolist()
    }

    with open(save_path, "w") as f:
        json.dump(calibration_data, f, indent=4)

    print("\nCalibration saved successfully")
    print("File location:", save_path)


# ---------------------------------------------------
# Run calibration
# ---------------------------------------------------
if __name__ == "__main__":

    run_calibration()