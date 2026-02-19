import numpy as np
import json


def load_calibration():
    with open("calibration/calibration.json11") as f:

        data = json.load(f)

    H = np.array(data["H"])
    return H


def pixel_to_robot(u, v, H):

    p = np.array([u, v, 1.0])
    pr = H @ p
    pr = pr / pr[2]

    # FIX FOR DOBOT MG400 AXIS
    X = pr[0]
    Y = pr[1]

    # Try swapping if incorrect:
    # X = pr[1]
    # Y = pr[0]

    return float(X), float(Y)



if __name__ == "__main__":
    H = load_calibration()

    print("Enter pixel coordinate to test:")
    u = float(input("u: "))
    v = float(input("v: "))

    X, Y = pixel_to_robot(u, v, H)

    print(f"\nRobot coordinates:")
    print(f"X = {X:.2f}")
    print(f"Y = {Y:.2f}")