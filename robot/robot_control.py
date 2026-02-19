from .dobot_controller import (
    ConnectRobot,
    StartFeedbackThread,
    SetupRobot,
    MoveJ,
    MoveL,
    WaitArrive,
    ControlDigitalOutput,
    DisconnectRobot
)

from time import sleep

ROBOT_IP = "192.168.1.6"

dashboard = None
move = None
feed = None
feed_thread = None


def robot_connect():
    global dashboard, move, feed, feed_thread

    print("Connecting to robot...")
    dashboard, move, feed = ConnectRobot(ip=ROBOT_IP, timeout_s=5.0)
    feed_thread = StartFeedbackThread(feed)
    SetupRobot(dashboard, speed_ratio=40, acc_ratio=40)
    print("Robot ready")


def robot_disconnect():
    global dashboard, move, feed, feed_thread

    print("Disconnecting robot...")
    DisconnectRobot(dashboard, move, feed, feed_thread)
    print("Robot disconnected")


def pick_one(target_xy):

    global dashboard, move

    # =========================
    # OBJECT PICK COORDINATES
    # =========================
    X, Y = target_xy

    SAFE_Z = -120
    PICK_Z = -158
    R_PICK = 0

    # =========================
    # BOX COORDINATES
    # =========================
    BOX_X = 248.44
    BOX_Y = -246.10
    BOX_SAFE_Z = -40
    BOX_DROP_Z = -70.52
    BOX_R = 98.48

    # =========================
    # VACUUM DO CONFIGURATION
    # =========================
    VACUUM_SUCTION_DO = 1   # suction
    VACUUM_BLOW_DO = 2      # blow

    print("\n==============================")
    print(f"Starting pick cycle at X={X:.2f}, Y={Y:.2f}")
    print("==============================")

    # =========================
    # STEP 1: Move above object
    # =========================
    print("Moving above object...")
    MoveL(move, [X, Y, SAFE_Z, R_PICK])
    WaitArrive([X, Y, SAFE_Z, R_PICK], tolerance=2.0)

    # =========================
    # STEP 2: Move down to object
    # =========================
    print("Moving down to object...")
    MoveL(move, [X, Y, PICK_Z, R_PICK])
    WaitArrive([X, Y, PICK_Z, R_PICK], tolerance=2.0)

    # =========================
    # STEP 3: TURN VACUUM SUCTION ON
    # =========================
    print("Activating vacuum suction")

    ControlDigitalOutput(dashboard, VACUUM_BLOW_DO, 0)   # stop blowing
    ControlDigitalOutput(dashboard, VACUUM_SUCTION_DO, 1) # suction ON

    sleep(0.7)

    # =========================
    # STEP 4: Lift object
    # =========================
    print("Lifting object...")
    MoveL(move, [X, Y, SAFE_Z, R_PICK])
    WaitArrive([X, Y, SAFE_Z, R_PICK], tolerance=2.0)

    # =========================
    # STEP 5: Move above box
    # =========================
    print("Moving above box...")
    MoveJ(move, [BOX_X, BOX_Y, BOX_SAFE_Z, BOX_R])
    WaitArrive([BOX_X, BOX_Y, BOX_SAFE_Z, BOX_R], tolerance=2.0)

    # =========================
    # STEP 6: Move down into box
    # =========================
    print("Lowering into box...")
    MoveL(move, [BOX_X, BOX_Y, BOX_DROP_Z, BOX_R])
    WaitArrive([BOX_X, BOX_Y, BOX_DROP_Z, BOX_R], tolerance=2.0)

    # =========================
    # STEP 7: RELEASE OBJECT
    # =========================
    print("Releasing object")

    ControlDigitalOutput(dashboard, VACUUM_SUCTION_DO, 0) # suction OFF
    ControlDigitalOutput(dashboard, VACUUM_BLOW_DO, 1)    # blow ON

    sleep(0.5)

    # Stop blowing
    ControlDigitalOutput(dashboard, VACUUM_BLOW_DO, 0)

    # =========================
    # STEP 8: Move up safely
    # =========================
    print("Moving up from box...")
    MoveL(move, [BOX_X, BOX_Y, BOX_SAFE_Z, BOX_R])
    WaitArrive([BOX_X, BOX_Y, BOX_SAFE_Z, BOX_R], tolerance=2.0)

    sleep(0.5)

    print("Pick and place completed successfully")