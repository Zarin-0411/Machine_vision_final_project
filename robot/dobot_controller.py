import threading
import time
from .dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType
from time import sleep
import numpy as np

current_actual = None
algorithm_queue = None
enableStatus_robot = None
robotErrorState = False
globalLockValue = threading.Lock()
stop_threads = False


def ConnectRobot(ip="192.168.1.6", timeout_s=5.0):
    dashboardPort = 29999
    movePort = 30003
    feedPort = 30004

    dashboard = DobotApiDashboard(ip, dashboardPort, timeout_s=timeout_s)
    move = DobotApiMove(ip, movePort, timeout_s=timeout_s)
    feed = DobotApi(ip, feedPort, timeout_s=timeout_s)

    return dashboard, move, feed


def GetFeed(feed: DobotApi):
    global current_actual, algorithm_queue, enableStatus_robot
    global robotErrorState, stop_threads

    feed.socket_dobot.settimeout(1.0)

    while not stop_threads:
        try:
            data = feed.socket_dobot.recv(1440)
            if len(data) == 0:
                continue

            feedInfo = np.frombuffer(data, dtype=MyType)

            if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
                with globalLockValue:
                    current_actual = feedInfo["tool_vector_actual"][0]
                    algorithm_queue = feedInfo['isRunQueuedCmd'][0]
                    enableStatus_robot = feedInfo['EnableStatus'][0]
                    robotErrorState = feedInfo['ErrorStatus'][0]

        except:
            sleep(0.01)


def StartFeedbackThread(feed: DobotApi):
    global stop_threads
    stop_threads = False

    feed_thread = threading.Thread(target=GetFeed, args=(feed,))
    feed_thread.daemon = True
    feed_thread.start()
    sleep(1)

    return feed_thread


def WaitArrive(target_point, tolerance=3.0, timeout=20.0):
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            return False

        with globalLockValue:
            if current_actual is not None:
                arrived = True
                for i in range(4):
                    if abs(current_actual[i] - target_point[i]) > tolerance:
                        arrived = False
                        break

                if arrived:
                    return True

        sleep(0.01)


def MoveJ(move: DobotApiMove, point):
    move.MovJ(point[0], point[1], point[2], point[3])


def MoveL(move: DobotApiMove, point):
    move.MovL(point[0], point[1], point[2], point[3])


def SetupRobot(dashboard: DobotApiDashboard, speed_ratio=50, acc_ratio=50, payload_weight=50):
    dashboard.ClearError()
    sleep(0.5)

    dashboard.EnableRobot()
    sleep(2)

    dashboard.SpeedJ(speed_ratio)
    dashboard.SpeedL(speed_ratio)
    dashboard.AccJ(acc_ratio)
    dashboard.AccL(acc_ratio)

    dashboard.PayLoad(payload_weight, 0)


def ControlDigitalOutput(dashboard: DobotApiDashboard, output_index, status):
    return dashboard.DO(output_index, status)


def GetCurrentPosition():
    with globalLockValue:
        return current_actual


def DisconnectRobot(dashboard, move, feed, feed_thread=None):
    global stop_threads
    stop_threads = True

    if feed_thread:
        feed_thread.join(timeout=2.0)

    try:
        dashboard.DisableRobot()
        sleep(0.5)
    except:
        pass

    dashboard.close()
    move.close()
    feed.close()