import argparse
from perception.detect_color import detect_objects
from robot.robot_control import robot_connect, robot_disconnect, pick_one


def run_detect(mode):

    results = detect_objects(show_windows=True)

    if len(results) == 0:
        print("No targets found")
        return

    print(f"{len(results)} targets found\n")

    for i, obj in enumerate(results):
        px = obj["pixel"]
        rb = obj["robot"]

        print(f"Target {i+1}")
        print(f"Pixel (u,v): {px}")
        print(f"Robot (X,Y): ({rb[0]:.2f}, {rb[1]:.2f})\n")

    if mode == "plan":
        print("Plan mode selected. Robot will not move.")
        return

    if mode == "execute":

        confirm = input("Robot will move. Continue? (y/n): ")
        if confirm.lower() != "y":
            print("Execution cancelled")
            return

        robot_connect()

        for obj in results:
            target_xy = obj["robot"]
            print(f"Sending robot to: {target_xy}")
            pick_one(target_xy)

        robot_disconnect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["detect"]
    )

    parser.add_argument(
        "--mode",
        choices=["plan", "execute"],
        default="plan"
    )

    args = parser.parse_args()

    if args.command == "detect":
        run_detect(args.mode)