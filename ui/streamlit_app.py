import sys
import os
import streamlit as st


BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.join(BASE_DIR, "machine_vision_project")
sys.path.append(PROJECT_DIR)


from machine_vision_project.perception.detect_color import detect_objects
from machine_vision_project.robot.robot_control import (
    robot_connect,
    robot_disconnect,
    pick_one
)


st.title("Machine Vision Robot Control")

mode = st.radio("Mode", ["Plan", "Execute"])
color = st.selectbox(
    "Select Color",
    ["any", "red", "green", "blue", "yellow"]
)


if st.button("Detect Objects"):

    st.write("Starting detection...")
    results = detect_objects(show_windows=False)

    if not results:
        st.warning("No objects detected")
        st.stop()

    st.success(f"Found {len(results)} objects")

    # Filter by selected color
    filtered_results = []
    for obj in results:
        obj_color = obj.get("color", "unknown")
        if color == "any" or obj_color == color:
            filtered_results.append(obj)

    if not filtered_results:
        st.warning("No objects match selected color")
        st.stop()

    st.write("Filtered objects:")
    for obj in filtered_results:
        st.write(obj)

    
    if mode == "Execute":

        try:
            st.info("Connecting to robot...")
            robot_connect()
            st.success("Robot connected")

            for obj in filtered_results:
                st.info(f"Picking object at {obj['robot']}")
                pick_one(obj["robot"])

            st.success("Execution completed")

        except Exception as e:
            st.error(f"Robot error: {e}")

        finally:
            st.info("Disconnecting robot...")
            robot_disconnect()
            st.success("Robot disconnected")

    else:
        st.info("Plan mode selected. No robot movement executed.")
