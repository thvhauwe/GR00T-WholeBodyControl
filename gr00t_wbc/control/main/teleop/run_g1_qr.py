import time
from copy import deepcopy
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R
import mujoco
import cv2

from gr00t_wbc.control.envs.g1.g1_env import G1Env
from gr00t_wbc.control.main.constants import (
    DEFAULT_BASE_HEIGHT,
    DEFAULT_NAV_CMD,
    DEFAULT_WRIST_POSE,
    JOINT_SAFETY_STATUS_TOPIC,
    LOWER_BODY_POLICY_STATUS_TOPIC,
    ROBOT_CONFIG_TOPIC,
    STATE_TOPIC_NAME,
)
from gr00t_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from gr00t_wbc.control.policy.wbc_policy_factory import get_wbc_policy
from gr00t_wbc.control.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from gr00t_wbc.control.utils.keyboard_dispatcher import (
    KeyboardDispatcher,
    KeyboardEStop,
    KeyboardListenerPublisher,
    ROSKeyboardDispatcher,
)
from gr00t_wbc.control.utils.ros_utils import (
    ROSManager,
    ROSMsgPublisher,
    ROSServiceServer,
)
from gr00t_wbc.control.utils.telemetry import Telemetry

CONTROL_NODE_NAME = "ControlPolicyQR"

class OmnidirectionalController:
    def __init__(self, max_linear_vel=0.2, max_angular_vel=1.0, kp_pos=1.0, kp_yaw=1.0):
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.kp_pos = kp_pos
        self.kp_yaw = kp_yaw

    def compute_command(self, current_pose, goal_pose):
        """
        current_pose: [x, y, yaw]
        goal_pose: [x, y, yaw]
        Returns: [vx_local, vy_local, omega]
        """
        # World frame translation error
        dx_world = goal_pose[0] - current_pose[0]
        dy_world = goal_pose[1] - current_pose[1]
        
        # Desired world velocity
        vx_world = self.kp_pos * dx_world
        vy_world = self.kp_pos * dy_world
        
        # Clip world velocity magnitude
        v_mag = np.sqrt(vx_world**2 + vy_world**2)
        if v_mag > self.max_linear_vel:
            vx_world = (vx_world / v_mag) * self.max_linear_vel
            vy_world = (vy_world / v_mag) * self.max_linear_vel
            
        # Rotate to local frame
        yaw = current_pose[2]
        vx_local = vx_world * np.cos(yaw) + vy_world * np.sin(yaw)
        vy_local = -vx_world * np.sin(yaw) + vy_world * np.cos(yaw)
        
        # Yaw error to goal orientation
        yaw_error = goal_pose[2] - current_pose[2]
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        
        omega = np.clip(self.kp_yaw * yaw_error, -self.max_angular_vel, self.max_angular_vel)
        
        return [vx_local, vy_local, omega]

def get_current_pose(obs):
    """Extract x, y, yaw from floating_base_pose [pos, quat]"""
    pose = obs["floating_base_pose"]
    x, y = pose[0], pose[1]
    # quat is [w, x, y, z]
    quat_wzyz = pose[3:7]
    r = R.from_quat([quat_wzyz[1], quat_wzyz[2], quat_wzyz[3], quat_wzyz[0]])
    yaw = r.as_euler('zyx')[0]
    return np.array([x, y, yaw])

def main(config: ControlLoopConfig):
    # Phase 2 overrides
    config.env_name = "qr_code"
    config.enable_offscreen = True
    config.enable_onscreen = True

    ros_manager = ROSManager(node_name=CONTROL_NODE_NAME)
    node = ros_manager.node

    ROSServiceServer(ROBOT_CONFIG_TOPIC, config.to_dict())
    wbc_config = config.load_wbc_yaml()

    data_exp_pub = ROSMsgPublisher(STATE_TOPIC_NAME)
    lower_body_policy_status_pub = ROSMsgPublisher(LOWER_BODY_POLICY_STATUS_TOPIC)
    joint_safety_status_pub = ROSMsgPublisher(JOINT_SAFETY_STATUS_TOPIC)

    telemetry = Telemetry(window_size=100)
    waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
    robot_model = instantiate_g1_robot_model(
        waist_location=waist_location, high_elbow_pose=config.high_elbow_pose
    )

    env = G1Env(
        env_name=config.env_name,
        robot_model=robot_model,
        config=wbc_config,
        wbc_version=config.wbc_version,
    )
    if env.sim and not config.sim_sync_mode:
        env.start_simulator()

    wbc_policy = get_wbc_policy("g1", robot_model, wbc_config, config.upper_body_joint_speed)

    keyboard_listener_pub = KeyboardListenerPublisher()
    keyboard_estop = KeyboardEStop()
    if config.keyboard_dispatcher_type == "raw":
        dispatcher = KeyboardDispatcher()
    elif config.keyboard_dispatcher_type == "ros":
        dispatcher = ROSKeyboardDispatcher()
    
    dispatcher.register(env)
    dispatcher.register(wbc_policy)
    dispatcher.register(keyboard_listener_pub)
    dispatcher.register(keyboard_estop)
    dispatcher.start()

    # Define Goals
    goals = [
        np.array([4.0, 0.0, 0.0]),
        np.array([0.0, 0.0, np.pi])
    ]
    current_goal_idx = 0
    controller = OmnidirectionalController()
    qr_detector = cv2.QRCodeDetector()

    rate = node.create_rate(config.control_frequency)
    
    print(f"Robot starting. Initial goal: {goals[current_goal_idx]}")

    initial_upper_body_pose = None
    last_log_time = 0
    state = "NAVIGATING_TO_WAYPOINT" # NAVIGATING_TO_WAYPOINT, SEARCHING_FOR_QR, MANEUVERING_TO_QR, DWELLING
    goal_reached_time = None
    dwell_time = 5.0
    qr_detected_time = None

    try:
        while ros_manager.ok():
            t_start = time.monotonic()
            with telemetry.timer("total_loop"):
                if env.sim and config.sim_sync_mode:
                    env.step_simulator()

                obs = env.observe()
                wbc_policy.set_observation(obs)
                
                # Capture initial pose once to keep arms steady
                if initial_upper_body_pose is None:
                    initial_upper_body_pose = obs["q"][
                        robot_model.get_joint_group_indices("upper_body")
                    ].copy()

                current_pose = get_current_pose(obs)
                goal_pose = goals[current_goal_idx]
                t_now = time.monotonic()
                nav_cmd = [0.0, 0.0, 0.0]

                if state == "NAVIGATING_TO_WAYPOINT":
                    dx = goal_pose[0] - current_pose[0]
                    dy = goal_pose[1] - current_pose[1]
                    dist = np.sqrt(dx**2 + dy**2)
                    yaw_error = goal_pose[2] - current_pose[2]
                    yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
                    
                    if dist < 0.2 and abs(yaw_error) < 0.15:
                        print(f"Waypoint {current_goal_idx} reached. Searching for static QR code...")
                        state = "SEARCHING_FOR_QR"
                    else:
                        nav_cmd = controller.compute_command(current_pose, goal_pose)

                elif state == "SEARCHING_FOR_QR":
                    # Look for QR in head_camera_image
                    if "head_camera_image" in obs:
                        img = obs["head_camera_image"]
                        # Convert RGB to BGR for OpenCV
                        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        data, bbox, rectified_qr = qr_detector.detectAndDecode(bgr_img)
                        if data:
                            print(f"QR detected: {data}. Switching to maneuvering.")
                            state = "MANEUVERING_TO_QR"
                            qr_detected_time = t_now
                        else:
                            # Rotate slowly to find it if not immediately visible
                            nav_cmd = [0.0, 0.0, 0.2]
                    else:
                        print("Waiting for camera image...")
                        nav_cmd = [0.0, 0.0, 0.0]

                elif state == "MANEUVERING_TO_QR":
                    if "head_camera_image" in obs:
                        img = obs["head_camera_image"]
                        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        data, bbox, _ = qr_detector.detectAndDecode(bgr_img)
                        if data and bbox is not None:
                            # Simple visual servoing: center the QR in the image
                            # Image center is (320, 240) for 640x480
                            qr_center = np.mean(bbox[0], axis=0)
                            err_x = (qr_center[0] - 320) / 320.0 # Normalized error -1 to 1
                            
                            # Forward velocity to get closer until it occupies large area
                            qr_area = cv2.contourArea(bbox[0])
                            target_area = 20000 # Tune this for "reaching" distance
                            err_area = (target_area - qr_area) / target_area
                            
                            vx = np.clip(0.1 * err_area, 0.0, 0.1)
                            omega = np.clip(-1.0 * err_x, -0.5, 0.5)
                            nav_cmd = [vx, 0.0, omega]
                            
                            if qr_area > target_area * 0.9 and abs(err_x) < 0.1:
                                print("QR maneuver complete. Dwelling...")
                                state = "DWELLING"
                                goal_reached_time = t_now
                        else:
                            # Lost QR, stay still or search again
                            nav_cmd = [0.0, 0.0, 0.0]
                            if t_now - qr_detected_time > 2.0:
                                state = "SEARCHING_FOR_QR"
                    else:
                        nav_cmd = [0.0, 0.0, 0.0]

                elif state == "DWELLING":
                    nav_cmd = [0.0, 0.0, 0.0]
                    if t_now - goal_reached_time > dwell_time:
                        current_goal_idx = (current_goal_idx + 1) % len(goals)
                        goal_reached_time = None
                        state = "NAVIGATING_TO_WAYPOINT"
                        print(f"Switching to waypoint {current_goal_idx}: {goals[current_goal_idx]}")

                # --- Visualization: Camera Stream ---
                if "head_camera_image" in obs:
                    img = obs["head_camera_image"]
                    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # Redetection for visualization if we haven't already in this frame
                    # (In a real scenario, we'd reuse the existing bbox/data)
                    data, bbox, _ = qr_detector.detectAndDecode(bgr_img)
                    if data and bbox is not None:
                        # Draw bounding box
                        bbox = bbox.astype(int)
                        for i in range(len(bbox[0])):
                            cv2.line(bgr_img, tuple(bbox[0][i]), tuple(bbox[0][(i+1) % 4]), (0, 255, 0), 3)
                        cv2.putText(bgr_img, data, (bbox[0][0][0], bbox[0][0][1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.imshow("Head Camera Stream (QR)", bgr_img)
                    cv2.waitKey(1)

                dt = 1 / config.control_frequency
                
                wbc_goal = {
                    "target_upper_body_pose": initial_upper_body_pose,
                    "wrist_pose": DEFAULT_WRIST_POSE,
                    "base_height_command": DEFAULT_BASE_HEIGHT,
                    "navigate_cmd": np.array(nav_cmd),
                    "target_time": t_now + dt,
                }
                
                wbc_goal["interpolation_garbage_collection_time"] = t_now - 2 * dt
                wbc_policy.set_goal(wbc_goal)

                with telemetry.timer("policy_action"):
                    wbc_action = wbc_policy.get_action(time=t_now)

                env.queue_action(wbc_action)

                # Periodic logging (~1Hz)
                if t_now - last_log_time > 1.0:
                    print(f"--- State: {state} ---")
                    if state == "NAVIGATING_TO_WAYPOINT":
                        print(f"Goal {current_goal_idx}: {goal_pose}")
                        print(f"Dist: {dist:.2f}, YawErr: {yaw_error:.2f}")
                    last_log_time = t_now

                # Visualization
                if env.sim and hasattr(env.sim.sim_env, "viewer") and env.sim.sim_env.viewer:
                    viewer = env.sim.sim_env.viewer
                    viewer.user_scn.ngeom = 0
                    for i, goal in enumerate(goals):
                        rgba = [0, 1, 0, 0.5] if i == current_goal_idx else [1, 1, 0, 0.3]
                        geom_id = viewer.user_scn.ngeom
                        viewer.user_scn.ngeom += 1
                        geom = viewer.user_scn.geoms[geom_id]
                        mujoco.mjv_initGeom(
                            geom,
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.2, 0.0, 0.0]),
                            np.array([goal[0], goal[1], 0.4]),
                            np.eye(3).flatten(),
                            np.array(rgba),
                        )

            if env.sim and (not env.sim.sim_thread or not env.sim.sim_thread.is_alive()):
                break

            rate.sleep()

    except Exception as e:
        print(f"Error in control loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        cv2.destroyAllWindows()
        dispatcher.stop()
        ros_manager.shutdown()
        env.close()

if __name__ == "__main__":
    config = tyro.cli(ControlLoopConfig)
    main(config)
