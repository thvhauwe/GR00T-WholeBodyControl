import time
from copy import deepcopy
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R

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

class KinematicCarController:
    def __init__(self, max_linear_vel=0.4, max_angular_vel=0.4, kp_pos=1.0, kp_yaw=1.0):
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.kp_pos = kp_pos
        self.kp_yaw = kp_yaw

    def compute_command(self, current_pose, goal_pose):
        """
        current_pose: [x, y, yaw]
        goal_pose: [x, y, yaw]
        Returns: [vx, vy, omega]
        """
        dx = goal_pose[0] - current_pose[0]
        dy = goal_pose[1] - current_pose[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Heading error
        target_yaw = np.arctan2(dy, dx)
        yaw_error = target_yaw - current_pose[2]
        # Normalize yaw error
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        
        # Velocity in world frame
        v_linear = np.clip(self.kp_pos * dist, 0, self.max_linear_vel)
        
        # If we are close to the goal, reduce velocity
        if dist < 0.1:
            v_linear = 0.0
            # Finally align with goal yaw
            yaw_error = goal_pose[2] - current_pose[2]
            yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
            
        omega = np.clip(self.kp_yaw * yaw_error, -self.max_angular_vel, self.max_angular_vel)
        
        # Transform velocity to robot frame
        # Robot frame: x forward, y left
        cos_y = np.cos(current_pose[2])
        sin_y = np.sin(current_pose[2])
        
        # Let's just use the car model: v is forward in robot frame
        vx_robot = v_linear if dist > 0.1 else 0.0
        vy_robot = 0.0 # pure car model
        
        return [vx_robot, vy_robot, omega]

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
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 0.0, np.pi])
    ]
    current_goal_idx = 0
    controller = KinematicCarController()

    rate = node.create_rate(config.control_frequency)
    
    print(f"Robot starting. Initial goal: {goals[current_goal_idx]}")

    initial_upper_body_pose = None
    last_log_time = 0
    goal_reached_time = None
    dwell_time = 2.0

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

                # Navigation logic
                current_pose = get_current_pose(obs)
                goal_pose = goals[current_goal_idx]
                
                t_now = time.monotonic()
                
                # Check for goal reached (incl orientation)
                dx = goal_pose[0] - current_pose[0]
                dy = goal_pose[1] - current_pose[1]
                dist = np.sqrt(dx**2 + dy**2)
                
                # Yaw error to goal orientation
                yaw_error = goal_pose[2] - current_pose[2]
                yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
                
                is_at_goal = dist < 0.1 and abs(yaw_error) < 0.1
                
                if goal_reached_time is None:
                    if is_at_goal:
                        goal_reached_time = t_now
                        print(f"Goal {current_goal_idx} reached! Dwelling for {dwell_time}s...")
                        nav_cmd = [0.0, 0.0, 0.0]
                    else:
                        nav_cmd = controller.compute_command(current_pose, goal_pose)
                else:
                    # Currently dwelling
                    nav_cmd = [0.0, 0.0, 0.0]
                    if t_now - goal_reached_time > dwell_time:
                        current_goal_idx = (current_goal_idx + 1) % len(goals)
                        goal_reached_time = None
                        print(f"Switching to goal {current_goal_idx}: {goals[current_goal_idx]}")

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

                # Status publishing
                policy_use_action = False
                try:
                    if hasattr(wbc_policy, "lower_body_policy"):
                        policy_use_action = getattr(
                            wbc_policy.lower_body_policy, "use_policy_action", False
                        )
                except:
                    pass

                policy_status_msg = {"use_policy_action": policy_use_action, "timestamp": t_now}
                lower_body_policy_status_pub.publish(policy_status_msg)

                joint_safety_ok = env.get_joint_safety_status()
                joint_safety_status_pub.publish({
                    "joint_safety_ok": joint_safety_ok,
                    "timestamp": t_now,
                })

                # Data exporting
                msg = deepcopy(obs)
                for key in list(msg.keys()):
                    if key.endswith("_image"):
                        del msg[key]
                
                msg.update({
                    "action": wbc_action["q"],
                    "action.eef": DEFAULT_WRIST_POSE,
                    "base_height_command": DEFAULT_BASE_HEIGHT,
                    "navigate_command": nav_cmd,
                    "timestamps": {"main_loop": time.time(), "proprio": time.time()},
                })
                data_exp_pub.publish(msg)

                # Periodic logging (~1Hz)
                if t_now - last_log_time > 1.0:
                    print(f"--- Navigation Status ---")
                    print(f"Goal {current_goal_idx}: {goal_pose}")
                    print(f"Current pose: {current_pose}")
                    print(f"Distance to goal: {dist:.2f}")
                    print(f"Nav cmd: linear=({nav_cmd[0]:.2f}, {nav_cmd[1]:.2f}), angular={nav_cmd[2]:.2f}")
                    last_log_time = t_now

            if env.sim and (not env.sim.sim_thread or not env.sim.sim_thread.is_alive()):
                break

            rate.sleep()

    except Exception as e:
        print(f"Error in control loop: {e}")
    finally:
        print("Cleaning up...")
        dispatcher.stop()
        ros_manager.shutdown()
        env.close()

if __name__ == "__main__":
    config = tyro.cli(ControlLoopConfig)
    main(config)
