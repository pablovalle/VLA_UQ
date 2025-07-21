import time
import traceback

import numpy as np
import trossen_arm as trossen

from lerobot.common.robot_devices.motors.configs import TrossenArmDriverConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)

TROSSEN_ARM_MODELS = {
    "V0_LEADER": [trossen.Model.wxai_v0, trossen.StandardEndEffector.wxai_v0_leader],
    "V0_FOLLOWER": [trossen.Model.wxai_v0, trossen.StandardEndEffector.wxai_v0_follower],
}


class TrossenArmDriver:
    """
    The `TrossenArmDriver` class provides an interface for controlling
    Trossen Robotics' robotic arms. It leverages the trossen_arm for communication with arms.

    This class allows for configuration, torque management, and motion control of robotic arms. It includes features for handling connection states, moving the
    arm to specified poses, and logging timestamps for debugging and performance analysis.

    ### Key Features:
    - **Multi-motor Control:** Supports multiple motors connected to a bus.
    - **Mode Switching:** Enables switching between position and gravity compensation modes.
    - **Home and Sleep Pose Management:** Automatically transitions the arm to home and sleep poses for safe operation.
    - **Error Handling:** Raises specific exceptions for connection and operational errors.
    - **Logging:** Captures timestamps for operations to aid in debugging.

    ### Example Usage:
    ```python
    motors = {
        "joint_0": (1, "4340"),
        "joint_1": (2, "4340"),
        "joint_2": (4, "4340"),
        "joint_3": (6, "4310"),
        "joint_4": (7, "4310"),
        "joint_5": (8, "4310"),
        "joint_6": (9, "4310"),
    }
    arm_driver = TrossenArmDriver(
        motors=motors,
        ip="192.168.1.2",
        model="V0_LEADER",
    )
    arm_driver.connect()

    # Read motor positions
    positions = arm_driver.read("Present_Position")

    # Move to a new position (Home Pose)
    # Last joint is the gripper, which is in range [0, 450]
    arm_driver.write("Goal_Position", [0, 15, 15, 0, 0, 0, 200])

    # Disconnect when done
    arm_driver.disconnect()
    ```
    """

    def __init__(
        self,
        config: TrossenArmDriverConfig,
    ):
        self.ip = config.ip
        self.model = config.model
        self.mock = config.mock
        self.driver = None
        self.calibration = None
        self.is_connected = False
        self.logs = {}
        self.fps = 30
        self.home_pose = [0, np.pi / 3, np.pi / 6, np.pi / 5, 0, 0, 0]
        self.sleep_pose = [0, 0, 0, 0, 0, 0, 0]

        self.motors = {
            # name: (index, model)
            "joint_0": [1, "4340"],
            "joint_1": [2, "4340"],
            "joint_2": [3, "4340"],
            "joint_3": [4, "4310"],
            "joint_4": [5, "4310"],
            "joint_5": [6, "4310"],
            "joint_6": [7, "4310"],
        }

        # Minimum time to move for the arm
        self.MIN_TIME_TO_MOVE = 3.0 / self.fps

    def connect(self):
        print(f"Connecting to {self.model} arm at {self.ip}...")
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"TrossenArmDriver({self.ip}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        print("Initializing the drivers...")

        # Initialize the driver
        self.driver = trossen.TrossenArmDriver()

        # Get the model configuration
        try:
            model_name, model_end_effector = TROSSEN_ARM_MODELS[self.model]
        except KeyError as e:
            raise ValueError(f"Unsupported model: {self.model}") from e

        print("Configuring the drivers...")

        # Configure the driver
        try:
            self.driver.configure(model_name, model_end_effector, self.ip, True)
        except Exception:
            traceback.print_exc()
            print(f"Failed to configure the driver for the {self.model} arm at {self.ip}.")
            raise

        # Move the arms to the home pose
        self.driver.set_all_modes(trossen.Mode.position)
        self.driver.set_all_positions(self.home_pose, 2.0, False)

        # Allow to read and write
        self.is_connected = True

    def reconnect(self):
        try:
            model_name, model_end_effector = TROSSEN_ARM_MODELS[self.model]
        except KeyError as e:
            raise ValueError(f"Unsupported model: {self.model}") from e
        try:
            self.driver.configure(model_name, model_end_effector, self.ip, True)
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to configure the driver for the {self.model} arm at {self.ip}.")
            raise e

        self.is_connected = True

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def set_calibration(self, calibration: dict[str, list]):
        self.calibration = calibration

    def apply_calibration_autocorrect(self, values: np.ndarray | list, motor_names: list[str] | None):
        pass

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        pass

    def autocorrect_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        pass

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        pass

    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"TrossenArmDriver({self.ip}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        # Read the present position of the motors
        if data_name == "Present_Position":
            # Get the positions of the motors
            values = self.driver.get_all_positions()
        elif data_name == "External_Efforts":
            values = self.driver.get_all_external_efforts()
        else:
            values = None
            print(f"Data name: {data_name} is not supported for reading.")

        # TODO: Add support for reading other data names as required

        self.logs["delta_timestamp_s_read"] = time.perf_counter() - start_time

        values = np.array(values, dtype=np.float32)
        return values

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"TrossenArmDriver({self.ip}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        # Write the goal position of the motors
        if data_name == "Goal_Position":
            values = np.array(values, dtype=np.float32)
            self.driver.set_all_positions(values.tolist(), self.MIN_TIME_TO_MOVE, False)

        # Enable or disable the torque of the motors
        elif data_name == "Torque_Enable":
            # Set the arms to POSITION mode
            if values == 1:
                self.driver.set_all_modes(trossen.Mode.position)
            else:
                self.driver.set_all_modes(trossen.Mode.external_effort)
                self.driver.set_all_external_efforts([0.0] * self.driver.get_num_joints(), 0.0, True)
        elif data_name == "Reset":
            self.driver.set_all_modes(trossen.Mode.velocity)
            self.driver.set_all_velocities([0.0] * self.driver.get_num_joints(), 0.0, False)
            self.driver.set_all_modes(trossen.Mode.position)
            self.driver.set_all_positions(self.home_pose, 2.0, False)
        elif data_name == "External_Efforts":
            self.driver.set_all_external_efforts(values.tolist(), 0.0, False)
        else:
            print(f"Data name: {data_name} value: {values} is not supported for writing.")

        self.logs["delta_timestamp_s_write"] = time.perf_counter() - start_time

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"TrossenArmDriver ({self.ip}) is not connected. Try running `motors_bus.connect()` first."
            )
        self.driver.set_all_modes(trossen.Mode.velocity)
        self.driver.set_all_velocities([0.0] * self.driver.get_num_joints(), 0.0, False)
        self.driver.set_all_modes(trossen.Mode.position)
        self.driver.set_all_positions(self.home_pose, 2.0, True)
        self.driver.set_all_positions(self.sleep_pose, 2.0, False)

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
