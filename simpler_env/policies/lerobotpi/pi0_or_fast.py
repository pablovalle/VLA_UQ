from typing import Optional, Sequence, List
import os
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle
from collections import deque
from PIL import Image
import torch
import cv2 as cv
from simpler_env.utils.action.action_ensemble import ActionEnsembler
from .geometry import quat2mat, mat2euler
import numpy as np
import torch
from uncertainty.token_based_metrics_fast import TokenMetricsFast
import time


def auto_model_fn(path, **kwargs):
    import sys; sys.path.append(path) # noqa
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    map_location = kwargs.pop("map_location", "cpu")
    
    return PI0Policy.from_pretrained(path, **kwargs).to(map_location)

class LerobotPiFastInference:
    def __init__(
        self,
        saved_model_path: str = "pretrained/pi0",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        exec_horizon: int = 1,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        action_ensemble_temp: float = -0.8,
    ) -> None:
        gpu_idx = os.environ.get("GPU_IDX", 0)
        self.device = f"cuda:{gpu_idx}"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig/1.0.0" if unnorm_key is None else unnorm_key
            action_ensemble = True
            self.sticky_gripper_num_repeat = 1
            # EE pose in Bridge data was relative to a top-down pose, instead of robot base
            self.default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203
        elif policy_setup == "google_robot":
            unnorm_key = (
                "fractal20220817_data/0.1.0" if unnorm_key is None else unnorm_key
            )
            action_ensemble = True
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key


        self.token_metrics_fast = TokenMetricsFast()

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

        # TODO: add pi0 loading ...
        #PI0Policy = auto_model_fn(saved_model_path, self.device)
        self.vla = auto_model_fn(saved_model_path, map_location="cuda")
        #self.vla = PI0Policy.from_pretrained(saved_model_path)
        self.vla.reset()

        self.image_size = image_size
        self.action_scale = action_scale
        self.obs_horizon = 1
        self.obs_interval = 1
        self.pred_action_horizon = 4
        self.image_history = deque(maxlen=self.obs_horizon)
        self.exec_horizon = exec_horizon

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp

        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(
                self.pred_action_horizon, self.action_ensemble_temp
            )
        else:
            self.action_ensembler = None

        self.task = None
        self.task_description = None

    def reset(self, task_description: str) -> None:
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.task_description = task_description
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.vla.reset()

    def preprocess_widowx_proprio(self, eef_pos) -> np.array:
        """convert ee rotation to the frame of top-down
        https://github.com/allenzren/open-pi-zero/blob/c3df7fb062175c16f69d7ca4ce042958ea238fb7/src/agent/env_adapter/simpler.py#L167
        """
        # StateEncoding.POS_EULER: xyz + rpy + pad + gripper(openness)
        proprio = eef_pos
        rm_bridge = quat2mat(proprio[3:7])
        rpy_bridge_converted = mat2euler(rm_bridge @ self.default_rot.T)
        gripper_openness = proprio[7] # from simpler, 0 for close, 1 for open
        raw_proprio = np.concatenate(
            [
                proprio[:3],
                rpy_bridge_converted,
                np.zeros(1),
                [gripper_openness],
            ]
        )
        return raw_proprio

    def preprocess_googple_robot_proprio(self, eef_pos) -> np.array:
        """convert wxyz quat from simpler to xyzw used in fractal
        https://github.com/allenzren/open-pi-zero/blob/c3df7fb062175c16f69d7ca4ce042958ea238fb7/src/agent/env_adapter/simpler.py#L204
        """
        # StateEncoding.POS_QUAT: xyz + q_xyzw + gripper(closeness)
        quat_xyzw = np.roll(eef_pos[3:7], -1)
        gripper_width = eef_pos[
            7
        ]  # from simpler, 0 for close, 1 for open
        # need invert as the training data comes from closeness
        gripper_closedness = (
            1 - gripper_width
        )  # TODO(allenzren): change fractal data processing in training so also use gripper openness in proprio (as in bridge) instead of closedness
        raw_proprio = np.concatenate(
            (
                eef_pos[:3],
                quat_xyzw,
                [gripper_closedness],
            )
        )
        return raw_proprio

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images: List[Image.Image] = self._obtain_image_history()

        eef_pos = kwargs.get("eef_pos", None)
        if self.policy_setup == "widowx_bridge":
            state = self.preprocess_widowx_proprio(eef_pos)
            image_key = "observation.images.image_0"
        elif self.policy_setup == "google_robot":
            state = self.preprocess_googple_robot_proprio(eef_pos)
            image_key = "observation.images.image"
        
        observation = {
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(self.device).float(),
            image_key: torch.from_numpy(images[0] / 255).permute(2, 0, 1).unsqueeze(0).to(self.device).float(),
            "task": [task_description], 
        }

        # model output gripper action, +1 = open, 0 = close
        raw_actions, vlm_logits, expert_logits_tuples = self.vla.select_action(observation)
        raw_actions = raw_actions[0][:self.pred_action_horizon].cpu().numpy()

        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        """
        here the PaliGemmaWithExpertModel is inferred multiple time,
        so we obtain logits_tuples containing the logits of all the inferences.
        We can calculate token-based-metrics for each logits.
        """
        # for logits in logits_tuples:
        #     self.token_metrics.calculate_entropy_prob(logits)
        #     print(self.token_metrics.shannon_entropy_list)

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(
                raw_actions[0, 6:7]
            ), 
             #"scores": np.array([self.token_metrics_fast.shannon_entropy_list, self.token_metrics_fast.token_prob,self.token_metrics_fast.pcs, self.token_metrics_fast.deepgini ]) # range [0, 1]; 1 = open; 0 = close
            "vlm_scores": vlm_logits,
            "expert_scores": expert_logits_tuples[-1]
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float64
        )
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            
            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def _add_image_to_history(self, image: np.ndarray) -> None:
        if len(self.image_history) == 0:
            self.image_history.extend([image] * self.obs_horizon)
        else:
            self.image_history.append(image)

    def _obtain_image_history(self) -> List[Image.Image]:
        image_history = list(self.image_history)
        images = image_history[:: self.obs_interval]
        # images = [Image.fromarray(image).convert("RGB") for image in images]
        return images

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[np.ndarray],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate(
                    [a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1
                )
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(
                pred_actions[:, action_dim], label="predicted action"
            )
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)