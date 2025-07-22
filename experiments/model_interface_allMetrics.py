"""
Name   : model_interface_allMetrics.py
Author : PABLO VALLE
Time   : 7/19/24
"""

import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien
import time
from pathlib import Path
import uncertainty.utils as uncerUtils
import torch

from uncertainty.token_based_metrics_fast import TokenMetricsFast
import uncertainty.uncertainty_metrics as uncerMetrics


SEED = 2024
TIME_RANGE = 8

PACKAGE_DIR = Path(__file__).parent.resolve()

RT_1_CHECKPOINTS = {
    "rt_1_x": "rt_1_x_tf_trained_for_002272480_step",
    "rt_1_400k": "rt_1_tf_trained_for_000400120",
    "rt_1_58k": "rt_1_tf_trained_for_000058240",
    "rt_1_1k": "rt_1_tf_trained_for_000001120",
}

TASKS = [
    "google_robot_pick_customizable",
    "google_robot_pick_customizable_ycb",
    "google_robot_pick_customizable_no_overlay",
    "google_robot_move_near_customizable",
    "google_robot_move_near_customizable_ycb",
    "google_robot_move_near_customizable_no_overlay",
    "widowx_put_on_customizable",
    "widowx_put_on_customizable_ycb",
    "widowx_put_on_customizable_no_overlay",
    "widowx_put_in_customizable",
    "widowx_put_in_customizable_ycb",
    "widowx_put_in_customizable_no_overlay",
]

ckpt_dir = str(PACKAGE_DIR) + '/../checkpoints'

sapien.render_config.rt_use_denoiser = True


class VLAInterface:
    def __init__(self, task, model_name, instability_methods):

        self.model_name = model_name
        self.token_metricsFast = TokenMetricsFast()
        self.instability_methods = instability_methods
        self.instability_method_fns = {
            name: getattr(uncerMetrics, f"compute_{name}")
            for name in self.instability_methods
        }
        self.instability_TCP_method_fns = {
            name: getattr(uncerMetrics, f"compute_TCP_{name}")
            for name in self.instability_methods
        }
        if task in TASKS:
            self.task = task
        else:
            raise ValueError(task)
        if "google" in self.task:
            self.policy_setup = "google_robot"
        else:
            self.policy_setup = "widowx_bridge"
        if "openvla" in model_name:
            from simpler_env.policies.openvla.openvla_model import OpenVLAInference
            self.model = OpenVLAInference(model_type='../checkpoints/openvla-7b', policy_setup=self.policy_setup)
            #self.followUpModel = OpenVLAInference(model_type=model_name, policy_setup=self.policy_setup)
            self.variability_models=[OpenVLAInference(model_type='../checkpoints/openvla-7b' , policy_setup=self.policy_setup) for i in range(0,uncerMetrics.VARIABILITY)]
            
        elif "pi0" in model_name:
            from simpler_env.policies.lerobotpi.pi0_or_fast import LerobotPiFastInference
            if self.policy_setup == "widowx_bridge":
                model_path = "../checkpoints/lerobot-pi0-bridge"
            else:
                model_path = "../checkpoints/lerobot-pi0-fractal"

            self.model = LerobotPiFastInference(saved_model_path=model_path, policy_setup=self.policy_setup)
            self.variability_models=[LerobotPiFastInference(saved_model_path=model_path, policy_setup=self.policy_setup) for i in range(0,uncerMetrics.VARIABILITY)]
            #self.followUpModel = LerobotPiFastInference(saved_model_path=model_path, policy_setup=self.policy_setup)
        elif "spatialvla" in model_name:
            from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference

            self.model = SpatialVLAInference(saved_model_path="../checkpoints/spatialvla-4b-mix-224-pt",policy_setup=self.policy_setup)
            self.variability_models=[SpatialVLAInference(saved_model_path="../checkpoints/spatialvla-4b-mix-224-pt",policy_setup=self.policy_setup) for i in range(0,uncerMetrics.VARIABILITY)]
            
            #self.followUpModel = SpatialVLAInference(model_type=model_name, policy_setup=self.policy_setup)
        else:
            raise ValueError(model_name)

    def run_interface(self, seed=None, options=None, task_type=None):

        env = simpler_env.make(self.task)
        obs, reset_info = env.reset(seed=seed, options=options)
        instruction = env.unwrapped.get_language_instruction()
        self.model.reset(instruction)
        [self.variability_models[i].reset(instruction) for i in range(0, len(self.variability_models))]
        image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
        images = [image]
        predicted_terminated, success, truncated = False, False, False
        timestep = 0
        episode_stats = {}
        actions = []
        tcp_poses = []
        token_based_dict = {'entropy': [],
                            'token_prob': [],
                            'pcs': [],
                            'deepgini': []}
        exec_times_dict={'inference':[],
                         'token-based':[],
                         'execution_variability':[],
                         'optimal_trajectory':[],
                         'trajectory_instability_gradients':[],
                         'instability':[]}
        variability = []
        optimal_traj = []
        traj_inst_gradients = []
        gradients = []
        traj_instability = {method_name: [[0, 0, 0, 0, 0, 0, 0] for i in range(0, TIME_RANGE - 1)] for method_name in
                            self.instability_methods}
        traj_instability_tcp = {method_name: [[0, 0, 0] for i in range(0, TIME_RANGE - 1)] for method_name in
                                self.instability_methods}
        traj_uncerActions = np.array([])
        traj_uncerTcp = []
        tcp_pose = obs['extra']['tcp_pose']

        if task_type == "grasp":
                
            object_pose = env.unwrapped.obj_pose
            total_dist = np.abs(np.linalg.norm(tcp_pose[:3] - object_pose.p))
        else:
            object_pose = env.unwrapped.source_obj_pose
            final_pose = env.unwrapped.target_obj_pose
            total_dist = np.abs(np.linalg.norm(tcp_pose[:3] - object_pose.p)) + np.abs(np.linalg.norm(final_pose.p - object_pose.p))

        while not (predicted_terminated or truncated):
            init_time = time.time()
            # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
            if "pi0" in self.model_name:
                raw_action, action = self.model.step(image, instruction, eef_pos=obs["agent"]["eef_pos"])
            elif "spatialvla" in self.model_name:
                raw_action, action = self.model.step(image, instruction)
            else:
                raw_action, action = self.model.step(image)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            obs, reward, success, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            )
            print(timestep, info)
            tcp_pose = obs['extra']['tcp_pose']
            tcp_poses.append(tcp_pose.tolist())
            episode_stats[timestep] = info
            action_norm = uncerUtils.normalize_action(action, env.action_space)
            actions.append(action_norm)

            # ----------------------------- UNCERTAINTY METRICS ------------------------------------------------------------------------

            if len(gradients) < TIME_RANGE:
                gradients.append(tcp_pose.tolist())
                traj_uncerActions = np.append(traj_uncerActions, action_norm)
                traj_uncerTcp.append(tcp_pose.tolist())
            else:
                gradients[-1] = tcp_pose.tolist()
                traj_uncerActions[-1] = action_norm
                traj_uncerTcp[-1] = tcp_pose.tolist()
            time_execution = time.time()
            print(f"Time to execute: {time_execution - init_time}")
            exec_times_dict['inference'].append(time_execution - init_time)
            # Token-based uncertainty

            if "pi0" in self.model_name:
                expert_logits = raw_action['expert_scores']
                token_uq_metrics = self.token_metricsFast.compute_norm_inv_token_metrics(expert_logits)


            else:
                logits = torch.stack(raw_action['scores'], dim=0).squeeze(1)
                token_uq_metrics = self.token_metricsFast.compute_norm_inv_token_metrics(logits)

            token_based_dict['entropy'].append(token_uq_metrics[0])
            token_based_dict['token_prob'].append(token_uq_metrics[1])
            token_based_dict['pcs'].append(token_uq_metrics[2])
            token_based_dict['deepgini'].append(token_uq_metrics[3])
            
            time_execution2 = time.time()

            print(f"Time to calculate the token-based ones: {time_execution2 - time_execution}")
            exec_times_dict['token-based'].append(time_execution2 - time_execution)
            
            # Execution variability
            result = uncerMetrics.compute_execution_variability(self.variability_models, image, env.action_space, instruction, obs, self.model_name)
            variability.append(result)
            
            time_execution3 = time.time()
            print(f"Time to calculate the execution variability: {time_execution3 - time_execution2}")
            exec_times_dict['execution_variability'].append(time_execution3 - time_execution2)
            # Optimal Trajectory
            if task_type == "grasp":
                object_pose = env.unwrapped.obj_pose
                final_pose = None
            else:
                object_pose = env.unwrapped.source_obj_pose
                final_pose = env.unwrapped.target_obj_pose

            if task_type == "grasp":

                optimal_traj.append(np.abs(np.linalg.norm(tcp_pose[:3] - object_pose.p))/total_dist)

            elif task_type == "put-in" or task_type == "put-on":
                if info["is_src_obj_grasped"]:
                    optimal_traj.append(np.abs(np.linalg.norm(tcp_pose[:3] - final_pose.p))/ total_dist)
                else:
                    optimal_traj.append((np.abs(np.linalg.norm(tcp_pose[:3] - object_pose.p)) + np.abs(np.linalg.norm(object_pose.p - final_pose.p)))/total_dist)

            else:
                if np.abs(np.linalg.norm(tcp_pose[:3] - object_pose.p)) < 0.04:
                    optimal_traj.append(np.abs(np.linalg.norm(tcp_pose[:3] - final_pose.p))/total_dist)
                else:
                    optimal_traj.append((np.abs(np.linalg.norm(tcp_pose[:3] - object_pose.p)) + np.abs(np.linalg.norm(object_pose.p - final_pose.p)))/total_dist)

            time_execution4 = time.time()
            print(f"Time to calculate the optimal trajectory: {time_execution4 - time_execution3}")
            exec_times_dict['optimal_trajectory'].append(time_execution4 - time_execution3)
            # Trajectory Instability Gradients
            if len(gradients) == TIME_RANGE:

                result = uncerMetrics.compute_TCP_jerk_instability_gradient(gradients)
                traj_inst_gradients = np.append(traj_inst_gradients, result[-3])

                gradients = np.roll(gradients, -1, axis=0)
                # input("Press Enter to continue...")
                # time.sleep(300000)
            elif len(gradients) == TIME_RANGE - 1:

                result = uncerMetrics.compute_TCP_jerk_instability_gradient(gradients)
                traj_inst_gradients = result[:-2]

            time_execution5 = time.time()
            print(f"Time to calculate the trajectory instability gradients: {time_execution5 - time_execution4}")
            exec_times_dict['trajectory_instability_gradients'].append(time_execution5 - time_execution4)

            time_execution7 = time.time()
            print(f"Time to calculate the metamorphic ps: {time_execution7 - time_execution5}")
            # Trajecotry Instability
            if len(traj_uncerActions) == TIME_RANGE:
                i = 2
                for name, fn in self.instability_method_fns.items():
                    result = fn(traj_uncerActions[i:])
                    traj_instability[name].append(result)
                    i = i - 1
                i = 2
                # print(uncerTcp)
                for name, fn in self.instability_TCP_method_fns.items():
                    result_tcp = fn(traj_uncerTcp[i:])
                    # print(uncerTcp[i:])
                    # print(result_tcp)
                    traj_instability_tcp[name].append(result_tcp)
                    i = i - 1
                traj_uncerActions = np.roll(traj_uncerActions, -1, axis=0)
                traj_uncerTcp = np.roll(traj_uncerTcp, -1, axis=0)
            time_execution8 = time.time()
            print(f"Time to calculate the trajectory instability: {time_execution8 - time_execution7}")
            exec_times_dict['instability'].append(time_execution8 - time_execution7)

            # ---------------------------------------------------------------------------------------------------------------------------

            # update image observation
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            timestep += 1

        print(f"Episode success: {success}")
        env.close()
        del env
        return images, episode_stats, actions, tcp_poses, token_based_dict, variability, optimal_traj, traj_inst_gradients, traj_instability, traj_instability_tcp, exec_times_dict


class VLAInterfaceLM(VLAInterface):
    def run_interface(self, seed=None, options=None, instruction=None):
        env = simpler_env.make(self.task)
        obs, reset_info = env.reset(seed=seed, options=options)
        if not instruction:
            instruction = env.get_language_instruction()
        self.model.reset(instruction)
        print(instruction)
        print("Reset info", reset_info)

        image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
        images = [image]
        predicted_terminated, success, truncated = False, False, False
        timestep = 0
        episode_stats = {}
        while not (predicted_terminated or truncated):
            # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
            raw_action, action = self.model.step(image)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            obs, reward, success, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            )
            print(timestep, info)
            episode_stats[timestep] = info
            # update image observation
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            timestep += 1

        print(f"Episode success: {success}")
        env.close()
        del env
        return images, episode_stats


if __name__ == '__main__':
    task_name = "google_robot_pick_customizable"
    model = "rt_1_x"  # @param ["rt_1_x", "rt_1_400k", "rt_1_58k", "rt_1_1k", "octo-base", "octo-small", "openvla-7b]

    vla = VLAInterface(model_name=model, task=task_name)
