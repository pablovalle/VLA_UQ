"""
Name   : utils.py
Author : Pablo Valle
Time   : 05/22/2025
"""

import numpy as np
import copy
def normalize_action(action, normalization_values):
    new_action=copy.deepcopy(action)
    low =  np.array(normalization_values.low)
    high =  np.array(normalization_values.high)
   
    action_raw = np.concatenate([
        np.array(action["world_vector"]),
        np.array(action["rot_axangle"]),
        np.array(action["gripper"])
    ])
    length_world=len(action["world_vector"])
    length_angles=len(action["world_vector"])+len(action["rot_axangle"])
    lenth_gripper=len(action["world_vector"])+len(action["rot_axangle"])+len(action["gripper"])
    
    normalized_action = (action_raw - low) / (high - low)
    normalized_action = np.clip(normalized_action, 0.0, 1.0)

    new_action['world_vector'] = normalized_action[:length_world]
    new_action["rot_axangle"] = normalized_action[length_world:length_angles]
    new_action["gripper"]= normalized_action[length_angles:lenth_gripper]  # assuming 1-dim gripper
        
    return new_action

def action_uncertainty(action, mutated_action):
    world_vectors = np.array([
        action['world_vector'],
        mutated_action['world_vector']
    ])

    rot_axangles = np.array([
        action['rot_axangle'],
        mutated_action['rot_axangle']
    ])

    grippers = np.array([
        action['gripper'],
        mutated_action['gripper']
    ])
    
    metamorphic = np.concatenate([
        np.std(world_vectors, axis=0),
        np.std(rot_axangles, axis=0),
        [np.std(grippers)]
    ])

    return metamorphic




