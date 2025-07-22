"""
Name   : uncertainty_metrics.py
Author : Pablo Valle
Time   : 05/22/2025
"""

import numpy as np
import uncertainty.utils as uncerUtils



VARIABILITY=1
def compute_position_instability(actions: np.ndarray) -> np.ndarray:
    action_array = np.array([
        np.concatenate((
            np.array(d["world_vector"]),
            np.array(d["rot_axangle"]),
            np.array(d["gripper"])
        ))
        for d in actions
    ])
    T, M = action_array.shape
    if T < 2:
        raise ValueError("At least 4 time steps are required to compute instability.")
    
    delta_a = np.abs(np.diff(action_array, axis=0))
    instability_per_t = np.sum(delta_a, axis=0) / delta_a.shape[0]
    return instability_per_t

def compute_velocity_instability(actions: np.ndarray) -> np.ndarray:
    action_array = np.array([
        np.concatenate((
            np.array(d["world_vector"]),
            np.array(d["rot_axangle"]),
            np.array(d["gripper"])
        ))
        for d in actions
    ])
    T, M = action_array.shape
    if T < 3:
        raise ValueError("At least 4 time steps are required to compute instability.")
    
    delta_a = np.diff(action_array, axis=0)
    delta2_a = np.abs(np.diff(delta_a, axis=0)) /2
    instability_per_t = np.sum(delta2_a, axis=0)  / delta2_a.shape[0]
    #print(instability_per_t)
    return instability_per_t

def compute_acceleration_instability(actions: np.ndarray) -> np.ndarray:
    action_array = np.array([
        np.concatenate((
            np.array(d["world_vector"]),
            np.array(d["rot_axangle"]),
            np.array(d["gripper"])
        ))
        for d in actions
    ])
    T, M = action_array.shape
    if T < 4:
        raise ValueError("At least 4 time steps are required to compute instability.")
    
    delta_a = np.diff(action_array, axis=0)
    delta2_a = np.diff(delta_a, axis=0)
    jerk = np.abs(np.diff(delta2_a, axis=0)) /4
    instability_per_t = np.sum(jerk, axis=0) / jerk.shape[0]
    #print(instability_per_t)
    return instability_per_t

def compute_TCP_position_instability(poses: list) -> np.ndarray:
    # Convert to np array and slice to x, y, z
    pos_array = np.array(poses)[:, :3]  # shape (T, 3)
    T = pos_array.shape[0]
    
    if T < 2:
        raise ValueError("At least 2 time steps are required to compute instability.")
    
    delta = np.abs(np.diff(pos_array, axis=0))  # (T-1, 3)
    instability_per_axis = np.sum(delta, axis=0) / delta.shape[0] 
    return instability_per_axis

def compute_TCP_velocity_instability(poses: list) -> np.ndarray:
    pos_array = np.array(poses)[:, :3]
    T = pos_array.shape[0]
    
    if T < 3:
        raise ValueError("At least 3 time steps are required to compute velocity instability.")
    
    delta = np.diff(pos_array, axis=0)
    delta2 = np.abs(np.diff(delta, axis=0)) /2  # (T-2, 3)
    instability_per_axis = np.sum(delta2, axis=0) / delta2.shape[0] 
    return instability_per_axis

def compute_TCP_acceleration_instability(poses: list) -> np.ndarray:
    pos_array = np.array(poses)[:, :3]
    T = pos_array.shape[0]
    
    if T < 4:
        raise ValueError("At least 4 time steps are required to compute acceleration instability.")
    
    delta = np.diff(pos_array, axis=0)
    delta2 = np.diff(delta, axis=0)
    jerk = np.abs(np.diff(delta2, axis=0)) /4 # (T-3, 3)
    instability_per_axis = np.sum(jerk, axis=0) / jerk.shape[0] 
    return instability_per_axis

def compute_TCP_jerk_instability_gradient(poses: list) -> np.ndarray:
    dt=1
    x=np.array(poses)[:, 0]
    y=np.array(poses)[:, 1]
    z=np.array(poses)[:, 2]

    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vz = np.gradient(z, dt)

    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    az = np.gradient(vz, dt)

    jx = np.gradient(ax, dt)
    jy = np.gradient(ay, dt)
    jz = np.gradient(az, dt)

    # Jerk magnitude
    jerk_magnitude = np.sqrt(jx**2 + jy**2 + jz**2)
    rms_jerk = np.sqrt(np.mean(jerk_magnitude**2))

    return jerk_magnitude



def compute_execution_variability(variability_models, image, action_space, instruction, obs, model_name):

    
    actions=[]

    for i in range(0, len(variability_models)):
        if "pi0" in model_name:
            raw_action, action = variability_models[i].step(image, instruction, eef_pos=obs["agent"]["eef_pos"])
        elif "spatialvla" in model_name:
            raw_action, action = variability_models[i].step(image, instruction)
            #print(raw_action)
            #time.sleep(30000)
            #scores= raw_action['scores']
        else:
            raw_action, action = variability_models[i].step(image)
        #raw_action, action = variability_models[i].step(image, instruction, eef_pos=obs["agent"]["eef_pos"])
        action=uncerUtils.normalize_action(action, action_space)
        #print(action)
        actions.append(action)
    world_vectors = np.array([d["world_vector"] for d in actions])
    rot_axangles = np.array([d["rot_axangle"] for d in actions])
    grippers = np.array([d["gripper"][0] for d in actions])

    
    # Compute standard deviations
    variability = np.concatenate([
        np.std(world_vectors, axis=0),
        np.std(rot_axangles, axis=0),
        [np.std(grippers)]
    ])
    return variability

    
  