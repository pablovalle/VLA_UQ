import gymnasium as gym
import mani_skill2_real2sim.envs

ENVIRONMENTS = [
    "google_robot_pick_coke_can",
    "google_robot_pick_horizontal_coke_can",
    "google_robot_pick_vertical_coke_can",
    "google_robot_pick_standing_coke_can",
    "google_robot_pick_object",
    "google_robot_move_near_v0",
    "google_robot_move_near_v1",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_open_top_drawer",
    "google_robot_open_middle_drawer",
    "google_robot_open_bottom_drawer",
    "google_robot_close_drawer",
    "google_robot_close_top_drawer",
    "google_robot_close_middle_drawer",
    "google_robot_close_bottom_drawer",
    "google_robot_place_in_closed_drawer",
    "google_robot_place_in_closed_top_drawer",
    "google_robot_place_in_closed_middle_drawer",
    "google_robot_place_in_closed_bottom_drawer",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
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

ENVIRONMENT_MAP = {
    "google_robot_pick_coke_can": ("GraspSingleOpenedCokeCanInScene-v0", {}),
    "google_robot_pick_horizontal_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"lr_switch": True},
    ),
    "google_robot_pick_vertical_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"laid_vertically": True},
    ),
    "google_robot_pick_standing_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"upright": True},
    ),
    "google_robot_pick_object": ("GraspSingleRandomObjectInScene-v0", {}),
    "google_robot_move_near": ("MoveNearGoogleBakedTexInScene-v1", {}),
    "google_robot_move_near_v0": ("MoveNearGoogleBakedTexInScene-v0", {}),
    "google_robot_move_near_v1": ("MoveNearGoogleBakedTexInScene-v1", {}),
    "google_robot_open_drawer": ("OpenDrawerCustomInScene-v0", {}),
    "google_robot_open_top_drawer": ("OpenTopDrawerCustomInScene-v0", {}),
    "google_robot_open_middle_drawer": ("OpenMiddleDrawerCustomInScene-v0", {}),
    "google_robot_open_bottom_drawer": ("OpenBottomDrawerCustomInScene-v0", {}),
    "google_robot_close_drawer": ("CloseDrawerCustomInScene-v0", {}),
    "google_robot_close_top_drawer": ("CloseTopDrawerCustomInScene-v0", {}),
    "google_robot_close_middle_drawer": ("CloseMiddleDrawerCustomInScene-v0", {}),
    "google_robot_close_bottom_drawer": ("CloseBottomDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_drawer": ("PlaceIntoClosedDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_top_drawer": ("PlaceIntoClosedTopDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_middle_drawer": ("PlaceIntoClosedMiddleDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_bottom_drawer": ("PlaceIntoClosedBottomDrawerCustomInScene-v0", {}),
    "google_robot_place_apple_in_closed_top_drawer": (
        "PlaceIntoClosedTopDrawerCustomInScene-v0", 
        {"model_ids": "baked_apple_v2"}
    ),
    "widowx_spoon_on_towel": ("PutSpoonOnTableClothInScene-v0", {}),
    "widowx_carrot_on_plate": ("PutCarrotOnPlateInScene-v0", {}),
    "widowx_stack_cube": ("StackGreenCubeOnYellowCubeBakedTexInScene-v0", {}),
    "widowx_put_eggplant_in_basket": ("PutEggplantInBasketScene-v0", {}),
    "google_robot_pick_customizable": ("GraspSingleCustomizable-v0", {}),
    "google_robot_pick_customizable_ycb": ("GraspSingleCustomizableYCB-v0", {}),
    "google_robot_pick_customizable_no_overlay": ("GraspSingleCustomizableNoOverlay-v0", {}),
    "google_robot_move_near_customizable": ("MoveNearCustomizable-v0", {}),
    "google_robot_move_near_customizable_ycb": ("MoveNearCustomizableYCB-v0", {}),
    "google_robot_move_near_customizable_no_overlay": ("MoveNearCustomizableNoOverlay-v0", {}),
    "widowx_put_on_customizable": ("PutOnCustomizable-v0", {}),
    "widowx_put_in_customizable": ("PutInCustomizable-v0", {}),
    "widowx_put_on_customizable_ycb": ("PutOnCustomizableYCB-v0", {}),
    "widowx_put_in_customizable_ycb": ("PutInCustomizableYCB-v0", {}),
    "widowx_put_on_customizable_no_overlay": ("PutOnCustomizableNoOverlay-v0", {}),
    "widowx_put_in_customizable_no_overlay": ("PutInCustomizableNoOverlay-v0", {}),
}


def make(task_name):
    """Creates simulated eval environment from task name."""
    assert task_name in ENVIRONMENTS, f"Task {task_name} is not supported. Environments: \n {ENVIRONMENTS}"
    env_name, kwargs = ENVIRONMENT_MAP[task_name]
    kwargs["prepackaged_config"] = True
    #env = gym.make(env_name, obs_mode="rgbd",max_episode_steps=80, **kwargs)
    env = gym.make(env_name, obs_mode="rgbd", **kwargs)

    return env
