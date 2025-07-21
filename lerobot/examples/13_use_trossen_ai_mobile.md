This tutorial explains how to use [Trossen AI Mobile](https://www.trossenrobotics.com/mobile-ai) with LeRobot.

## Setup

Follow the [documentation from Trossen Robotics](https://docs.trossenrobotics.com/trossen_arm/main/getting_started/hardware_setup.html) for setting up the hardware.


## Install LeRobot

On your computer:

1. [Install Miniconda](https://docs.anaconda.com/miniconda/#quick-command-line-install):
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

2. Restart shell or `source ~/.bashrc`

3. Create and activate a fresh conda environment for lerobot
```bash
conda create -y -n lerobot python=3.10 && conda activate lerobot
```

4. Clone LeRobot:
```bash
git clone -b trossen-ai https://github.com/Interbotix/lerobot.git ~/lerobot
```

5. Install LeRobot with dependencies for the Trossen AI arms (trossen-arm) and cameras (pyrealsense2):

```bash
cd ~/lerobot && pip install --no-binary=av -e ".[trossen_ai]"
```

6. Install ffmpeg for miniconda
```bash
conda install -c conda-forge 'ffmpeg>=7.0' -y
```

## Teleoperate

By running the following code, you can start your first **SAFE** teleoperation:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=trossen_ai_mobile \
  --robot.max_relative_target=5 \
  --control.type=teleoperate
```

By adding `--robot.max_relative_target=5`, we override the default value for `max_relative_target` defined in [`TrossenAIMobilelRobot`](../lerobot/common/robot_devices/robots/configs.py). It is expected to be `5` to limit the magnitude of the movement for more safety, but the teleoperation won't be smooth. When you feel confident, you can disable this limit by adding `--robot.max_relative_target=null` to the command line:

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=trossen_ai_mobile \
  --robot.max_relative_target=null \
  --control.type=teleoperate
```
By adding `--robot.force_feedback_gain=0.1`, we override the default value for `force_feedback_gain` defined in [`TrossenAIMobileRobot`](../lerobot/common/robot_devices/robots/configs.py). This enables **force feedback** from the follower arm to the leader arm — meaning the user can **feel contact forces** when the robot interacts with external objects (e.g., gripping or bumping into something). A typical starting value is `0.1` for a responsive feel. The default value is `0.0`, which disables force feedback.

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=trossen_ai_mobile \
  --robot.max_relative_target=null \
  --robot.force_feedback_gain=0.1 \
  --control.type=teleoperate
```
This parameter can be used in both teleoperate and record modes, depending on whether you want the operator to feel contact feedback during data collection.

## Record a dataset

Once you're familiar with teleoperation, you can record your first dataset with Trossen AI Mobile Kit.

If you want to use the Hugging Face hub features for uploading your dataset and you haven't previously done it, make sure you've logged in using a write-access token, which can be generated from the [Hugging Face settings](https://huggingface.co/settings/tokens):

```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Store your Hugging Face repository name in a variable to run these commands:

```bash
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
```

Record 2 episodes and upload your dataset to the hub:

Note: We recommend using `--control.num_image_writer_threads_per_camera=8` for best results while recording episodes.

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=trossen_ai_mobile \
  --robot.max_relative_target=null \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Testing episode recording for Trossen AI Mobile." \
  --control.repo_id=${HF_USER}/trossen_ai_mobile_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.push_to_hub=true \
  --control.num_image_writer_threads_per_camera=8 \
  --control.display_cameras=false
```

The **SLATE base** works in two modes:
- **Torque OFF** (default): You can push the base around manually.
- **Torque ON**: Enables the motors so you can control the base using the **SLATE remote controller**.

To enable torque-on mode during recording, add the following argument:
```bash
--robot.enable_motor_torque=true
```

For more information about the SLATE remote controller, refer to the official documentation:
[SLATE RC Controller Guide](https://docs.trossenrobotics.com/slate_docs/operation/rc_controller.html)

## Visualize a dataset

If you uploaded your dataset to the hub with `--control.push_to_hub=true`, you can [visualize your dataset online](https://huggingface.co/spaces/lerobot/visualize_dataset) by copy pasting your repo id given by:
```bash
echo ${HF_USER}/trossen_ai_mobile_test
```

If you didn't upload with `--control.push_to_hub=false`, you can also visualize it locally with:
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/trossen_ai_mobile_test
```

## Replay an episode


Now try to replay the first episode on your robot:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=trossen_ai_mobile \
  --robot.max_relative_target=null \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/trossen_ai_mobile_test \
  --control.episode=1 \
  --robot.enable_motor_torque=true
```

Note: For replaying an episode, you need to turn on motor torque using ``--robot.enable_motor_torque=true``, so that the robot can actively follow the trajectory instead of remaining in a passive (torque-off) state.

## Train a policy

To train a policy to control your robot, use the [`python lerobot/scripts/train.py`](../lerobot/scripts/train.py) script. A few arguments are required. Here is an example command:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/trossen_ai_mobile_test \
  --policy.type=act \
  --output_dir=outputs/train/act_trossen_ai_mobile_test \
  --job_name=act_trossen_ai_mobile_test \
  --device=cuda \
  --wandb.enable=true
```

Let's explain it:
1. We provided the dataset as argument with `--dataset.repo_id=${HF_USER}/trossen_ai_mobile_test`.
2. We provided the policy with `policy.type=act`. This loads configurations from [`configuration_act.py`](../lerobot/common/policies/act/configuration_act.py). Importantly, this policy will automatically adapt to the number of motor sates, motor actions and cameras of your robot (e.g. `laptop` and `phone`) which have been saved in your dataset.
4. We provided `device=cuda` since we are training on a Nvidia GPU, but you could use `device=mps` to train on Apple silicon.
5. We provided `wandb.enable=true` to use [Weights and Biases](https://docs.wandb.ai/quickstart) for visualizing training plots. This is optional but if you use it, make sure you are logged in by running `wandb login`.

For more information on the `train` script see the previous tutorial: [`examples/4_train_policy_with_script.md`](../examples/4_train_policy_with_script.md)

Training should take several hours. You will find checkpoints in `outputs/train/act_trossen_ai_mobile_test/checkpoints`.

## Evaluate your policy

You can use the `record` function from [`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py) but with a policy checkpoint as input. For instance, run this command to record 10 evaluation episodes:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=trossen_ai_mobile \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=${HF_USER}/eval_act_trossen_ai_mobile_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/act_trossen_ai_mobile_test/checkpoints/last/pretrained_model \
  --control.num_image_writer_processes=1 \
  --robot.enable_motor_torque=true
```

Note: For evaluation, you need to turn on motor torque using ``--robot.enable_motor_torque=true``, so that the robot can actively follow the trajectory instead of remaining in a passive (torque-off) state.

As you can see, it's almost the same command as previously used to record your training dataset. Two things changed:
1. There is an additional `--control.policy.path` argument which indicates the path to your policy checkpoint with  (e.g. `outputs/train/eval_act_trossen_ai_mobile_test/checkpoints/last/pretrained_model`). You can also use the model repository if you uploaded a model checkpoint to the hub (e.g. `${HF_USER}/act_trossen_ai_mobile_test`).
2. The name of dataset begins by `eval` to reflect that you are running inference (e.g. `${HF_USER}/eval_act_trossen_ai_mobile_test`).
3. We use `--control.num_image_writer_processes=1` instead of the default value (`0`). On our computer, using a dedicated process to write images from the 4 cameras on disk allows to reach constant 30 fps during inference. Feel free to explore different values for `--control.num_image_writer_processes`.

## More

Follow this [previous tutorial](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md#4-train-a-policy-on-your-data) for a more in-depth explanation.
