"""
Name   : run_fuzzer_allMetrics.py
Author : PABLO VALLE
Time   : 8/8/24
"""
import argparse
import numpy as np
from model_interface_allMetrics import VLAInterface
from pathlib import Path
from tqdm import tqdm
import json
import os
import shutil
import cv2
import re
import torch
import subprocess


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"



# Setup paths
PACKAGE_DIR = Path(__file__).parent.resolve()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

class StableJSONizer(json.JSONEncoder):
    def default(self, obj):
        return super().encode(bool(obj)) \
            if isinstance(obj, np.bool_) \
            else super().default(obj)
    
def convert_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  
        return obj.item()
    elif isinstance(obj, list):
        return [convert_to_native(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    return obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="VLA Fuzzing")
    parser.add_argument('-d', '--data', type=str, help="Testing data")
    parser.add_argument('-o', '--output', type=str, default=None, help="Output path, e.g., folder")
    parser.add_argument('-io', '--image_output', type=str, default=None, help="Image output path, e.g., folder")
    parser.add_argument('-s', '--seed', type=int, default=None, help="Random Seed")
    parser.add_argument('-m', '--model', type=str,
                        choices=["rt_1_x", "rt_1_400k", "rt_1_58k", "rt_1_1k", "octo-base", "octo-small", "openvla-7b", "pi0", "spatialvla-4b"],
                        default="spatialvla-4b",
                        help="VLA model")
    parser.add_argument('-r', '--resume', type=bool, default=True, help="Resume from where we left.")

    args = parser.parse_args()

    random_seed = args.seed if args.seed else np.random.randint(0, 4294967295)  # max uint32
    
    task_data="t-grasp_n-1000_o-m3_s-2498586606.json"
   

    data_path = args.data if args.data else str(PACKAGE_DIR) + "/../data/"+task_data

    dataset_name = data_path.split('/')[-1]

    match = re.search(r't-(.*?)_n', dataset_name)

    if match:
        task_type = match.group(1)
        #print(task_type)  # Output: put-in
    else:
        print("No match found")

    metamorphic_VS_methods=['vs1','vs2', 'vs3', 'vs4']
    instability_methods=['position_instability','velocity_instability', 'acceleration_instability']

    if "grasp" in dataset_name:
        vla = VLAInterface(model_name=args.model, task="google_robot_pick_customizable", instability_methods=instability_methods)
    elif "move" in dataset_name:
        vla = VLAInterface(model_name=args.model, task="google_robot_move_near_customizable",  instability_methods=instability_methods)
    elif "put-on" in dataset_name:
        vla = VLAInterface(model_name=args.model, task="widowx_put_on_customizable",  instability_methods=instability_methods)
    elif "put-in" in dataset_name:
        vla = VLAInterface(model_name=args.model, task="widowx_put_in_customizable",  instability_methods=instability_methods)
    else:
        raise NotImplementedError

    with open(data_path, 'r') as f:
        tasks = json.load(f)

    if args.output:
        result_dir = args.output + data_path.split('/')[-1].split(".")[0]
    else:
        result_dir = str(PACKAGE_DIR) + "/../results2/" + data_path.split('/')[-1].split(".")[0]
    os.makedirs(result_dir, exist_ok=True)
    result_dir += f'/{args.model}'#_{random_seed}'
    if not args.resume:
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    
    #if args.image_output:
    #    image_dir = args.image_output + data_path.split('/')[-1].split(".")[0]
    #    os.makedirs(image_dir, exist_ok=True)
    #    image_dir += f'/{args.model}_{random_seed}'
    #    os.makedirs(image_dir, exist_ok=True)
    #else:
    #    image_dir = None

    for idx in tqdm(range(round(tasks["num"]/2))):
        if args.resume and os.path.exists(result_dir + f"/allMetrics/{idx}/" + '/log.json'):  # if resume allowed then skip the finished runs.
            continue
        options = tasks[str(idx)]
        images, episode_stats, actions, tcp_poses, uncertainty_token, uncertainty_variability, optimal_traj, traj_inst_gradients, metamorphic_VS_results, metamorphic_VS_actions, mutated_images, metamorphic_PS_results, metamorphic_PS_actions, traj_instability, traj_instability_tcp, exec_times_dict  = vla.run_interface(seed=random_seed, options=options, task_type=task_type, template=templates_ps1)
        os.makedirs(result_dir + f"/allMetrics/{idx}", exist_ok=True)

# ----------------------------- UNCERTAINTY METRICS ----------------------------------------------------------------------------------------------------
        #Execution time
        for method_name, values in exec_times_dict.items():
            file_path = result_dir + f"/allMetrics/{idx}/" + f"exec_time_{method_name}.json"
            with open(file_path, "w") as f:
                json.dump(convert_to_native(values), f, indent=2)

        
        # Token-based metrics
        for method_name, values in uncertainty_token.items():
            file_path = result_dir + f"/allMetrics/{idx}/" + f"Token_based_{method_name}.json"
            with open(file_path, "w") as f:
                json.dump(convert_to_native(values), f, indent=2)

        

        # Execution Variability
        serialized_values = [v.tolist() if isinstance(v, np.ndarray) else v for v in uncertainty_variability]
        file_path = result_dir + f"/allMetrics/{idx}/" + f"Execution_Variability.json"
        with open(file_path, "w") as f:
            json.dump(serialized_values, f, indent=4)


        # Optimal trejectory
        optimal_traj=np.array(optimal_traj).tolist()
        uncertainty = np.diff(optimal_traj, prepend=optimal_traj[0])
        normalized_uncertainty = ((uncertainty + 1) / 2).tolist()
        with open(result_dir + f"/allMetrics/{idx}/" + '/Optimal_Trajectory.json', "w") as f:
            json.dump(normalized_uncertainty, f, indent=4)


        # Trajectory Instability Gradients
        serialized_values = [v.tolist() if isinstance(v, np.ndarray) else v for v in traj_inst_gradients]
        file_path = result_dir + f"/allMetrics/{idx}/" + f"Trajectory_Instability_Gradients.json"
        with open(file_path, "w") as f:
            json.dump(serialized_values, f, indent=4)
        
       
        # Trajectory Instability
        for method_name, values in traj_instability.items():
            serialized_values = [v.tolist() if isinstance(v, np.ndarray) else v for v in values]
            file_path = result_dir + f"/allMetrics/{idx}/" + f"{method_name}.json"
            with open(file_path, "w") as f:
                json.dump(serialized_values, f, indent=4)

        for method_name, values in traj_instability_tcp.items():
            serialized_values = [v.tolist() if isinstance(v, np.ndarray) else v for v in values]
            file_path = result_dir + f"/allMetrics/{idx}/" + f"TCP_{method_name}.json"
            with open(file_path, "w") as f:
                json.dump(serialized_values, f, indent=4)


#-------------------------------------------------------------------------------------------------------------------------------------------------------


        with open(result_dir + f"/allMetrics/{idx}/" + '/log.json', "w") as f:
            json.dump(episode_stats, f, cls=StableJSONizer)
        #print(actions)
        json_ready_actions=[
            {key: value.tolist() for key, value in entry.items()}
            for entry in actions
        ]
        with open(result_dir + f"/allMetrics/{idx}/" + '/actions.json', "w") as f:
            json.dump(json_ready_actions, f, indent=2)
        
        with open(result_dir + f"/allMetrics/{idx}/" + '/tcp_poses.json', "w") as f:
            json.dump(tcp_poses, f, indent=2)
        #if image_dir:
            #os.makedirs(image_dir + f"/{idx}", exist_ok=True)
            
        video_path = os.path.join(result_dir , f"allMetrics/{idx}", f"{idx}_simulation_orig.mp4")
        video_path_dest = os.path.join(result_dir , f"allMetrics/{idx}", f"{idx}_simulation.mp4")

        # Get frame size from the first image
        height, width = images[0].shape[:2]

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        fps = 10  # Set your desired FPS
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        imagePath=os.path.join(result_dir , f"allMetrics/{idx}","images")
        os.makedirs(imagePath)
        for img_idx in range(len(images)):
            frame = images[img_idx]

            # Ensure frame is uint8 and in BGR (OpenCV uses BGR not RGB)
            if frame.dtype != np.uint8:
                frame = (255 * (frame - frame.min()) / (frame.ptp() + 1e-8)).astype(np.uint8)
            
            if frame.shape[2] == 3:  # RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            out.write(frame)
            #im = Image.fromarray(images[img_idx])
            #im.save(imagePath+"/Input_"+str(img_idx)+".jpg")



        out.release()
        command = [
            "ffmpeg",
            "-i", video_path,
            "-c:v", "h264",  # Or "openh264" if needed
            video_path_dest
        ]

        # Run the command
        try:
            subprocess.run(command, check=True)
            print(f"Video saved to {video_path_dest}")
        except subprocess.CalledProcessError as e:
            print("Error during conversion:", e)
        os.remove(video_path)
        print(f"Video saved to {video_path}")
            #for img_idx in range(len(images)):
            #    print(len(images[img_idx]))
            #    im = Image.fromarray(images[img_idx])
            #    im.save(result_dir + f"/{idx}/" + f'{img_idx}.j
