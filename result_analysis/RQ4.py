import os
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

base_dir = "../results"
human_eval_dir = base_dir+"/human_eval"

TOTAL_EXECUTIONS = 500

summary = defaultdict(lambda: defaultdict(dict))

exec_time_metrics=["exec_time_execution_variability", "exec_time_inference", "exec_time_token-based", "exec_time_instability", "exec_time_optimal_trajectory", "exec_time_trajectory_instability_gradients"]

models=["openvla-7b",'pi0', 'spatialvla-4b']
data = []
for model in models:
    
    for task in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task)
        if not os.path.isdir(task_path) or task_path==human_eval_dir:
            continue

        
        model_path = os.path.join(task_path, model, 'allMetrics')

        success=0
        for scene in tqdm(os.listdir(model_path), desc=f"Calculating for model {model}"):
            
            for metric in exec_time_metrics:
                file_path = os.path.join(model_path , scene, metric+'.json')
                with open(file_path, 'r') as f:
                    log_data = json.load(f)
                    avg_time = np.mean(log_data)
                    if metric == "exec_time_instability":
                        # Split into action and TCP versions
                        half_time = avg_time / 2.0
                        data.append({
                            "Task": task,
                            "Model": model,
                            "Scene": scene,
                            "Metric": "Instability_Action",
                            "ExecTime": half_time
                        })
                        data.append({
                            "Task": task,
                            "Model": model,
                            "Scene": scene,
                            "Metric": "Instability_TCP",
                            "ExecTime": half_time
                        })
                    else:
                        data.append({
                            "Task": task,
                            "Model": model,
                            "Scene": scene,
                            "Metric": metric.replace("exec_time_", "").replace("-", "_").title(),
                            "ExecTime": avg_time
                        })
df = pd.DataFrame(data)
metric_name_map = {
    "Execution_Variability": "Execution Variability",
    "Instability_Action": "Action Instability",
    "Instability_TCP": "TCP Instability",
    "Optimal_Trajectory": "Optimal Trajectory",
    "Token_Based": "Token Based",
    "Trajectory_Instability_Gradients": "Trajectory Instability",
}

# === Map them in the DataFrame ===
    # === Map metric names ===
# Compute mean and std
summary_stats = df.groupby(['Metric', 'Model'])['ExecTime'].agg(['mean', 'std']).reset_index()



# Pivot the table so models become columns and metrics are rows
mean_table = summary_stats.pivot(index='Metric', columns='Model', values='mean')
std_table = summary_stats.pivot(index='Metric', columns='Model', values='std')

# Rename columns to indicate mean and std
mean_table.columns = [f"{col}_mean" for col in mean_table.columns]
std_table.columns = [f"{col}_std" for col in std_table.columns]

# Combine the mean and std tables side-by-side
final_table = pd.concat([mean_table, std_table], axis=1)

# Optional: sort columns for neatness
final_table = final_table[sorted(final_table.columns)]

latex_table = final_table.to_latex(index=True, 
                                   column_format='l' + 'r' * len(final_table.columns),
                                   multicolumn=True,
                                   multirow=True,
                                   caption="Mean and standard deviation of execution times for each model and metric.",
                                   label="tab:exec_time_summary")

# Save to .tex file
with open("exec_time_summary_table.tex", "w") as f:
    f.write(latex_table)

# Optional: Print LaTeX to preview
print(latex_table)