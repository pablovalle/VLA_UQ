import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

results_root = "results"
excluded_files = {
    'log.json', 'actions.json', 'tcp_poses.json',
    'exec_time_execution_variability.json', 'exec_time_inference.json',
    'exec_time_instability.json', 'exec_time_optimal_trajectory.json',
    'exec_time_token-based.json', 'exec_time_trajectory_instability_gradients.json'
}
excluded_tasks = ['475', '459', '46', '435', '39', '353', '33', '249', '203', '168', '104']

all_metrics = set()
correlation_records = []
ordered_metrics = [
        'Token_based_token_prob',
        'Token_based_pcs',
        'Token_based_deepgini',
        'Token_based_entropy',
        
        'position_instability',
        'velocity_instability',
        'acceleration_instability',

        'TCP_position_instability',
        'TCP_velocity_instability',
        'TCP_acceleration_instability',

        'Trajectory_Instability_Gradients',
        'Optimal_Trajectory',
        'Execution_Variability'
    ]
metric_acronyms = {
        'Token_based_token_prob':"TB-TP",
        'Token_based_pcs':"TB-PCS",
        'Token_based_deepgini':"TB-D",
        'Token_based_entropy':"TB-E",
        
        'position_instability':"A-PI",
        'velocity_instability':"A-VI",
        'acceleration_instability':"A-AI",

        'TCP_position_instability':"TCP-PI",
        'TCP_velocity_instability':"TCP-VI",
        'TCP_acceleration_instability':"TCP-AI",

        'Trajectory_Instability_Gradients':"TI",
        'Optimal_Trajectory':"OT",
        'Execution_Variability':"EV"
    }
# Loop over task types
for task_type in sorted(os.listdir(results_root)):
    task_path = os.path.join(results_root, task_type)
    if not os.path.isdir(task_path):
        continue

    for model in sorted(os.listdir(task_path)):
        model_path = os.path.join(task_path, model, "allMetrics")
        if not os.path.isdir(model_path):
            continue
        base_task=task_type.split('_')[0]
        base_task=base_task[2:]
        eval_path = f"results/human_eval/final_evaluations_{model}_{base_task}.xlsx"
        if not os.path.exists(eval_path):
            print(f"⚠️ Missing eval file: {eval_path}")
            continue

        # Load human evaluation
        df_quality = pd.read_excel(eval_path)
        df_quality["quality_rank"] = df_quality["final_evaluation"].map({"High Quality": 1, "Medium Quality": 2, "Low Quality": 3})
        df_quality["task_id"] = df_quality["simulation"].apply(lambda x: x.split("/")[-1].split("_")[0])
        quality_map = df_quality.set_index("task_id")["quality_rank"].to_dict()

        metrics = []

        for task_id in os.listdir(model_path):
            task_folder = os.path.join(model_path, task_id)
            if not os.path.isdir(task_folder) or task_id not in quality_map:
                continue
            
            if task_type == 't-put-in_n-1000_o-m3_s-2905191776' and task_id in excluded_tasks:
                continue

            quality_rank = quality_map[task_id]

            for file_name in ordered_metrics:
                file_name=file_name+'.json'
                if not file_name.endswith(".json") or file_name in excluded_files:
                    continue
                try:
                    with open(os.path.join(task_folder, file_name)) as f:
                        metric_value = json.load(f)

                    if isinstance(metric_value, list) and all(isinstance(x, list) for x in metric_value):
                        metric_value = [np.mean(x) for x in metric_value if len(x) > 0]

                    if not isinstance(metric_value, list) or len(metric_value) < 3:
                        continue

                    metrics.append({
                        "task_id": task_id,
                        "quality_rank": quality_rank,
                        "metric_name": file_name,
                        "metric_value": np.mean(metric_value)
                    })

                    all_metrics.add(file_name)

                except Exception as e:
                    print(f"⚠️ Error reading {file_name} for task {task_id}: {e}")
                    continue

        df_metrics = pd.DataFrame(metrics)
        
        if df_metrics.empty:
            continue

        df_pivot = df_metrics.pivot_table(
            index=["task_id", "quality_rank"],
            columns="metric_name",
            values="metric_value"
        ).reset_index()

        #print(df_pivot)

        for metric_name in sorted(all_metrics):
            if metric_name not in df_pivot.columns:
                continue

            x = df_pivot["quality_rank"]
            y = df_pivot[metric_name]
            valid = x.notna() & y.notna()

            if valid.sum() >= 2 and x[valid].nunique() > 1 and y[valid].nunique() > 1:
                rho, pval = spearmanr(x[valid], y[valid])
                
                correlation_records.append({
                    "task_type": task_type,
                    "model": model,
                    "metric": metric_name,
                    "stat": "ρ",
                    "value": round(rho, 3)
                })
                correlation_records.append({
                    "task_type": task_type,
                    "model": model,
                    "metric": metric_name,
                    "stat": "p-value",
                    "value": "<0.0001" if pval < 0.0001 else f"{pval:.4f}"
                })

# Create final DataFrame
df_corr = pd.DataFrame(correlation_records)
print(df_corr)
# Pivot to table: rows = metric + stat, columns = (task_type, model)
df_pivot = df_corr.pivot_table(
    index=["metric", "stat"],
    columns=["task_type", "model"],
    values="value",
    aggfunc="first"
).reset_index()
print(df_pivot)
# Save
df_pivot.to_csv("combined_spearman_table.csv", index=False)
print("✅ Saved combined Spearman correlation table to: combined_spearman_table.csv")
