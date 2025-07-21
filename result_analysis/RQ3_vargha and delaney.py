import matplotlib.pyplot as plt
import json, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib
from tqdm import tqdm
from scipy.stats import mannwhitneyu

# ---------- NEW: Vargha & Delaney A12 ----------
def vargha_delaney_a12(x, y):
    """
    Computes Vargha and Delaney A12 effect size.
    A12 = P(X > Y) + 0.5 * P(X == Y)
    """
    nx, ny = len(x), len(y)
    ranks = pd.Series(x + y).rank()
    rank_x = sum(ranks[:nx])
    A12 = (rank_x - (nx * (nx + 1)) / 2) / (nx * ny)
    return A12

# ---------- NEW: Statistical Comparison ----------
def compute_statistics(all_metrics_data, stat='avg'):
    comparisons = []
    for metric, group_data in all_metrics_data.items():
        failing_vals = group_data['Failing'][stat]

        for group in ['High Quality', 'Medium Quality', 'Low Quality']:
            successful_vals = group_data[group][stat]

            if len(successful_vals) < 3 or len(failing_vals) < 3:
                p_val = "-"
                a12 = "-"
            else:

                u_stat, p_val = mannwhitneyu(successful_vals, failing_vals, alternative='two-sided')
                a12 = vargha_delaney_a12(successful_vals, failing_vals)

            comparisons.append({
                'metric': metric,
                'group': group,
                'stat': stat,
                'p_value': p_val,
                'a12': a12,
                'n_success': len(successful_vals),
                'n_failing': len(failing_vals)
            })

    return pd.DataFrame(comparisons)

# ---------- ORIGINAL FUNCTION: collect_metrics ----------
def collect_metrics(base_path, evaluation_excel, excluded_files, excluded_tasks):
    ordered_metrics = [
        'Token_based_token_prob',
        'Token_based_pcs',
        'Token_based_deepgini',
        'Token_based_entropy',
        'position_instability',
        'velocity_instability',
        'acceleration_instability',
        'Execution_Variability',
        'TCP_position_instability',
        'TCP_velocity_instability',
        'TCP_acceleration_instability',
        'Trajectory_Instability_Gradients',
        'Optimal_Trajectory'
    ]

    eval_df = pd.read_excel(evaluation_excel)
    eval_df['simulation'] = eval_df['simulation'].astype(str).str.strip()
    eval_df['sim_base'] = eval_df['simulation'].str.replace('_simulation.mp4', '', regex=False)

    quality_map = {}
    for _, row in eval_df.iterrows():
        sim_base = row['sim_base'].split('/')[-1]
        quality = row['final_evaluation']
        quality_map[sim_base] = quality

    folders = [f for f in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, f))]

    all_metrics_data = {}

    for folder in tqdm(folders, desc="Processing folders"):
        if folder in excluded_tasks:
            continue
        folder_path = os.path.join(base_path, folder)
        sim_name = folder.split('_trajectory')[0]

        group = quality_map.get(sim_name, 'Failing')
        if group not in {'High Quality', 'Medium Quality', 'Low Quality', 'Failing'}:
            continue

        for file_name in ordered_metrics:
            file_name = file_name + '.json'
            if not file_name.endswith('.json') or file_name in excluded_files:
                continue

            metric_path = os.path.join(folder_path, file_name)

            try:
                with open(metric_path) as f:
                    metrics = json.load(f)

                if isinstance(metrics, list) and all(isinstance(x, list) for x in metrics):
                    metrics = [np.mean(x) for x in metrics if isinstance(x, list) and len(x) > 0]

                if not isinstance(metrics, list) or len(metrics) < 3:
                    continue

                max_val = max(metrics)
                avg_val = np.mean(metrics)
                sum_val = sum(metrics)

                metric_key = file_name.replace('.json', '')
                if metric_key not in all_metrics_data:
                    all_metrics_data[metric_key] = {
                        'High Quality': {'max': [], 'avg': [], 'sum': []},
                        'Medium Quality': {'max': [], 'avg': [], 'sum': []},
                        'Low Quality': {'max': [], 'avg': [], 'sum': []},
                        'Failing': {'max': [], 'avg': [], 'sum': []}
                    }

                all_metrics_data[metric_key][group]['max'].append(max_val)
                all_metrics_data[metric_key][group]['avg'].append(avg_val)
                all_metrics_data[metric_key][group]['sum'].append(sum_val)

            except Exception as e:
                print(f"⚠️ Error in {metric_path}: {e}")
                continue

    return all_metrics_data


def build_summary_table(all_stats, model_name):
    records = []

    for task, stats_df in all_stats.items():
        task_short = task.split('_')[0][2:]  # 't-grasp...' → 'grasp'
        for _, row in stats_df.iterrows():
            records.append({
                'Metric': row['metric'],
                'Task_Quality': f"{task_short}_{row['group'].replace(' ', '')}",  # e.g., grasp_HighQuality
                'a12': row['a12'],
                'p_value': row['p_value']
            })

    df = pd.DataFrame(records)
    def format_cell(a12, p):
        if str(p) != '-' and p >= 0.05:
            return f"{a12:.2f}"  # No significance → plain
        elif str(p) == '-':
            return p
        d = 2 * abs(a12 - 0.5)

        # Determine effect size magnitude
        if d < 0.147:
            strength = '10'
        elif d < 0.33:
            strength = '30'
        elif d < 0.474:
            strength = '60'
        else:
            strength = '85'

        # Color direction
        if a12 < 0.5:
            color = f"green!{strength}"
        else:
            color = f"red!{strength}"

        return f"\\cellcolor{{{color}}} {a12:.2f}"
    df['cell'] = df.apply(lambda r: format_cell(r['a12'], r['p_value']), axis=1)

    pivot_df = df.pivot(index='Metric', columns='Task_Quality', values='cell')
    quality_order = ['HighQuality', 'MediumQuality', 'LowQuality']

    def sort_key(col_name):
        task, quality = col_name.split('_')
        return (task, quality_order.index(quality))

    pivot_df = pivot_df[sorted(pivot_df.columns, key=sort_key)]
    pivot_df = pivot_df.sort_index()

    # Output LaTeX table
    latex_path = f"{output_dir}/{model_name}_summary_table.tex"
    with open(latex_path, 'w') as f:
        f.write(r"\usepackage[table]{xcolor}" + "\n")
        f.write(pivot_df.to_latex(escape=False, na_rep="", column_format='l' + 'c'*len(pivot_df.columns)))

    print(f"📄 LaTeX summary table saved to: {latex_path}")
    return pivot_df


# ------------------- MAIN SCRIPT -------------------
excluded_files = {
    'log.json', 'actions.json', 'tcp_poses.json',
    'exec_time_execution_variability.json',
    'exec_time_inference.json',
    'exec_time_instability.json',
    'exec_time_optimal_trajectory.json',
    'exec_time_token-based.json',
    'exec_time_trajectory_instability_gradients.json'
}
excluded_tasks = []

tasks = [
    't-grasp_n-1000_o-m3_s-2498586606',
    't-move_n-1000_o-m3_s-2263834374',
    't-put-in_n-1000_o-m3_s-2905191776',
    't-put-on_n-1000_o-m3_s-2593734741'
]

models = ['openvla-7b', 'pi0', 'spatialvla-4b']
output_dir = 'figures/combined_tables'
os.makedirs(output_dir, exist_ok=True)

for model in models:
    model_data = {}
    model_stats = {}

    for task in tasks:
        base_task = task.split('_')[0][2:]  # remove 't-' prefix
        base_path = f"../results/{task}/{model}/allMetrics"
        eval_file = f"../results/human_eval/final_evaluations_{model}_{base_task}.xlsx"

        if not os.path.exists(base_path) or not os.path.exists(eval_file):
            print(f"⚠️ Skipping {model}/{task} (missing data)")
            continue

        all_metrics = collect_metrics(base_path, eval_file, excluded_files, excluded_tasks)
        model_data[task] = all_metrics

        stats_df = compute_statistics(all_metrics, stat='avg')
        stats_path = os.path.join(output_dir, f"{model}_{base_task}_statistics.csv")
        #stats_df.to_csv(stats_path, index=False)
        model_stats[task] = stats_df  # <-- add this line
        #print(f"✅ Stats saved: {stats_path}")

    # 🔄 Build combined table
    build_summary_table(model_stats, model)
