import matplotlib.pyplot as plt
import json, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib
from tqdm import tqdm

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
    # === LOAD HUMAN LABELS ===
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
            file_name=file_name+'.json'
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

def plot_violin_matrix(all_metrics_data, stat, title=None):
    
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
    quality_acronyms ={

        'High Quality':"High",
        'Medium Quality':"Medium",
        'Low Quality':"Low",
        'Failing':"Fail"
    }

    sns.set_style("whitegrid", {"grid.color": "black"}) 
    quality_groups = ['High Quality', 'Medium Quality', 'Low Quality', 'Failing']
    metric_names = list(all_metrics_data.keys())
    n_rows = len(metric_names)
    n_cols = len(quality_groups)

    boxplot_width = 3.5
    boxplot_height = 2
    fig_width = n_cols * boxplot_width
    fig_height = n_rows * boxplot_height

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharey='row')

    # Handle cases with 1 row or 1 column
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    colors = {
        "High Quality": "#8fd694",
        "Medium Quality": "#fce98e",
        "Low Quality": "#f8b774",
        "Failing": "#f27d7d",
    }

    for i, metric in enumerate(metric_names):
        row_vals = []
        for quality in quality_groups:
            row_vals.extend(all_metrics_data[metric][quality][stat])
        y_min, y_max = min(row_vals), max(row_vals)
        y_margin = (y_max - y_min) * 0.25 if y_max != y_min else 1e-2
        y_lim = (y_min - y_margin, y_max + y_margin)
        y_ticks = [y_lim[0] if y_lim[0] >= 0 else 0, (y_lim[0] + y_lim[1]) / 2, y_lim[1]]

        for j, quality in enumerate(quality_groups):
            ax = axes[i][j]
            ax.patch.set_edgecolor('black')
            ax.patch.set_linewidth(1.5)
            values = all_metrics_data[metric][quality][stat]

            sns.violinplot(
                y=values,
                ax=ax,
                inner=None,
                color=colors[quality],
                linewidth=1.5,
                edgecolor='black'
            )
            median = np.median(values)
            ax.plot(0, median, marker='o', color='black', markersize=6, zorder=3)
            ax.set_ylim(y_lim)
            ax.set_xticks([])
            ax.set_yticks(y_ticks)
            ax.tick_params(axis='y', labelsize=27)
            formatter = mticker.ScalarFormatter(useMathText=False)
            formatter.set_scientific(False)
            formatter.set_powerlimits((-3, 4))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

            if i == 0:
                ax.set_title(quality_acronyms.get(quality), fontsize=32, fontweight='bold')

            if j == n_cols - 1:
                ax.text(1.05, 0.5,
                        f"{metric_acronyms.get(metric)}",
                        va='center', ha='center',
                        fontsize=32, fontweight='bold', rotation=-90,
                        transform=ax.transAxes)
            
                if i == 4:
                    fig.text(0.95, (n_rows - i )/n_rows, "Uncertainty Metrics", 
                            va='center', ha='center', fontsize=32, fontweight='bold', rotation=-90)

                if i == len(metric_names) - 3:
                    fig.text(0.95, (n_rows - i - 0.5)/n_rows, "Quality Metrics", 
                            va='center', ha='left', fontsize=32, fontweight='bold', rotation=-90)
                
            elif j == 0:
                ax.yaxis.tick_left()
            else:
                ax.set_ylabel('')
                
    try:
        ev_index = metric_names.index('Execution_Variability')
        total_rows = len(metric_names)
        # Calculate vertical position in figure coordinates (1 at top, 0 at bottom)
        # Center between EV row and the next row
        y_pos = 1 - ((ev_index + 1) / total_rows)

        fig_line = matplotlib.lines.Line2D(
            [0.00, 1],  # From left to right across the figure
            [y_pos, y_pos],  # Horizontal line at this vertical position
            transform=fig.transFigure,
            color='black',
            linewidth=4,
            linestyle='--'
        )
        fig.add_artist(fig_line)
    except ValueError:
        print("⚠️ Execution_Variability not found in metric list.")
    plt.tight_layout(pad=1.0, w_pad=0.25, h_pad=0.25)
    plt.subplots_adjust(left=0.1, right=0.9)

    if title:
        fig.suptitle(title, fontsize=25, fontweight='bold', y=1.02)

    return fig

def plot_combined_model(model_name, model_task_data, stat, output_dir):
    task_order = ['t-grasp_n-1000_o-m3_s-2498586606', 't-move_n-1000_o-m3_s-2263834374', 't-put-in_n-1000_o-m3_s-2905191776', 't-put-on_n-1000_o-m3_s-2593734741']
    sub_figs = []
    categories = ['Pick Up', 'Move Near' ,'Put in', 'Put on']
    for task in task_order:
        if task in model_task_data:
            fig = plot_violin_matrix(model_task_data[task], stat, title=task.upper())
            sub_figs.append(fig)

    # Extract axes as images to stitch horizontally
    fig_width = sum(fig.get_size_inches()[0] for fig in sub_figs)
    fig_height = max(fig.get_size_inches()[1] for fig in sub_figs)
    
    combo_fig, combo_ax = plt.subplots(figsize=(fig_width, fig_height))
    combo_ax.axis('off')

    # Combine saved images as subplots
    matplotlib.use('Agg') 
    images = []
    for fig in sub_figs:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        img = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img = img.reshape((h, w, 4))
        img_rgb = img[..., :3]
        images.append(img_rgb)
        plt.close(fig)

    full_img = np.concatenate(images, axis=1)  # Horizontal concat

    combo_fig, combo_ax = plt.subplots(figsize=(sum(fig.get_size_inches()[0] for fig in sub_figs),
                                            max(fig.get_size_inches()[1] for fig in sub_figs)))
    combo_ax.axis('off')
    combo_ax.imshow(full_img)

    # Add category titles centered above each image
    cumulative_widths = np.cumsum([img.shape[1] for img in images])  # widths in pixels
    prev = 0
    for idx, category in enumerate(categories):
        center_x = prev + images[idx].shape[1] // 2
        prev = cumulative_widths[idx]

        # Use figure coordinates for text:
        # transform=ax.transData means coordinates in pixels (image pixels)
        combo_ax.text(center_x, -30,  # negative y to go slightly above image
                    category.replace('_', ' ').title(),
                    ha='center', va='bottom',
                    fontsize=40, fontweight='bold',
                    color='black',
                    transform=combo_ax.transData)

    plt.subplots_adjust(top=0.9)  # adjust top margin for titles if needed
    output_path = os.path.join(output_dir, f"{model_name}_combined_violin_matrix.pdf")
    combo_fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(combo_fig)
    print(f"✅ Saved combined figure for {model_name}: {output_path}")

    """
    combo_ax.imshow(full_img)
    output_path = os.path.join(output_dir, f"{model_name}_combined_violin_matrix.png")
    combo_fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(combo_fig)
    print(f"✅ Saved combined figure for {model_name}: {output_path}")
    """




excluded_files = {'log.json', 'actions.json', 'tcp_poses.json', 'exec_time_execution_variability.json', 'exec_time_inference.json', 'exec_time_instability.json', 'exec_time_optimal_trajectory.json', 'exec_time_token-based.json', 'exec_time_trajectory_instability_gradients.json'}
excluded_tasks = []     # if any
tasks = ['t-grasp_n-1000_o-m3_s-2498586606', 't-move_n-1000_o-m3_s-2263834374', 't-put-in_n-1000_o-m3_s-2905191776', 't-put-on_n-1000_o-m3_s-2593734741']
models = ['openvla-7b', 'pi0', 'spatialvla-4b']
output_dir = 'figures/combined_tables'
os.makedirs(output_dir, exist_ok=True)

for model in models:
    model_data = {}

    for task in tasks:
        base_task=task.split('_')[0]
        base_task=base_task[2:]
        base_path = f"../results/{task}/{model}/allMetrics"
        eval_file = f"../results/human_eval/final_evaluations_{model}_{base_task}.xlsx"

        if not os.path.exists(base_path) or not os.path.exists(eval_file):
            print(f"⚠️ Skipping {model}/{task} (missing data)")
            continue

        all_metrics = collect_metrics(base_path, eval_file, excluded_files, excluded_tasks)
        model_data[task] = all_metrics

    plot_combined_model(model, model_data, stat='avg', output_dir=output_dir)
