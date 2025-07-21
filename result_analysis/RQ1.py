import os
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

def generate_latex_summary_table(summary):
    import re

    # Desired task order (normalize)
    task_order = ["grasp", "move", "put-in", "put-on"]
    normalized_tasks = {}
    for task in summary:
        for pattern in task_order:
            if re.search(pattern, task, re.IGNORECASE):
                normalized_tasks[pattern.title()] = task

    # Use normalized task keys sorted by custom order
    tasks = [k.title() for k in task_order if k.title() in normalized_tasks]

    # Use all metrics in desired row order
    metrics = ["Success", "Fail", "High Quality", "Medium Quality", "Low Quality", "False Negative"]

    # Alphabetically ordered models (taken from first task)
    example_task = next(iter(summary.values()))
    models = sorted(example_task.keys())

    # Build LaTeX table
    latex = []
    latex.append(r"\begin{tabular}{l|" + "c" * (len(tasks) * len(models)) + "}")
    latex.append(r"\toprule")

    # First header row (task names)
    header1 = ["Metric"]
    for task in tasks:
        header1.append(r"\multicolumn{" + str(len(models)) + r"}{c}{" + task + "}")
    latex.append(" & ".join(header1) + r" \\")

    # Second header row (model names)
    header2 = [""]
    for _ in tasks:
        header2.extend(models)
    latex.append(" & ".join(header2) + r" \\")
    latex.append(r"\midrule")

    # Fill in each metric row
    for metric in metrics:
        row = [metric]
        for task in tasks:
            task_key = normalized_tasks[task]
            for model in models:
                value = summary[task_key][model][metric]  # already formatted as "42 (84.0%)"
                row.append(str(value))
        latex.append(" & ".join(row) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    return "\n".join(latex)



base_dir = "../results"
human_eval_dir = base_dir+"/human_eval"

TOTAL_EXECUTIONS = 500

summary = defaultdict(lambda: defaultdict(dict))

for task in os.listdir(base_dir):
    task_path = os.path.join(base_dir, task)
    if not os.path.isdir(task_path) or task_path==human_eval_dir:
        continue

    for model in os.listdir(task_path):
        model_path = os.path.join(task_path, model, 'allMetrics')

        success=0
        for scene in tqdm(os.listdir(model_path), desc=f"Calculating for model {model}"):
            log_path = os.path.join(model_path , scene, 'log.json')
    
            with open(log_path, 'r') as f:
                log_data = json.load(f)
                last_timestep_key = str(max(map(int, log_data.keys())))
                success_flag = log_data[last_timestep_key]["success"]
                label = 0 if success_flag=='false' or success_flag==False else 1

                success=success+label
        fail=500-success

        print(fail)
        print(success)
        eval_file = f"../results/human_eval/final_evaluations_{model}_{task.split('_')[0][2:]}.xlsx"
        eval_df = pd.read_excel(eval_file)
        counts = eval_df['final_evaluation'].value_counts()
        print(counts['High Quality'])
        
        False_negative=success - len(eval_df)
        success_false=success-False_negative
        total_execs_nofalse=TOTAL_EXECUTIONS-False_negative

        summary[task][model] = {
            "total": TOTAL_EXECUTIONS,
            "Success": f"{success_false} ({success_false / TOTAL_EXECUTIONS:.1%})",
            "Fail": f"{fail} ({fail / TOTAL_EXECUTIONS:.1%})",
            "High Quality": f"{counts['High Quality']} ({counts['High Quality'] / success_false:.1%})" if success_false else "0 (0.0%)",
            "Medium Quality": f"{counts['Medium Quality']} ({counts['Medium Quality'] / success_false:.1%})" if success_false else "0 (0.0%)",
            "Low Quality": f"{counts['Low Quality']} ({counts['Low Quality'] / success_false:.1%})" if success_false else "0 (0.0%)",
            "False Negative": f"{False_negative} ({(False_negative) / TOTAL_EXECUTIONS:.1%})" if TOTAL_EXECUTIONS else "0 (0.0%)"
        }
        
latex_code = generate_latex_summary_table(summary)
with open("RQ1_table.tex", "w") as f:
    f.write(latex_code)
# Print summary
import pprint
pprint.pprint(dict(summary))
