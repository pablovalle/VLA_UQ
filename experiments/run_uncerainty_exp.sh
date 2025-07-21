#!/bin/zsh

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                  UQ4VLAs v1.0                    ┃
# ┃  Run uncertainty and quality experiments across  ┃
# ┃          models and datasets in style!           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' # No color

# Usage message
if [[ $# -lt 1 ]]; then
  echo -e "${RED}✘ Error: Missing Conda environment argument.${NC}"
  echo -e "Usage: ${YELLOW}./eun_UQ_exp.sh <conda_env_name> [optional_model]${NC}"
  exit 1
fi

conda_env=$1
specific_model=$2

# Conda init (adjust this if needed)
source ~/anaconda3/etc/profile.d/conda.sh

# Activate conda environment
echo -e "${BLUE}➤ Activating Conda environment: ${GREEN}${conda_env}${NC}"
conda activate "$conda_env" || {
  echo -e "${RED}✘ Failed to activate conda environment '${conda_env}'. Exiting.${NC}"
  exit 1
}

# Define model list
models=(openvla-7b spatialvla-4b pi0)
if [[ -n "$specific_model" ]]; then
  echo -e "${BLUE}➤ Running only for model: ${YELLOW}${specific_model}${NC}"
  models=("$specific_model")
fi

# Define datasets
datasets=(
  t-grasp_n-1000_o-m3_s-2498586606.json
  t-move_n-1000_o-m3_s-2263834374.json
  t-put-in_n-1000_o-m3_s-2905191776.json
  t-put-on_n-1000_o-m3_s-2593734741.json
)

# Start banner
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "      🚀 Launching Uncertainty4VLAs Engine"
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Main experiment loop
for data in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    echo -e "\n${YELLOW}▶ Running model: ${model} | Dataset: ${data}${NC}"
    python3.10 run_fuzzer_allMetrics.py -m "${model}" -d "../data/${data}"

    if [[ $? -ne 0 ]]; then
      echo -e "${RED}✘ Failed: ${model} on ${data}${NC}"
    else
      echo -e "${GREEN}✔ Completed: ${model} on ${data}${NC}"
    fi
  done
done

# Done!
echo -e "\n${GREEN}🎉 All experiments completed successfully.${NC}"
