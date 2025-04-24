#!/bin/bash

#SBATCH --job-name=accelerate
#SBATCH --mem=128G
#SBATCH --partition=AISS2024110401
#SBATCH -D .
#SBATCH --output=logs/O-%x.%j
#SBATCH --error=logs/E-%x.%j
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:8                # number of GPUs per node
#SBATCH --cpus-per-task=160         # 请确认节点实际CPU核心数是否支持该配置
#SBATCH --time=11:59:00             # maximum execution time (HH:MM:SS)

######################
### Debugging Setup ###
######################
set -x  # 开启命令回显
echo "Starting job at $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOBID"
######################

######################
### Set enviroment ###
######################
# Work dir
export WORK_DIR=/lustre/projects/polyullm/yggu/FDPO
echo "WORK_DIR set to: $WORK_DIR"

# Conda
export CONDA_ENV=/lustre/projects/polyullm/yggu/Envs/yggu
export CONDA_CMD='eval "$(/lustre/projects/polyullm/anaconda3/bin/conda shell.bash hook)" && conda activate ${CONDA_ENV}'
$CONDA_CMD
echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# Container
export GPUS_PER_NODE=8
export CONTAINER_IMAGE="/cm/shared/containers/ngc/nemo.dev.sqsh"
export CONTAINER_NAME="yggu"
export CONTAINER_MOUNT=/lustre/projects/polyullm:/lustre/projects/polyullm,/home/projects/polyullm:/home/projects/polyullm

# Env to pass in contrainer
export WANDB_MODE=offline
export TRITON_CACHE_DIR=../
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "Head node IP: $head_node_ip"
######################

# 设置accelerate脚本
CONFIG=configs/zero3_config.yaml
SCRIPT_FILE=scripts/fdpo.py
DATASET=/lustre/projects/polyullm/yggu/data-generation/data/datasets/po-60k-pqqg-refp-fuseqqg
MODEL=/lustre/projects/polyullm/models/phi4/phi4

# 设置(F)DPO
USE_FUSION=False
USE_LN=True
FUSION_TYPE=max-min

# 设置超参
MAX_PROMPT_LEN=2048
MAX_RESPONSE_LEN=2048
MAX_LEN=4096
SAVE=outputs/phi4-fuseqqg-wo-sft-v2

# 动态构建路径后缀
SAVE_SUFFIX=""
if [ "$USE_FUSION" = "True" ]; then
    SAVE_SUFFIX+="-fdpo"
    # 当使用融合时添加融合类型
    if [ -n "$FUSION_TYPE" ]; then
        SAVE_SUFFIX+="-$FUSION_TYPE"
    fi
else
    SAVE_SUFFIX+="-dpo"
fi

# 添加长度归一化标识
if [ "$USE_LN" = "True" ]; then
    SAVE_SUFFIX+="-ln"
fi

# 根据USE_LN设置beta值
if [ "$USE_LN" = "True" ]; then
    BETA_VAL=2.5
else
    BETA_VAL=0.1
fi

# 关键文件检查
echo "Checking critical files:"
ls -lh ${WORK_DIR}/${CONFIG} || echo "Missing config file!"
ls -lh ${WORK_DIR}/${SCRIPT_FILE} || echo "Missing script file!"
ls -lh ${DATASET} || echo "Missing dataset!"
ls -lh ${MODEL} || echo "Missing model!"

export LAUNCHER="accelerate launch \
    --config_file ${WORK_DIR}/${CONFIG} \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    "

export SCRIPT="${WORK_DIR}/${SCRIPT_FILE}"
export SCRIPT_ARGS=" \
    --dataset_name "$DATASET" \
    --model_name_or_path "$MODEL" \
    --learning_rate 3.0e-7 \
    --bf16 \
    --num_train_epochs 1 \
    --max_prompt_length $MAX_PROMPT_LEN \
    --max_completion_length $MAX_RESPONSE_LEN \
    --max_length $MAX_LEN \
    --beta $BETA_VAL \
    --fuse $USE_FUSION \
    --length_norm $USE_LN \
    --fusion_type $FUSION_TYPE \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_first_step \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 20 \
    --output_dir "$SAVE$SAVE_SUFFIX" \
    --no_remove_unused_columns
    "

# 打印完整命令用于调试
echo "Full launch command:"
echo "$LAUNCHER $SCRIPT $SCRIPT_ARGS"

# 执行命令时传递所有需要的环境变量
srun --nodes=${SLURM_NNODES} \
    --container-name=${CONTAINER_NAME} \
    --container-mounts=${CONTAINER_MOUNT} \
    --container-image=${CONTAINER_IMAGE} \
    --container-workdir=${WORK_DIR} \
    --container-writable \
    --container-env="WANDB_MODE,TRITON_CACHE_DIR" \
    bash -c "
    echo 'Inside container at $(date)'
    ${CONDA_CMD}
    ${LAUNCHER} ${SCRIPT} ${SCRIPT_ARGS}
    echo 'Job finished at $(date)'
    "

echo "End of job script at $(date)"
