#!/bin/bash
#SBATCH --job-name=gen500
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1024G
#SBATCH --partition=AISS2024110401
#SBATCH --time=222:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=88
#SBATCH --output=/lustre/projects/polyullm/yggu/gen-example/gen-%j.out
#SBATCH --error=/lustre/projects/polyullm/yggu/gen-example/gen-%j.err
# set -x

# replace these information with your own
gen_workdir=/lustre/projects/polyullm/yggu/gen-example
container_image=/lustre/projects/polyullm/container/easy_r1+dev+0303.sqsh
container_name=easy_r1_0305
container_mounts=/lustre/projects/polyullm:/lustre/projects/polyullm,/home/projects/polyullm:/home/projects/polyullm
# replace these information with your own

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)


# Get the IP address of the head node
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)


# Start Ray head node
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"


# make sure we set environment variables before Ray initialization
# export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=offline
export HOME=$gen_workdir
export PYTHONPATH=$PYTHONPATH:/opt/nvidia/Megatron-LM

printenv

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    --container-name=$container_name \
    --container-mounts=$container_mounts \
    --container-image=$container_image \
    --container-workdir=$gen_workdir \
    --container-writable \
    bash -c "ray start --head --node-ip-address=$head_node_ip --port=$port --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block" &

sleep 10


# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        --container-name=$container_name \
        --container-mounts=$container_mounts \
        --container-image=$container_image \
        --container-workdir=$gen_workdir \
        --container-writable \
        bash -c "ray start --address $ip_head --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block" &
    sleep 10
done

echo "Waiting for 120 seconds..."
sleep 120
echo "Starting training..."



# ######################################################### config ##############################################################
SCRIPTS="python gen.py $@"
# ###############################################################################################################################

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    --container-name=$container_name \
    --container-mounts=$container_mounts \
    --container-image=$container_image \
    --container-workdir=$gen_workdir \
    --container-writable \
    bash -c "$SCRIPTS"

# Clean up Ray processes
cleanup() {
    echo "Shutting down Ray cluster..."
    srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
        --container-name=$container_name \
        --container-mounts=$container_mounts \
        --container-image=$container_image \
        --container-writable \
        bash -c "ray stop"
    
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        srun --overlap --nodes=1 --ntasks=1 -w "$node_i" \
            --container-name=$container_name \
            --container-mounts=$container_mounts \
            --container-image=$container_image \
            --container-writable \
            bash -c "ray stop"
    done
}

# Set up trap to call cleanup function on script exit
trap cleanup EXIT
