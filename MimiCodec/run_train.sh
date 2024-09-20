
# export CUDA_VISIBLE_DEVICES=0,1
export HOST_GPU_NUM=8
export HOST_NUM=1
export NODE_NUM=1
export INDEX=0
export CHIEF_IP="localhost"
port=12353
NCCL_DEBUG=TRACE torchrun   --nproc_per_node ${HOST_GPU_NUM} --master_port $port \
        --nnodes=${HOST_NUM} --node_rank=${INDEX} --master_addr=${CHIEF_IP} \
        train.py \
        --log_dir log --basic_model_config "config/mimi24k.yaml" \
        --semantic_feature_type wavlm 

