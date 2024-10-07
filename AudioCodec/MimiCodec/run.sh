stage=1
stop_stage=1
ngpu=8  # how many GPUs, you want to use to train the model

train_set="train"
valid_set="val"
test_sets=""
. ./path.sh
# Dataset paths
db_root="speech_data" # the audio file path
processed_root="./processed" # the wav.scp save path
if [ ! -d "$processed_root" ]; then
  echo "no exist $processed_root , we first mkdir..."
  mkdir -p "$processed_root"
  echo "$processed_root has been created"
else
  echo "the path $processed_root existed"
fi
ckpt_root="exp" # the path to save training ckpt
valid_prop=0.001 # the probality rate for split validation set

# Prepare data following Espnet and split
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Prepare training and validation dataset"
    python get_scp.py \
           --input_path $db_root \
           --output_path $processed_root \
           --val_rate $valid_prop 
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "training the codec"
    export HOST_GPU_NUM=8 # the number of GPUs to train the audio codec
    export HOST_NUM=1 
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="localhost"
    port=12353
    NCCL_DEBUG=TRACE torchrun   --nproc_per_node ${HOST_GPU_NUM} --master_port $port \
            --nnodes=${HOST_NUM} --node_rank=${INDEX} --master_addr=${CHIEF_IP} \
            train.py \
            --log_dir $ckpt_root \
            --basic_model_config "config/mimi24k.yaml" \
            --semantic_feature_type wavlm 

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Inference the codec"
    run_dir="$ckpt_root/model_ckpts"

    ckpt_name=$( basename ${ckpt} )
    ckpt_path=$run_dir/ckpt_00115000.pth # you can choose the different ckpt
    outdir=./output # save inference audios
    echo Output in $outdir

    wav_dir="exp/data/source" # test wav path
    exp_model_config="config/mimi24k.yaml" # the model config

    python inference.py --input ${wav_dir} --output ${outdir} --exp_model_config ${exp_model_config} \
        --resume_path $ckpt_path

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "evaluation the codec"
    cd ../../Evaluation/codec  
    bash compute_metrics.sh  # refer the details in evaluation folders
