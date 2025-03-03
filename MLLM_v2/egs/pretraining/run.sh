stage=7
stop_stage=7
ngpu=48  # how many GPUs, you want to use to train the model

train_set="train"
valid_set="val"
test_sets=""

sr=true
se=true

# Dataset paths
db_root="/path/to/your/dataset"
processed_root="exp_data/expresso_processed"
ckpt_root="exp_data/expresso_processed"
aero_root="/path/to/aero"
valid_prop=0.1
tag="pre-training"

# training config
seed=999

batch_scale=2000
learning_rate=0.0001
tag="test"

### stage 1-5: data preparation ###
for part in $test_sets $valid_set $train_set; do
    mkdir -p data/${part}
done

mkdir -p data
# wav_scp=data/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
wav_scp=data/wav.scp
train_scp=data/"$train_set"/wav.scp; [[ -f "$train_scp" ]] && rm $train_scp
val_scp=data/"$valid_set"/wav.scp; [[ -f "$val_scp" ]] && rm $val_scp


# Split the data for $ngpu GPUs
# This is done before data preprocessing such that multiple GPUs can be used for data preprocessing
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "split the data for $ngpu GPUs"
    for part in $test_sets $valid_set $train_set; do
        mkdir -p data/${part}/${ngpu}splits
        # extra shuf to ensure balance across GPUs
        # So the generated data cannot be reproduced due to the shuffle randomness
        cat data/${part}/wav.scp | shuf >  data/${part}/wav.scp.shuf
        split_scp=
        for n in `seq 1 $ngpu`; do
            split_scp="$split_scp data/${part}/${ngpu}splits/wav.${n}.scp"
        done
        utils/split_scp.pl data/${part}/wav.scp.shuf $split_scp
    done
fi

# Data Preprocessing
# We use relative path in wav.scp
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # samping text
    # This will yield wav_asr.JOB.scp and utt2json
    source /home-dongchao/env/RSTnet/bin/activate
    echo "samping text"
    for part in $train_set; do
        utils/run.pl JOB=1:$ngpu  ./log/asr.JOB.log \
        python ../../tools/tokenizer/sampling_text.py \
            --rank JOB \
            --output-file /exp/MLLM/text_data2/64split/text.JOB.json
    done
fi


# conda activate open-moshi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Prepare text and audio sequence"
    for part in $train_set; do
        utils/run.pl JOB=1:$ngpu  /home-dongchao/exp/MLLM/MLS/train/48splits/log/text_bpe.JOB.log \
        python  data_scripts/text_tokenization_scp.py \
            --rank JOB \
            --input-file  /home-dongchao/exp/MLLM/MLS/train/48splits/text.JOB.scp \
            --checkpoint_dir /home-dongchao/data/checkpoints/meta-llama/Meta-Llama-3-8B  \
            --output-file /home-dongchao/exp/MLLM/MLS/train/48splits/text.JOB.pt
    done
    
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Prepare text and audio sequence"
    ngpu=48
    for part in $train_set; do
        utils/run.pl JOB=1:$ngpu  ./log/whisperx.JOB.log \
        python  local/asr_whisperx_tar.py \
            --rank JOB \
            --input-file  /home-dongchao/exp/stablevoice/data/train_mimi/48splits/tar.JOB.scp \
            --output-file /home-dongchao/exp/stablevoice/data/train_mimi/48splits/whisperx.JOB.scp  \
            --alignment_file /home-dongchao/exp/stablevoice/data/train_mimi/48splits/text_align.JOB.pt
    done
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "create data json"
    for part in $train_set $valid_set; do
      mkdir -p $processed_root/${part}/${ngpu}splits
      for n in `seq 0 $[$ngpu-1]`; do
        python3 data_scripts/create_data_json.py \
         --task moshi_ft \
         --out-json   $processed_root/${part}/${ngpu}splits/data.${n}.json \
         --audio_seq  ${processed_root}/codecs/${part}/${ngpu}splits/codec.$[$n+1].pt \
         & 
      done; wait
    done
fi

### Stage 6: Training ###

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    if [ -z $tag ]; then
        echo "please provide a tag for this experiment" && exit 1;
    fi
    echo "stage 6: training..."
    #export CUDA_VISIBLE_DEVICES=0,1,2,3, #5,6,7,
    export HOST_GPU_NUM=8 # set the number of GPU to use
    export HOST_NUM=1
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="localhost"
    export CUDA_LAUNCH_BLOCKING=1
    train_data_jsons="/home-dongchao/exp/MLLM/MLS/train/48splits/data.ALL.json"
    valid_data_jsons="/home-dongchao/exp/MLLM/tasks/audio/libritts/test/8splits/data_tts.ALL.json"

    # if you want to use LORA for LLM backbone, please call pre_training_lora.py. Instead, you can call pre_training_full.py
    # note that for fully-pre-training, you can consider to set the lora_r=0
    # set the LLM checkpoint path to  model_config and checkpoint_path
    NCCL_DEBUG=TRACE python3 -m torch.distributed.run --nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=$CHIEF_IP --master_port=2999  \
            --node_rank=$INDEX ../../trainer/pre_training_lora.py \
            --train_data_jsons $train_data_jsons \
            --valid_data_jsons $valid_data_jsons \
            --exp_dir /home-dongchao/exp/MLLM/exp/exp/audiollm_v2_llama3B_11_25_tts \
            --n_epoch 250  \
            --max_length 1000  \
            --batch_scale 2500 \
            --global_learning_rate 1e-3 \
            --local_learning_rate 2e-4 \
            --model_config /home-dongchao/data/checkpoints/meta-llama/Llama-3.2-3B/model_config.yaml \
            --audio_card 2050 \
            --n_q 8 \
            --dep_q 8 \
            --codecformer_heads 16 \
            --codecformer_layers 6 \
            --checkpoint_path /home-dongchao/data/checkpoints/meta-llama/Llama-3.2-3B/lit_model.pth \
            --lora_r 32 \
            --lora_alpha 16 \
            --lora_query true \
            --lora_key true \
            --lora_value true \
            --lora_projection true \
            --lora_mlp true  \
            --lora_head true \
            --save_interval 5000 \
     
fi
