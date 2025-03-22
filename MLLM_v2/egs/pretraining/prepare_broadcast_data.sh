stage=0
stop_stage=7
ngpu=2

db_root=/mnt/users/hccl.local/jkzhao/data/fisher
processed_metadata_root=/mnt/users/hccl.local/jkzhao/projects/RSTnet/debug_data
processed_audio_root=/mnt/users/hccl.local/jkzhao/projects/RSTnet/debug_data_processed

export CUDA_VISIBLE_DEVICES=3,4,5,6,7
available_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Available GPUs: $available_gpus"

source /mnt/users/hccl.local/jkzhao/softwares/miniconda3/etc/profile.d/conda.sh
conda activate AudioPipeline
export PYTHONPATH=$PYTHONPATH:/mnt/users/hccl.local/jkzhao/projects/RSTnet/MLLM_v2

mkdir -p $processed_metadata_root
wav_scp=$processed_metadata_root/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp

# Prepare wav.scp
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Prepare Fisher dataset"
    find "${db_root}" -type f -name "*.wav" | while read -r wav_file; do
        id=$(basename $wav_file .wav)
        echo "$id $wav_file" >> $wav_scp
    done
    # For debugging, only use the first 3 lines
    mv $wav_scp ${wav_scp}.tmp
    head -n 3 ${wav_scp}.tmp > $wav_scp
    rm ${wav_scp}.tmp
fi

# Split the $processed_metadata_root for $ngpu GPUs
# This is done before $processed_metadata_root preprocessing such that multiple GPUs can be used for $processed_metadata_root preprocessing
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "split the $processed_metadata_root for $ngpu GPUs"
    mkdir -p $processed_metadata_root/${ngpu}splits
    # extra shuf to ensure balance across GPUs
    # So the generated $processed_metadata_root cannot be reproduced due to the shuffle randomness
    if [ -f $processed_metadata_root/wav.scp.shuf ]; then
        rm -f $processed_metadata_root/wav.scp.shuf
    fi
    
    cat $processed_metadata_root/wav.scp | shuf >  $processed_metadata_root/wav.scp.shuf
    split_scp=
    for n in `seq 1 $ngpu`; do
        split_scp="$split_scp $processed_metadata_root/${ngpu}splits/wav.${n}.scp"
    done
    utils/split_scp.pl $processed_metadata_root/wav.scp.shuf $split_scp
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Emilia pipeline: Spk -> VAD -> ASR"
    utils/run.pl JOB=1:$ngpu  $processed_metadata_root/${ngpu}splits/log/emilia.JOB.log \
    python data_scripts/emilia/main.py \
        --rank JOB \
        --input_scp $processed_metadata_root/${ngpu}splits/wav.JOB.scp \
        --input_folder_path $db_root \
        --output_scp $processed_metadata_root/${ngpu}splits/wav_seg.JOB.scp \
        --output_utt2json $processed_metadata_root/${ngpu}splits/utt2json.JOB \
        --output_folder_path $processed_audio_root \
        --max_duration 60 \
        --config_path data_scripts/emilia/config.json
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Audio Tokenization"

    conda activate open-moshi
    utils/run.pl JOB=1:$ngpu $processed_metadata_root/${ngpu}splits/log/mimi.JOB.log \
        python3 local/offline_codec_tokenization.py \
            --input-file  $processed_metadata_root/${ngpu}splits/wav_seg.JOB.scp \
            --output-file  $processed_metadata_root/${ngpu}splits/audio_codec.JOB.pt \
            --tokenizer mimi --rank JOB || exit 1;
    conda activate AudioPipeline
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Prepare text sequence"
    utils/run.pl JOB=1:$ngpu  $processed_metadata_root/${ngpu}splits/log/text_bpe.JOB.log \
    python  data_scripts/text_tokenization_utt2json.py \
        --rank JOB \
        --input-file  $processed_metadata_root/${ngpu}splits/utt2json.JOB \
        --checkpoint_dir /mnt/users/hccl.local/jkzhao/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062 \
        --output-file $processed_metadata_root/${ngpu}splits/text.JOB.pt
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "create $processed_metadata_root json"
    #mkdir -p $processed_metadata_root/${ngpu}splits
    for n in `seq 0 $[$ngpu-1]`; do
    python3 data_scripts/create_data_json.py \
        --task setence_level_text_audio_interleaved \
        --out-json $processed_metadata_root/${ngpu}splits/broadcast_data.${n}.json \
        --text_seq $processed_metadata_root/${ngpu}splits/text.$[$n+1].pt \
        --audio_seq $processed_metadata_root/${ngpu}splits/audio_codec.$[$n+1].pt \
        & 
    done; wait
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "Dataloader test"
    conda activate open-moshi
    python3 ../../utils/dataloader.py \
        --train_data_jsons $processed_metadata_root/${ngpu}splits/broadcast_data.ALL.json \
        --valid_data_jsons $processed_metadata_root/${ngpu}splits/broadcast_data.ALL.json \
        --checkpoint_path /mnt/users/hccl.local/jkzhao/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062/lit_model.pth
fi
