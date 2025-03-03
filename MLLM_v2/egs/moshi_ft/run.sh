stage=1
stop_stage=6
ngpu=4  # how many GPUs, you want to use to train the model

train_set="train"
valid_set="val"
test_sets=""

sr=true
se=true

# Dataset paths
db_root="/path/to/your/dataset"
processed_root="exp_data/processed"
ckpt_root="exp"
aero_root="/path/to/aero"
valid_prop=0.1
tag="fisher_ft"

export CUDA_VISIBLE_DEVICES=0,1
available_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Available GPUs: $available_gpus"

# training config
seed=999

batch_scale=2000
learning_rate=0.0001
tag="test"

# Add Moshi root directory to system PYTHONPATH
moshi_root=$(dirname $(dirname $(dirname $(readlink -f $0))))
mimi_root=$(dirname $moshi_root)/MimiCodec
printf "Moshi root directory: %s\n" "$moshi_root"
printf "Mimi root directory: %s\n" "$mimi_root"
export PYTHONPATH="$moshi_root:$PYTHONPATH"
original_pythonpath=$PYTHONPATH

### stage 1-5: data preparation ###
for part in $test_sets $valid_set $train_set; do
    mkdir -p data/${part}
done

mkdir -p data
# wav_scp=data/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
wav_scp=data/wav.scp
train_scp=data/"$train_set"/wav.scp; [[ -f "$train_scp" ]] && rm $train_scp
val_scp=data/"$valid_set"/wav.scp; [[ -f "$val_scp" ]] && rm $val_scp

# conda activate moshi-data
# Prepare data following Espnet and split
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Prepare Fisher dataset"
    # STEP 0: please download the Fisher English Training Speech Part 1 and Part 2 from LDC

    # STEP 1: Untar dataset
    if [ ! -f ${db_root}/.p1_untar_finished ]; then
        if [ ! -f ${db_root}/fisher_eng_tr_sp_LDC2004S13.zip.001 ]; then
            echo "Please download the Fisher English Training Speech Part 1 from LDC and put it in ${db_root}"
            exit 1
        fi
        cat ${db_root}/fisher_eng_tr_sp_LDC2004S13.zip.* > ${db_root}/fisher_eng_tr_sp_LDC2004S13.zip
        unzip ${db_root}/fisher_eng_tr_sp_LDC2004S13.zip -d ${db_root}
        touch ${db_root}/.p1_untar_finished
    fi

    if [ ! -f ${db_root}/.p2_untar_finished ]; then
        if [ ! -f ${db_root}/fe_03_p2_LDC2005S13.zip.001 ]; then
            echo "Please download the Fisher English Training Speech Part 2 from LDC and put it in ${db_root}"
            exit 1
        fi
        cat ${db_root}/fe_03_p2_LDC2005S13.zip* > ${db_root}/fe_03_p2_LDC2005S13.zip
        unzip ${db_root}/fe_03_p2_LDC2005S13.zip -d ${db_root}
        touch ${db_root}/.p2_untar_finished
    fi

    # STEP 2: Convert sph to wav
    if [ ! -f ${db_root}/.sph2wav_finished ]; then
        find "${db_root}" -type f -name "*.sph" | while read -r sph_file; do
            echo "Processing ${sph_file}"
            wav_file="${sph_file%.sph}.wav"
            # Follow https://github.com/robd003/sph2pipe to install sph2pipe, and add it to PATH
            sph2pipe -f wav "${sph_file}" "${sph_file%.sph}.wav"
        done
        # Remove .sph files
        find "${db_root}" -type f -name "*.sph" -delete
        touch ${db_root}/.sph2wav_finished
    fi

    # STEP 3: Prepare wav.scp
    find "${db_root}" -type f -name "*.wav" | while read -r wav_file; do
        id=$(basename $wav_file .wav)
        echo "$id $wav_file" >> $wav_scp
    done

    # STEP 4: Split validation set
    # Randomly select 10% of the lines from wav_scp for validation set
    total_lines=$(wc -l < "$wav_scp")
    val_lines=$(printf "%.0f" "$(echo "$total_lines * $valid_prop" | bc)")  # int($total_lines * $valid_prop)
    shuf -n "$val_lines" "$wav_scp" > "$val_scp"    # Randomly select $val_lines lines
    grep -v -F -f "$val_scp" "$wav_scp" > "$train_scp"    # Remove val_lines from wav_scp to get train_scp
fi

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
    # VAD
    # Please prepare your huggingface token in advance
    echo "Preprocess: VAD with pyannote"
    for part in $test_sets $valid_set $train_set; do
        utils/run.pl JOB=1:$ngpu  data/${part}/${ngpu}splits/log/vad.JOB.log \
        python local/vad_segment.py \
            --rank JOB \
            --input-file data/${part}/${ngpu}splits/wav.JOB.scp \
            --audio-input-dir "${db_root}" \
            --output-file data/${part}/${ngpu}splits/wav_vad.JOB.scp \
            --audio-output-dir "${processed_root}/audio"
    done

    # ASR
    # This will yield wav_asr.JOB.scp and utt2json
    echo "ASR with whisperX"
    for part in $test_sets $valid_set $train_set; do
        utils/run.pl JOB=1:$ngpu  data/${part}/${ngpu}splits/log/asr.JOB.log \
        python local/asr_whisperx.py \
            --rank JOB \
            --input-file data/${part}/${ngpu}splits/wav_vad.JOB.scp \
            --audio-input-dir "${processed_root}"/audio \
            --output-file data/${part}/${ngpu}splits/wav_asr.JOB.scp \
            --metadata-output-dir "${processed_root}/metadata"
    done
    
    # SR
    if [ "${sr}" = true ]; then
        echo "Preprocess: SR with AERO"
        for part in $test_sets $valid_set $train_set; do
            # Extract audio paths from wav_seg.JOB.scp to wav_seg.JOB.lst
            for n in `seq 1 $ngpu`; do
                if [ ! -f data/${part}/${ngpu}splits/wav_asr.${n}.lst ]; then
                    while IFS= read -r line; do
                        audio_path=$(echo "$line" | awk '{print $NF}')
                        echo $audio_path >> \
                            data/${part}/${ngpu}splits/wav_asr.${n}.lst
                    done < data/${part}/${ngpu}splits/wav_asr.${n}.scp
                fi
            done

            utils/run.pl JOB=1:$ngpu  data/${part}/${ngpu}splits/log/sr.JOB.log \
            python ${aero_root}/batch_inference.py \
                dset=debug \
                experiment=aero_8-24_512_64 \
                +rank=JOB \
                +il="${moshi_root}/llm_egs/ft/data/${part}/${ngpu}splits/wav_asr.JOB.lst" \
                +audio_data_dir="${processed_root}/audio" \
                +output="${processed_root}/sr_audio"
        done
    fi

    # SE
    # deepFilter does not support batch processing
    if [ "${se}" = true ]; then
        echo "Preprocess: SE with DeepFilter"
        dir_list=$(find "${processed_root}/sr_audio" -type f -name "*.wav" -exec dirname {} \; | sort -u)
        for dir in ${dir_list}; do
            # Filter out the files that have been processed
            file_list=()
            for file in $(find "${dir}" -type f -name "*.wav"); do
                rel_path=$(realpath --relative-to="${processed_root}/sr_audio" "${file}")
                se_file="${processed_root}/se_audio/${rel_path}"
                # Filter out the files that have been processed
                if [ ! -f "${se_file}" ]; then
                    file_list+=("${file}")
                fi
            done

            # dertemine output path
            rel_path=$(realpath --relative-to="${processed_root}/sr_audio" "${dir}")
            se_dir="${processed_root}/se_audio/${rel_path}"
            mkdir -p "${se_dir}"

            deepFilter "${file_list[@]}" -o ${se_dir} --no-suffix
        done
    fi
fi

# conda activate open-moshi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Prepare text and audio sequence"
    mkdir -p ${processed_root}/codecs
    export PYTHONPATH="$mimi_root:$PYTHONPATH"
    for part in $valid_set $train_set; do
        utils/run.pl JOB=1:$ngpu  data/${part}/${ngpu}splits/log/tokenize_dump.JOB.log \
        python3 data_scripts/offline_tokenization.py \
            --input-audio-file data/${part}/${ngpu}splits/wav_vad.JOB.scp \
            --input-text-file data/${part}/${ngpu}splits/utt2json.JOB \
            --output-file ${processed_root}/codecs/${part}/${ngpu}splits/codec.JOB.pt \
            --root-dir $processed_root \
            --rank JOB || exit 1;
    done
    export PYTHONPATH=$original_pythonpath
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "create data json"
    for part in $train_set $valid_set; do
      mkdir -p data/${part}/${ngpu}splits
      for n in `seq 0 $[$ngpu-1]`; do
        python3 data_scripts/create_data_json.py \
         --task moshi_ft \
         --out-json   data/${part}/${ngpu}splits/data.${n}.json \
         --audio_seq  ${processed_root}/codecs/${part}/${ngpu}splits/codec.$[$n+1].pt \
         & 
      done; wait
    done
fi

### Stage 6: Training ###

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    if [ -z $tag ]; then
        echo "please provide a tag for this experiment" && exit 1;
    fi
    echo "stage 6: training..."
    export HOST_GPU_NUM=4 # set the number of GPU to use
    export HOST_NUM=1
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="localhost"
    export CUDA_LAUNCH_BLOCKING=1
    train_data_jsons="data/train/4splits/data.ALL.json"
    valid_data_jsons="data/val/4splits/data.ALL.json"

    NCCL_DEBUG=TRACE torchrun --nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=$CHIEF_IP --master_port=20004  \
            --node_rank=$INDEX ../../trainer/finetuning_full_fsdp.py \
            --train_data_jsons $train_data_jsons \
            --valid_data_jsons $valid_data_jsons \
            --exp_dir exp_data/Moshi/v3_full_emo_v0 \
            --n_epoch 2  \
            --max_length 1500  \
            --batch_scale 2000 \
            --global_learning_rate 2e-6 \
    
fi
