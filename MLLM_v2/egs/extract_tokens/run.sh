# extract tokens from wav.scp
stage=3
stop_stage=3
ngpu=32  # how many GPUs, you want to use to train the model

train_set="train"
valid_set="val"
test_sets=""

sr=true
se=true

# Dataset paths
db_root="/path/to/your/dataset"
processed_root="/weka2/home-dongchao/data/exp_data/simple_val"
ckpt_root="/weka2/home-dongchao/data/exp_data/simple_val"
valid_prop=0.1
tag="pre-training"

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

# conda activate moshi-data
# Split the data for $ngpu GPUs
# This is done before data preprocessing such that multiple processing can be used for data preprocessing
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "split the data for $ngpu GPUs"
    for part in  $train_set; do
        mkdir -p data/${part}/${ngpu}splits
        # extra shuf to ensure balance across GPUs
        # So the generated data cannot be reproduced due to the shuffle randomness
        cat data/${part}/tar.scp | shuf >  data/${part}/tar.scp.shuf
        split_scp=
        for n in `seq 1 $ngpu`; do
            split_scp="$split_scp data/${part}/${ngpu}splits/tar.${n}.scp"
        done
        utils/split_scp.pl data/${part}/tar.scp.shuf $split_scp
    done
fi

# Data Preprocessing
# We use relative path in wav.scp
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    # ASR
    # This will yield wav_asr.JOB.scp and utt2json
    source /weka2/home-dongchao/env/moshi/bin/activate
    echo "ASR with whisperX"
    for part in $train_set; do
        utils/run.pl JOB=1:$ngpu  data/${part}/${ngpu}splits/log/asr.JOB.log \
        python local/asr_whisperx_tar.py \
            --rank JOB \
            --input-file data/${part}/${ngpu}splits/tar.JOB.scp \
            --output-file data/${part}/${ngpu}splits/wav_asr.JOB.scp \
            --alignment_file data/${part}/${ngpu}splits/word_alignment.JOB.pt
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

    # samping text
    # This will yield wav_asr.JOB.scp and utt2json
    source /weka2/home-dongchao/env/RSTnet/bin/activate
    echo "samping text"
    for part in $train_set; do
        utils/run.pl JOB=1:$ngpu  ./log/asr.JOB.log \
        python ../../tools/tokenizer/sampling_text.py \
            --rank JOB \
            --output-file /weka2/home-dongchao/exp/MLLM/text_data/32splits/text.JOB.json
    done
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

