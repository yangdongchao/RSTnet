# Below we need splitted:
# tar.scp
  # name WebDataset[{__key__, json{speaker_id, transcript}, wav}]
# or wav.scp
  # name wav_path

# So that we yields:
# 1. tar_info.scp
  # name
# 2. text.scp
  # name text
# 3. utt2spk.scp
  # name spk
# 4. semantic_codec.pt
  # {name: torch.Tensor(int16)[T]}, with SSLTokenizer

# 5. text.pt
  # {name: token_ids[torch.Tensor(int32)]}

# And finally packed into:
# 1. asr_data.json
# Format: setence_level_text_audio_interleaved
# {keys: {test_seq: text.pt, audio_seq: semantic_codec.pt}}

. ./path.sh

stage=2
stop_stage=2
ngpu=8  # how many GPUs, you want to use to train the model

train_set="train"
valid_set="val"
test_sets="test"
# nohup bash run.sh >output 2>&1 &
# training config 
seed=999
debug=false
batch_scale=11000 # the total number of tokens in one batch
learning_rate=0.005 # the learning rate
port=12351
train_opts=
inference_opts=
tag='multi-scaleLM'
inference_tag=default
resume=
data_tag=
TASK='TTS'
data_root=/home-dongchao/exp/MLLM/tasks/audio/libritts
experiment_name='exp_5wh' # set the experiments name

if [ ! -z $resume ]; then
    train_opts="--resume $resume"
    inference_opts="--resume $resume"
fi

if [ $debug == true ]; then
    export HOST_GPU_NUM=1
    export HOST_NUM=1
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="localhost"
    train_opts="$train_opts"

else
    export HOST_GPU_NUM=8
    export HOST_NUM=1
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="localhost"
    train_opts="$train_opts"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "split the data for $ngpu GPUs"

    for part in $test_sets; do
      mkdir -p $data_root/${part}/${ngpu}splits_2
      # extra shuf to ensure balance across GPUs
      # So the generated data cannot be reproduced due to the shuffle randomness
      cat $data_root/${part}/tar.scp | shuf >  $data_root/${part}/tar.scp.shuf
      split_scp=
      for n in `seq 1 $ngpu`; do
          split_scp="$split_scp $data_root/${part}/${ngpu}splits_2/tar.${n}.scp"
      done
      utils/split_scp.pl $data_root/${part}/tar.scp.shuf $split_scp

    done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "extract MLS tokens"
    for part in $train_set; do
    # for part in $valid_set; do
      echo "prepare $part ... "

      # Audio
      utils/run.pl JOB=1:$ngpu $data_root/${part}/${ngpu}splits_2/log/semantic_codec_dump_new.JOB.log \
        python3 local/offline_codec_tokenization.py \
          --tar-file  $data_root/${part}/${ngpu}splits_2/tar.JOB.scp \
          --tar-info  $data_root/${part}/${ngpu}splits_2/tar_info.JOB.scp \
          --output-text $data_root/${part}/${ngpu}splits_2/text.JOB.scp \
          --output-utt2spk $data_root/${part}/${ngpu}splits_2/utt2spk.JOB.scp \
          --output-file  $data_root/${part}/${ngpu}splits_2/semantic_codec.JOB.pt \
          --tokenizer ssl --rank JOB || exit 1;
      
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Prepare text sequence"
    ngpu=16
    for part in $train_set; do
        utils/run.pl JOB=1:$ngpu  /home-dongchao/exp/MLLM/MLS/val/16splits_2/log/text_bpe.JOB.log \
        python  data_scripts/text_tokenization_scp.py \
            --rank JOB \
            --input-file  /home-dongchao/exp/MLLM/MLS/val/16splits_2/text.JOB.scp \
            --checkpoint_dir /home-dongchao/data/checkpoints/meta-llama/Llama-3.2-3B  \
            --output-file /home-dongchao/exp/MLLM/MLS/val/16splits_2/text.JOB.pt
    done
    
fi

# Below for streaming version, not TTS
# 6. text_align.pt
  # {name: padded_text_tokens: [1, T]}
# 7. audio_align.pt
  # {name: audio_tokens: [8, T]}, with MimiCodec

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "extract alignment libriheavy tokens $train_set"
    for part in $train_set; do
      echo "prepare $part ... "
      # Audio
      utils/run.pl JOB=1:$ngpu $data_root/${part}/${ngpu}splits/log/audio_codec_dump_alignement.JOB.log \
        python3 data_scripts/offline_tokenization_tar.py \
          --tar-file  $data_root/${part}/${ngpu}splits/tar.JOB.scp \
          --tar-info  $data_root/${part}/${ngpu}splits/tar_info.JOB.scp \
          --output-text-file $data_root/${part}/${ngpu}splits/text_align.JOB.pt \
          --output-audio-file $data_root/${part}/${ngpu}splits/audio_align.JOB.pt \
          --llm-ckpt-dir /home-dongchao/data/checkpoints/meta-llama/Meta-Llama-3-8B \
          --rank JOB || exit 1;
    done
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "create data json"
    for part in $train_set; do
      #mkdir -p $data_root/${part}/${ngpu}splits
      for n in `seq 0 $[$ngpu-1]`; do
        python3 data_scripts/create_data_json.py \
         --task setence_level_text_audio_interleaved \
         --out-json /home-dongchao/exp/MLLM/MLS/train/48splits_2/asr_data.${n}.json \
         --text_seq /home-dongchao/exp/MLLM/MLS/train/48splits_2/text.$[$n+1].pt \
         --audio_seq /home-dongchao/exp/MLLM/MLS/train/48splits_2/semantic_codec.$[$n+1].pt \
         & 
      done; wait
    done
fi

