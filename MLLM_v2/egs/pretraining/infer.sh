# inference
. ./path.sh
ngpu=1
inference_dir='/home-dongchao/exp/MLLM/exp/infer'
part='simple_infer'


python3 ../../infer_no_streaming.py \
    --exp_dir /home-dongchao/exp/MLLM/exp/exp/audiollm_v2_llama3B_11_25_tts \
    --resume /home-dongchao/exp/MLLM/exp/exp/audiollm_v2_llama3B_11_25_tts/ep1-iter125000.checkpoint \
    --inference_mode 'sampling' \
    --rank 0 \
    --output_dir /home-dongchao/code3/RSTnet_private/MLLM2_11_24/egs/pretraining/tts_only_11_25 \
    --data_json /home-dongchao/exp/MLLM/tasks/audio/libritts/test/8splits/data_tts.0.json \
    --generate_target 'audio' \
    --task_name 'TTS'
