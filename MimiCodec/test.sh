
run_dir="exp_data/Moshi/MimiCodec/v1/log/model_ckpts"

ckpt="$(ls -dt "${run_dir}"/*.pth | head -1 || true)"
ckpt_name=$( basename ${ckpt} )

outdir=$( dirname $( dirname ${ckpt} ))/output_${ckpt_name%.*}_bw${bw}kpbs_$(date '+%Y-%m-%d-%H-%M-%S')
echo Output in $outdir

wav_dir="exp/data/source"
exp_model_config="Open-Moshi/MimiCodec/config/mimi24k.yaml"

python inference.py --input ${wav_dir} --output ${outdir} --exp_model_config ${exp_model_config} \
       --resume_path model_ckpts/ckpt_00115000.pth
