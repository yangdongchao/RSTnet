log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

db_root="/mnt/Corpus-Upload/fisher/sph_dev"
processed_root="/mnt/users/hccl.local/jkzhao/data/fisher_processed_dev"
stage=0
stop_stage=0
sr=true
se=true

export CUDA_VISIBLE_DEVICES=2

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: download_and_untar"
    # # Please download the Fisher English Training Speech Part 1 and Part 2 from LDC
    # cat ${db_root}/fisher_eng_tr_sp_LDC2004S13.zip* > ${db_root}/fisher_eng_tr_sp_LDC2004S13.zip
    # unzip ${db_root}/fisher_eng_tr_sp_LDC2004S13.zip

    # cat ${db_root}/fe_03_p2_LDC2005S13.zip* > ${db_root}/fe_03_p2_LDC2005S13.zip
    # unzip ${db_root}/fe_03_p2_LDC2005S13.zip

    find "${db_root}" -type f -name "*.sph" | while read -r sph_file; do
        log "Processing ${sph_file}"
        # Convert sph to wav
        # Follow https://github.com/robd003/sph2pipe to install sph2pipe, and add it to PATH
        sph2pipe -f wav "${sph_file}" "${sph_file%.sph}.wav"
        
        rm "${sph_file}"
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    # All the wav files in db_root will be processed
    # VAD & ASR
    python asr_segment.py \
        --input-dir "${db_root}" \
        --audio-output-dir "${processed_root}/audio" \
        --metadata-output-dir "${processed_root}/metadata"
    # SR
    if [ "${sr}" = true ]; then
        temp_dir="${processed_root}/cache"
        mkdir -p "${temp_dir}"
        dir_list=$(find "${processed_root}/audio" -type f -name "*.wav" -exec dirname {} \; | sort -u)
        for dir in ${dir_list}; do
            # dertemine output path
            # sr_dir=$dir
            rel_path=$(realpath --relative-to="${processed_root}/audio" "${dir}")
            sr_dir="${processed_root}/sr_audio/${rel_path}"
            mkdir -p "${sr_dir}"

            find "${dir}" -type f -name "*.wav" > "${dir}/wav.lst"
            audiosr -il "${dir}/wav.lst" -s "${temp_dir}" --suffix "_sr"

            # Move the processed files to the original directory
            find "${temp_dir}" -type f -name "*.wav" | while read -r wav_file; do
                mv "${wav_file}" "${sr_dir}"
            rm -rf "${temp_dir}/*"
            done
        done
        rm -rf "${temp_dir}"
    fi
    # SE
    if [ "${se}" = true ]; then
        dir_list=$(find "${processed_root}/sr_audio" -type f -name "*.wav" -exec dirname {} \; | sort -u)
        for dir in ${dir_list}; do
            # dertemine output path
            # se_dir=$dir
            rel_path=$(realpath --relative-to="${processed_root}/sr_audio" "${dir}")
            se_dir="${processed_root}/se_audio/${rel_path}"
            mkdir -p "${se_dir}"

            deepFilter -i ${dir} -o ${se_dir}

            # Rename the processed files
            find "${se_dir}" -type f -name "*DeepFilterNet3*.wav" | while read -r se_file; do
                new_name=$(echo "${se_file}" | sed 's/DeepFilterNet3/se/')
                mv "${se_file}" "${new_name}"
            done
        done
    fi
    # local/data_prep.sh "${db_root}/LibriTTS/${name}" "data/${name}"
    # utils/fix_data_dir.sh "data/${name}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: utils/combine_data.sh"
    utils/combine_data.sh data/train-960 data/train-clean-100 data/train-clean-360 data/train-other-500
fi

log "Successfully finished. [elapsed=${SECONDS}s]"