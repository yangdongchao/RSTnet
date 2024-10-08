import argparse
import os
from typing import Optional
from pyannote.audio import Pipeline
import torch
import torchaudio

def merge_chunks(
    segments,
    chunk_size,
    blank_threshold = 3.0,
    length_threshold = 3.0,
):
    """
    Merge operation described in paper
    """
    curr_end = 0
    merged_segments = []
    seg_idxs = []

    assert chunk_size > 0
    segments_list = list(segments.get_timeline())

    if len(segments_list) == 0:
        print("No active speech found in audio")
        return []
    # assert segments_list, "segments_list is empty."
    # Make sur the starting point is the start of the segment.
    curr_start = segments_list[0].start

    for seg in segments_list:
        # Open a new section
        if (seg.end - curr_start > chunk_size) or \
            (seg.start - curr_end > blank_threshold):
            # If previous section is not empty, add it to the list
            if curr_end-curr_start > length_threshold:
                merged_segments.append({
                    "start": curr_start,
                    "end": curr_end,
                    "segments": seg_idxs,
                })
            
            curr_start = seg.start
            seg_idxs = []
        # Add segment to current section
        curr_end = seg.end
        seg_idxs.append((seg.start, seg.end))
    # add final
    merged_segments.append({ 
                "start": curr_start,
                "end": curr_end,
                "segments": seg_idxs,
            })    
    return merged_segments

def main(args):
    rank = args.rank - 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
    max_gpu = torch.cuda.device_count()
    rank = (rank % max_gpu)
    device = torch.device(f"cuda:{rank}")

    audio_input_dir = args.audio_input_dir
    audio_output_dir = args.audio_output_dir
    audio_files = []
    with open(args.input_file, 'r') as f:
        for line in f:
            audio_files.append(line.split()[-1])
    
    # Filter out already processed files
    processed = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            for line in f:
                id = line.split()[0]
                utt = id.split("_seg")[0]
                processed.add(utt)
    
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection").to(device)
    pipeline.instantiate({
    "onset": 0.2,  # 设置语音活动开始的阈值
    "offset": 0.2,  # 设置语音活动结束的阈值
    "min_duration_on": 0.0,  # 最小语音活动持续时间
    "min_duration_off": 0.0  # 最小静音持续时间
    })

    for audio_file in audio_files:
        # Filter out already processed files
        utt = audio_file.split("/")[-1].split(".")[0]
        if utt in processed:
            print(f"Skipping {utt}")
            continue
        print(f"Processing {audio_file}")
        vad_result = pipeline(audio_file)
        vad_segments = merge_chunks(vad_result, args.seg_len, args.blank_threshold)

        # load audio
        try:
            orig_wav, ori_sr = torchaudio.load(audio_file)
        except:
            print(f"Failed to load {audio_file}")
            continue

        # Initialize saving directory
        wav_relative_path = os.path.relpath(audio_file, audio_input_dir)
        audio_dir = os.path.join(audio_output_dir, os.path.dirname(wav_relative_path))
        os.makedirs(audio_dir, exist_ok=True)
        
        num_channels = orig_wav.shape[0]
        for channel in range(num_channels):
            # Segment audio
            for i, seg in enumerate(vad_segments):
                # Save audio and transcript
                relative_path = os.path.join(os.path.splitext(wav_relative_path)[0] + f"_seg{i}_ch{channel}.wav")
                audio_path = os.path.join(audio_output_dir, relative_path)
                wav = orig_wav[channel, int(seg["start"] * ori_sr):int(seg["end"] * ori_sr)].unsqueeze(0)
                torchaudio.save(audio_path, wav, ori_sr)

                with open(args.output_file, 'a') as f:
                    f.write(f"{utt}_seg{i}_ch{channel} {relative_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--audio-input-dir", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--audio-output-dir", type=str, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--seg-len", type=int, default=105)
    parser.add_argument("--blank-threshold", type=float, default=3.0)
    args = parser.parse_args()
    main(args)