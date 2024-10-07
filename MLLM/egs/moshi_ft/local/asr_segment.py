import argparse
import json
import os

import torch
import torchaudio
import whisperx

def main(args):
    rank = args.rank - 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
    max_gpu = torch.cuda.device_count()
    rank = (rank % max_gpu)
    device = torch.device(f"cuda:{rank}")

    model = whisperx.load_model("large-v2", device="cuda", device_index=rank, compute_type=args.compute_type)
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)

    audio_input_dir = args.audio_input_dir
    audio_output_dir = args.audio_output_dir
    metadata_output_dir = args.metadata_output_dir
    audio_files = []
    with open(args.input_file, 'r') as f:
        for line in f:
            audio_files.append(line.split()[-1])

    # Filter out already processed files
    processed = set()
    scp_dir = os.path.dirname(args.input_file)
    with open(os.path.join(scp_dir, f"wav_seg.{args.rank}.scp"), 'r') as f:
        for line in f:
            id = line.split()[0]
            utt = line.split("_seg")[0]
            processed.add(utt)

    for audio_file in audio_files:
        if audio_file.split("/")[-1].split(".")[0] in processed:
            utt = audio_file.split("/")[-1].split(".")[0]
            print(f"Skipping {utt}")
            continue
        print(f"Processing {audio_file}")

        try:
            orig_wav, ori_sr = torchaudio.load(audio_file)
        except:
            print(f"Failed to load {audio_file}")
            continue
        wav = torchaudio.transforms.Resample(ori_sr, 16000)(orig_wav)

        # Step 1: VAD segmentation
        audio = wav.sum(dim=0)
        vad_segments = model.vad_model({"waveform": audio.unsqueeze(0), "sample_rate": 16000})
        vad_segments = whisperx.vad.merge_chunks(
            vad_segments,
            chunk_size=args.seg_len,
            onset=model._vad_params["vad_onset"],
            offset=model._vad_params["vad_offset"],
        )
        # Merge small segments
        if vad_segments[-1]["end"] - vad_segments[0]["start"] < 5:
            vad_segments[-2]["end"] = vad_segments[-1]["end"]
            vad_segments.pop(-1)
                    
        # Initialize saving directory
        wav_relative_path = os.path.relpath(audio_file, audio_input_dir)
        audio_dir = os.path.join(audio_output_dir, os.path.dirname(wav_relative_path))
        os.makedirs(audio_dir, exist_ok=True)
        json_dir = os.path.join(metadata_output_dir, os.path.dirname(wav_relative_path))
        os.makedirs(json_dir, exist_ok=True)

        # Segment audio and transcript
        new_audio_files = []
        new_json_files = []
        num_channels = orig_wav.shape[0]
        for channel in range(num_channels):
            # Segment audio
            wav_segments = []
            # for end_time in boader_time:
            for seg in vad_segments:
                wav = orig_wav[channel, int(seg["start"] * ori_sr):int(seg["end"] * ori_sr)].unsqueeze(0)
                wav_segments.append(wav)
            
            # Transcribe and align each segment
            for i, wav_segment in enumerate(wav_segments):
                audio = torchaudio.transforms.Resample(ori_sr, 16000)(wav_segment)[0]
                audio = audio.numpy()
                result = model.transcribe(audio, batch_size=args.batch_size, language="en")
                result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            
                # Save audio and transcript
                audio_path = os.path.join(audio_output_dir, os.path.splitext(wav_relative_path)[0] + f"_seg{i}_ch{channel}.wav")
                json_path = os.path.join(metadata_output_dir, os.path.splitext(wav_relative_path)[0] + f"_seg{i}_ch{channel}.json")
                
                id = os.path.splitext(os.path.basename(audio_path))[0]
                new_audio_files.append((id, os.path.splitext(wav_relative_path)[0] + f"_seg{i}_ch{channel}.wav"))
                new_json_files.append((id, os.path.splitext(wav_relative_path)[0] + f"_seg{i}_ch{channel}.json"))

                torchaudio.save(audio_path, wav_segment, ori_sr)
                with open(json_path, 'w') as f:
                    json.dump({
                        "wav": os.path.relpath(audio_path, audio_output_dir), 
                        "duration": wav_segment.shape[1] / ori_sr,
                        "segments": result["segments"], 
                        "word_segments": result["word_segments"], 
                        }, f, ensure_ascii=False)

        # Save the list of new audio files
        scp_dir = os.path.dirname(args.input_file)
        with open(os.path.join(scp_dir, f"wav_seg.{args.rank}.scp"), 'a') as f:
            for id, path in new_audio_files:
                f.write(f"{id} {path}\n")
        with open(os.path.join(scp_dir, f"utt2json.{args.rank}"), 'a') as f:
            for id, path in new_json_files:
                f.write(f"{id} {path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--audio-input-dir", type=str, required=True)
    parser.add_argument("--audio-output-dir", type=str, required=True)
    parser.add_argument("--metadata-output-dir", type=str, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compute-type", type=str, default="float16")  # change to "int8" if low on GPU memory
    parser.add_argument("--seg-len", type=int, default=105)
    args = parser.parse_args()
    main(args)
