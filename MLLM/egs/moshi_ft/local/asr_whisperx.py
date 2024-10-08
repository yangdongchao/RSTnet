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
    model = whisperx.load_model("medium.en", device="cuda", device_index=rank, compute_type=args.compute_type)
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)

    audio_input_dir = args.audio_input_dir
    metadata_output_dir = args.metadata_output_dir
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
                processed.add(id)

    for audio_file in audio_files:
        # Filter out already processed files
        id = audio_file.split("/")[-1].split(".")[0]
        if id in processed:
            print(f"Skipping {id}")
            continue
        print(f"Processing {audio_file}")

        try:
            audio_file_path = os.path.join(audio_input_dir, audio_file)
            audio = whisperx.load_audio(audio_file_path)
        except:
            print(f"Failed to load {audio_file}")
            continue

        result = model.transcribe(audio, batch_size=args.batch_size, language="en")
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            
        # Save audio and transcript
        json_dir = os.path.join(metadata_output_dir, os.path.dirname(audio_file))
        os.makedirs(json_dir, exist_ok=True)
        json_path = os.path.join(metadata_output_dir, os.path.splitext(audio_file)[0] + ".json")
        
        with open(json_path, 'w') as f:
            json.dump({
                "wav": audio_file, 
                "duration": audio.shape[-1] / 16000,
                "segments": result["segments"], 
                "word_segments": result["word_segments"], 
                }, f, ensure_ascii=False)

        # Save the list of new audio files
        scp_dir = os.path.dirname(args.output_file)
        metadata_rel_path = os.path.splitext(audio_file)[0] + ".json"
        with open(os.path.join(scp_dir, f"utt2json.{args.rank}"), 'a') as f:
            f.write(f"{id} {metadata_rel_path}\n")
        
        with open(args.output_file, 'a') as f:
            f.write(f"{id} {audio_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--audio-input-dir", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--metadata-output-dir", type=str, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compute-type", type=str, default="float16")
    args = parser.parse_args()
    main(args)
