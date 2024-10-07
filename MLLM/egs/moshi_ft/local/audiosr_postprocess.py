import argparse
import os
import glob


def main(args):
    # Determine target path for each audio file
    utt2outpath = {}
    with open(args.input_file) as f:
        for line in f:
            id, audio_file = line.strip().split()
            target_path = os.path.join(args.audio_output_dir, audio_file)
            utt2outpath[id] = target_path
    
    # Move audio files to target path
    wav_files = glob.glob(os.path.join(args.audio_cache_dir, '**', '*.wav'), recursive=True)
    for wav_file in wav_files:
        id = os.path.basename(wav_file).split('.')[0]
        if id in utt2outpath:
            target_path = utt2outpath[id]
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            os.rename(wav_file, target_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--audio-original-root", type=str, required=True)
    parser.add_argument("--audio-cache-dir", type=str, required=True)
    parser.add_argument("--audio-output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args)