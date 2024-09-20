# the inference code. refer to Meta's Encodec
import argparse
from pathlib import Path
import sys
import torchaudio
import os
from models.MimiCodec import MimiCodec
import torch
import typing as tp
from collections import OrderedDict
from omegaconf import OmegaConf

SUFFIX = '.ecdc'
def save_audio(wav: torch.Tensor, path: tp.Union[Path, str],
               sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    torchaudio.save(path, wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

def build_codec_model(config):
    model = eval(config.generator.name)(**config.generator.config)
    return model

def get_parser():
    parser = argparse.ArgumentParser(
        'encodec',
        description='High fidelity neural audio codec. '
                    'If input is a .ecdc, decompresses it. '
                    'If input is .wav, compresses it. If output is also wav, '
                    'do a compression/decompression cycle.')
    parser.add_argument(
        '--input', type=Path,
        help='Input file, whatever is supported by torchaudio on your system.')
    parser.add_argument(
        '--output', type=Path, nargs='?',
        help='Output file, otherwise inferred from input file.')
    parser.add_argument(
        '--exp_model_config', type=Path
    )
    parser.add_argument('--resume_path', type=str, default='/apdcephfs/share_1316500/donchaoyang/audio_framework/data/siri_renshe_1024utts', 
                        help='resume_path')
    parser.add_argument(
        '-r', '--rescale', action='store_true',
        help='Automatically rescale the output to avoid clipping.')
    parser.add_argument('--num_bands', type=int, default=1, help='set the model number of bands')
    parser.add_argument('--init_channel', type=int, default=16, help = 'set the init channel of encoder and decoder')
    parser.add_argument('--res_kernel_size', type=int, default=16, help = 'set the res_kernel_size of encoder and decoder')
    parser.add_argument('--causal', type=bool,  default=True, help='set whether using causal model')
    parser.add_argument('--num_samples', type=int, default=2, help='set num_samples')
    parser.add_argument('--downsample_factors', type=list, default=[2, 4, 4, 5], help='set the downsample rate')
    parser.add_argument('--downsample_kernel_sizes', type=list, default=[4, 8, 8, 10], help='set the downsample_kernel_sizes')
    parser.add_argument('--upsample_factors', type=list, default=[5, 4, 4, 2], help='reverse the downsample rate')
    parser.add_argument('--upsample_kernel_sizes', type=list, default=[10, 8, 8, 4], help='the upsample kernel sizes')
    parser.add_argument('--latent_hidden_dim', type=int, default=80, help='the upsample kernel sizes')
    parser.add_argument('--default_kernel_size', type=int, default=7, help='set the kernel size')
    parser.add_argument('--delay_kernel_size', type=int, default=5, help='set the kernel size')
    return parser


def fatal(*args):
    print(*args, file=sys.stderr)
    sys.exit(1)


def check_output_exists(args):
    if not args.output.parent.exists():
        fatal(f"Output folder for {args.output} does not exist.")
    if args.output.exists() and not args.force:
        fatal(f"Output file {args.output} exist. Use -f / --force to overwrite.")


def check_clipping(wav, args):
    if args.rescale:
        return
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr)


def check_clipping2(wav, rescale):
    if rescale:
        return
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr)

def test_one(wav_root, store_root, rescale, args, codec_model):
    #compressing
    wav, sr = torchaudio.load(wav_root)
    import time
    if sr != 24000:
        wav = convert_audio(wav, sr, 24000, 1)
    wav = wav.unsqueeze(1).cuda()
    # wav = torch.randn(1, 1, 72000).cuda()
    # print('wav ', wav.shape)
    # print('codec_model ', codec_model)
    st_time = time.time()
    with torch.no_grad():
        codes  = codec_model.encode(wav)
        #print('time2 ', time.time()-st_time)
        x = codec_model.decode(codes)
        #print('time3', time.time()-st_time)
        # assert 1==2
        # print(len(codes),codes[0].shape)
    out = x.detach().cpu().squeeze(0)
    check_clipping2(out, rescale)
    save_audio(out, store_root, 24000, rescale=rescale)
    print('finish decompressing')

def test_batch():
    args = get_parser().parse_args()
    if not args.input.exists():
        fatal(f"Input file {args.input} does not exist.")
    input_lists = os.listdir(args.input)
    input_lists.sort()
    exp_model_config = OmegaConf.load(args.exp_model_config)
    codec_model = MimiCodec(**exp_model_config.generator.config)  
    parameter_dict = torch.load(args.resume_path)
    new_state_dict = OrderedDict()
    codec_model.load_state_dict(parameter_dict['codec_model']) # load model
    codec_model = codec_model.cuda()
    os.makedirs(args.output, exist_ok=True)
    for audio in input_lists:
        test_one(os.path.join(args.input,audio), os.path.join(args.output,audio), args.rescale, args, codec_model)

if __name__ == '__main__':
    #main()
    test_batch()
