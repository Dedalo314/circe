import argparse
import os
import glob

import librosa
import torch
import numpy as np
from rich.progress import track
import laion_clap

def parse_args(args=None):
    parser = argparse.ArgumentParser("Precompute CLAP audio features for given .wav")
    parser.add_argument("-i", "--in_folder", type=str, required=True,
        help="Input folder with .wav to extract quantized features with EnCodec")
    parser.add_argument("-o", "--out_folder", type=str, required=True,
        help="Output folder to store extracted quantized features")
    parser.add_argument("--max_samples", type=int,
        help="Maximum no. samples in waveform at 48kHz")
    return parser.parse_args(args=args)

def main(args):
    # Instantiate a pretrained CLAP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clap = laion_clap.CLAP_Module(enable_fusion=True).to(device)
    clap.eval()
    clap.requires_grad_(False)
    audios = glob.glob(os.path.join(args.in_folder, "*.wav"))
    os.makedirs(args.out_folder, exist_ok=True)
    for audio in track(audios, description="Processing..."):
        outfile = os.path.join(args.out_folder, f"{os.path.splitext(os.path.basename(audio))[0]}.npy")
        if os.path.exists(outfile):
            continue
        wav, _ = librosa.load(audio, sr=48_000)
        if args.max_samples is not None and wav.shape[-1] > args.max_samples:
            continue
        with torch.inference_mode():
            audio_emb = clap.get_audio_embedding_from_data([wav])
        assert audio_emb.shape[-1] == 512

        np.save(
            outfile,
            audio_emb
        )

if __name__ == "__main__":
    main(parse_args())