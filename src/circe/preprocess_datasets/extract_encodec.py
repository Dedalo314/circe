import argparse
import os
import glob

from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import torch
import numpy as np
from rich.progress import track

def parse_args(args=None):
    parser = argparse.ArgumentParser("Precompute EnCodec features for given .wav")
    parser.add_argument("-i", "--in_folder", type=str, required=True,
        help="Input folder with .wav to extract quantized features with EnCodec")
    parser.add_argument("-o", "--out_folder", type=str, required=True,
        help="Output folder to store extracted quantized features")
    parser.add_argument("--max_samples", type=int,
        help="Maximum no. samples in waveform at 24 kHz")
    return parser.parse_args(args=args)

def main(args):
    # Instantiate a pretrained EnCodec model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncodecModel.encodec_model_24khz()
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    model.set_target_bandwidth(3.0)

    audios = glob.glob(os.path.join(args.in_folder, "*.wav"))
    os.makedirs(args.out_folder, exist_ok=True)
    for audio in track(audios, description="Extracting with EnCodec..."):
        outfile = os.path.join(args.out_folder, f"{os.path.splitext(os.path.basename(audio))[0]}.npy")
        if os.path.exists(outfile):
            continue
        # Load and pre-process the audio waveform
        wav, sr = torchaudio.load(audio)
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        if args.max_samples is not None and wav.shape[-1] > args.max_samples:
            wav = wav[..., :args.max_samples]
        wav = wav.to(device)
        wav = wav.unsqueeze(0)

        # Extract discrete codes from EnCodec
        with torch.inference_mode():
            encoded_frames = model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
        np.save(
            outfile,
            codes.cpu().numpy()
        )

if __name__ == "__main__":
    main(parse_args())
