import argparse
import os
import glob

from circe.specvqgan.feature_extraction.demo_utils import (calculate_codebook_bitrate,
                                                           extract_melspectrogram,
                                                           get_audio_file_bitrate,
                                                           get_duration,
                                                           load_neural_audio_codec)
from circe.specvqgan.sample_visualization import tensor_to_plt
from torch.utils.data.dataloader import default_collate
import torchaudio
import torch
import numpy as np
from tqdm import tqdm


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        "Precompute SpecVQGAN discrete codes for given .wav")
    parser.add_argument("-i", "--in_folder", type=str, required=True,
                        help="Input folder with .wav to extract discrete indeces with SpecVCQAN")
    parser.add_argument("-o", "--out_folder", type=str, required=True,
                        help="Output folder to store extracted discrete indeces")
    return parser.parse_args(args=args)


def main(args):
    # Instantiate a pretrained EnCodec model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = '2021-05-19T22-16-54_vggsound_codebook'
    log_dir = '/home/daedalus/Development/Circe/vggsound'
    # loading the models might take a few minutes
    config, model, _ = load_neural_audio_codec(model_name, log_dir, device)
    model.eval()
    model.requires_grad_(False)
    model_sr = config.data.params.sample_rate

    audios = glob.glob(os.path.join(args.in_folder, "*.wav"))
    os.makedirs(args.out_folder, exist_ok=True)
    for audio in tqdm(audios, total=len(audios)):
        outfile = os.path.join(
            args.out_folder, f"{os.path.splitext(os.path.basename(audio))[0]}.npy")
        if os.path.exists(outfile):
            continue
        # Spectrogram Extraction
        duration = get_duration(audio)
        spec = extract_melspectrogram(audio, sr=model_sr, duration=duration)

        # Prepare Input
        spectrogram = {'input': spec}
        batch = default_collate([spectrogram])
        batch['image'] = batch['input'].to(device)
        x = model.get_input(batch, 'image')

        with torch.inference_mode():
            _, _, info = model.encode(x)

        x = x.cpu()
        batch['image'] = batch['image'].cpu()

        np.save(
            outfile,
            info[2].cpu().numpy()
        )


if __name__ == "__main__":
    main(parse_args())
