"""
Dataset to train a GPT for EnCodec codebook 1 generation.
"""
import os
import logging
import random
import glob

import numpy as np
import torch
from einops import rearrange

logger = logging.getLogger(__name__)


class CLAPEmbedSpecVQGANCodebookDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()

        codes_files = glob.glob(os.path.join(cfg.codes_data_dir, "*.npy"))
        random.Random(0).shuffle(codes_files)
        clap_files = [
            os.path.join(
                cfg.clap_embeds_data_dir,
                f"{os.path.splitext(os.path.basename(codes_file))[0]}.npy"
            ) for codes_file in codes_files
        ]

        if split == "train":
            self.clap_files = clap_files[:int(
                len(clap_files)*cfg.train_percentage)]
            self.codes_files = codes_files[:int(
                len(codes_files)*cfg.train_percentage)]
        else:
            self.clap_files = clap_files[int(
                len(clap_files)*cfg.train_percentage):]
            self.codes_files = codes_files[int(
                len(codes_files)*cfg.train_percentage):]

        self.chunk_duration = cfg.chunk_duration
        self.clap_sr = 48_000
        self.chunk_samples = self.chunk_duration * self.clap_sr
        self.chunk_frames_22050 = int(
            ((22.05/48) * self.chunk_samples) // (256 * 16))
        self.F = 5  # from SpecVQGAN

    def __len__(self):
        return len(self.clap_files)*5

    def __getitem__(self, idx):
        clap_file = self.clap_files[idx % len(self.clap_files)]
        codes_file = self.codes_files[idx % len(self.codes_files)]

        # Load codes from SpecVQGAN
        # Codes are FxT, where F=5. Upsampling rate is 1/16 in spectrogram
        # Spectrogram 256 frame size (computed with an example)
        # T = samples / (256*16)
        # The error is because the 5 is missing in the rand start
        codes = np.lib.format.open_memmap(codes_file, dtype=np.int64)
        codes = rearrange(codes, "(f t) 1 -> (t f)", f=self.F)
        rand_start_frame = random.randint(
            0, codes.shape[-1] - self.chunk_frames_22050 * self.F - 2
        )
        codes_chunk = torch.from_numpy(
            codes[rand_start_frame:rand_start_frame + self.chunk_frames_22050 * self.F])
        labels_chunk = torch.from_numpy(
            codes[rand_start_frame:rand_start_frame + self.chunk_frames_22050 * self.F + 1])

        clap_embed = np.load(clap_file)

        return clap_embed, codes_chunk, labels_chunk
