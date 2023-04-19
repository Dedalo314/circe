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


class CLAPEmbedEnCodecCodebookDataset(torch.utils.data.Dataset):
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
        # EnCodec has a downsampling of 320, and num_codebooks tokens per timestamp
        self.chunk_frames_encodec = int(
            self.chunk_duration * cfg.encodec.sampling_rate // 320
        )
        self.num_codebooks = cfg.encodec.num_codebooks
        self.chunk_ncodes_encodec = self.chunk_frames_encodec * self.num_codebooks

    def __len__(self):
        return len(self.clap_files)*5

    def __getitem__(self, idx):
        clap_file = self.clap_files[idx % len(self.clap_files)]
        codes_file = self.codes_files[idx % len(self.codes_files)]

        codes = np.lib.format.open_memmap(codes_file, dtype=np.int64)
        codes = rearrange(codes, "1 nq s -> (s nq)", nq=self.num_codebooks)
        # Start must be a token from the first codebook, so multiple of num_codebooks
        rand_start_frame = (random.randint(
            0, codes.shape[-1] - self.chunk_ncodes_encodec - 2
        ) // self.num_codebooks) * self.num_codebooks
        rand_start_frame = int(rand_start_frame)
        codes_chunk = torch.from_numpy(
            codes[rand_start_frame:rand_start_frame + self.chunk_ncodes_encodec]
        )
        labels_chunk = torch.from_numpy(
            codes[rand_start_frame:rand_start_frame + self.chunk_ncodes_encodec + 1]
        )

        clap_embed = np.load(clap_file)

        return clap_embed, codes_chunk, labels_chunk
