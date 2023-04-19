"""
Dataset to train a GPT for SpecVQGAN generation.
"""
import os
import logging
import random
import glob

import numpy as np
import torch
from einops import rearrange

class EnCodecCodebookDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()

        codes_files = glob.glob(os.path.join(cfg.codes_data_dir, "*.npy"))
        random.Random(0).shuffle(codes_files)

        if split == "train":
            self.codes_files = codes_files[:int(len(codes_files)*cfg.train_percentage)]
        else:
            self.codes_files = codes_files[int(len(codes_files)*cfg.train_percentage):]

        print(f"{split=}\n{self.codes_files=}\n")
        self.chunk_duration = cfg.chunk_duration
        # EnCodec has a downsampling of 320, and num_codebooks tokens per timestamp
        self.chunk_frames_encodec = int(
            self.chunk_duration * cfg.encodec.sampling_rate // 320
        )
        self.num_codebooks = cfg.encodec.num_codebooks
        self.chunk_ncodes_encodec = self.chunk_frames_encodec * self.num_codebooks

    def __len__(self):
        return len(self.codes_files)*10

    def __getitem__(self, idx):
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

        return codes_chunk, labels_chunk
