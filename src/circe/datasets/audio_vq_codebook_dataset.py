"""
Dataset to train a GPT for EnCodec codebook 1 generation.
"""
import os
import logging
import random
import glob

import numpy as np
import torch
import librosa
from einops import rearrange

logger = logging.getLogger(__name__)

class AudioVQCodebookDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()

        codes_files = glob.glob(os.path.join(cfg.codes_data_dir, "*.npy"))
        random.Random(0).shuffle(codes_files)
        audio_files = [
            os.path.join(
                cfg.audio_data_dir,
                f"{os.path.splitext(os.path.basename(codes_file))[0]}.wav"
            ) for codes_file in codes_files
        ]

        if split == "train":
            self.audio_files = audio_files[:int(len(audio_files)*cfg.train_percentage)]
            self.codes_files = codes_files[:int(len(codes_files)*cfg.train_percentage)]
        else:
            self.audio_files = audio_files[int(len(audio_files)*cfg.train_percentage):]
            self.codes_files = codes_files[int(len(codes_files)*cfg.train_percentage):]

        self.chunk_duration = cfg.chunk_duration
        self.clap_sr = 48_000
        self.chunk_samples = self.chunk_duration * self.clap_sr
        self.chunk_frames_22050 = int(((22.05/48) * self.chunk_samples) // (256 * 16))
        self.F = 5 # from SpecVQGAN

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        codes_file = self.codes_files[idx]

        # Load audio waveform for CLAP
        audio_waveform, _ = librosa.load(audio_file, sr=self.clap_sr)

        # Load codes from SpecVQGAN
        # Codes are FxT, where F=5. Upsampling rate is 1/16 in spectrogram
        # Spectrogram 256 frame size (computed with an example)
        # T = samples / (256*16)
        # The error is because the 5 is missing in the rand start
        codes = np.lib.format.open_memmap(codes_file, dtype=np.int64)
        codes = rearrange(codes, "(f t) 1 -> f t", f=self.F)
        # print(f"{audio_waveform.shape=}")
        # print(f"{codes.shape=}")
        max_len = int(((22.05/48) * audio_waveform.shape[-1]) // (256 * 16)) - self.chunk_frames_22050 - 2
        rand_start_frame = random.randint(0, max_len)
        while rand_start_frame > codes.shape[-1] - 2 - self.chunk_frames_22050:
            # For some cases it happens, the downsampling is not exactly 256 * 16, do not know why
            logger.warning("Batch does not have an exact downsampling rate")
            rand_start_frame = random.randint(0, max_len)
        # print(f"rand_start={rand_start_frame*16*256}\n{rand_start_frame=}\n{self.chunk_frames_22050=}")
        codes_chunk = torch.from_numpy(codes[:, rand_start_frame:rand_start_frame + self.chunk_frames_22050])
        labels_chunk = torch.from_numpy(codes[:, rand_start_frame+1:rand_start_frame + self.chunk_frames_22050 + 1])
        codes_chunk = rearrange(codes_chunk, "f t -> (t f)", f=self.F)
        labels_chunk = rearrange(labels_chunk, "f t -> (t f)", f=self.F)

        # Chunk audio waveform
        audio_waveform = audio_waveform[..., rand_start_frame*256*16:rand_start_frame*256*16 + self.chunk_frames_22050*256*16]
        # print(f"{codes_chunk.shape=}\n{labels_chunk.shape=}\n{audio_waveform.shape=}\n")
        
        return audio_waveform, codes_chunk, labels_chunk

    def _sample_to_frame(self, sample: int) -> int:
        return int(sample // (256 * 16))

    def _frame_to_sample(self, frame: int) -> int:
        return frame * 16 * 256