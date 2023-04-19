"""
Dataset to train a GPT for SpecVQGAN generation.
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


class AudioSpecVQGANCodebookDataset(torch.utils.data.Dataset):
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
            self.audio_files = audio_files[:int(
                len(audio_files)*cfg.train_percentage)]
            self.codes_files = codes_files[:int(
                len(codes_files)*cfg.train_percentage)]
        else:
            self.audio_files = audio_files[int(
                len(audio_files)*cfg.train_percentage):]
            self.codes_files = codes_files[int(
                len(codes_files)*cfg.train_percentage):]

        self.chunk_duration = cfg.chunk_duration
        self.clap_sr = 48_000
        self.chunk_samples = self.chunk_duration * self.clap_sr
        self.chunk_frames_22050 = int(
            ((22.05/48) * self.chunk_samples) // (256 * 16))
        self.F = 5  # from SpecVQGAN

    def __len__(self):
        return len(self.audio_files)*10

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx % len(self.audio_files)]
        codes_file = self.codes_files[idx % len(self.codes_files)]

        # Load audio waveform for CLAP
        audio_waveform, _ = librosa.load(audio_file, sr=self.clap_sr)

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

        # Chunk audio waveform
        audio_waveform = audio_waveform[
            ...,
            self._flatted_frame_to_sample_48k(
                rand_start_frame
            ):self._flatted_frame_to_sample_48k(
                rand_start_frame
            ) + self._frame_to_sample_48k(
                self.chunk_frames_22050
            )
        ]

        return audio_waveform, codes_chunk, labels_chunk

    def _flatted_frame_to_sample_48k(self, frame: int) -> int:
        return int((frame // 5) * 256 * 16 * 48 // 22.05)

    def _frame_to_sample_48k(self, frame: int) -> int:
        return int(frame * 256 * 16 * 48 // 22.05)
