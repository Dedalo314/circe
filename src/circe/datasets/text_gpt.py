"""
Dataset to train a GPT for text generation.
"""
import os
import logging
import random

import numpy as np
import torch

class TextGPTDataset(torch.utils.data.Dataset):
    """
    Dataset as in nanoGPT:
    https://github.com/karpathy/nanoGPT/blob/ae3a8d5fdd3ddb8b13fab182723476523961e3ab/train.py#L108
    """
    def __init__(self, cfg, split):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        if split == "train":
            self.data = np.memmap(os.path.join(cfg.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            self.data = np.memmap(os.path.join(cfg.data_dir, 'val.bin'), dtype=np.uint16, mode='r')

        self.block_size = cfg.block_size

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        i = random.randint(0, len(self.data) - self.block_size - 1)
        x = torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64))
        y = torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64))

        return x, y
