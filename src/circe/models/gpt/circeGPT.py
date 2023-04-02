import logging

import torch
from torch import nn
import torchaudio
from einops import rearrange
from transformers import GPT2Config, GPT2LMHeadModel, RobertaTokenizer
from colossalai.nn.optimizer import HybridAdam

logger = logging.getLogger(__name__)

class CirceGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # GPT2 for LM
        self.gpt2 = GPT2LMHeadModel(
            GPT2Config(
                n_embd=cfg.n_embd,
                n_layer=cfg.n_layer,
                n_head=cfg.n_head,
                n_positions=cfg.n_positions,
                vocab_size=cfg.vocab_size,
                return_dict=False
            )
        )
        # 512 comes from CLAP embedding dimension
        self.wc = nn.Linear(512, cfg.n_embd, bias=False)
        self.we = nn.Embedding(cfg.codebook_size, cfg.n_embd)

    def forward(self, codes, clap_embedding, attention_mask=None):
        # Project audio embeddings (and codes if necessary) to GPT dimension
        print(f"\n{codes.shape=}\n{clap_embedding.shape=}\n",)
        with torch.cuda.amp.autocast():
            clap_embed = self.wc(clap_embedding)
            if codes.numel() > 0:
                codes = self.we(codes)
        # print(f"{clap_embed.shape=}\n{codes.shape=}\n")
        if codes.numel() > 0:
            inputs_embeds = torch.cat([clap_embed, codes], 1)
        else:
            inputs_embeds = clap_embed
        print(f"{inputs_embeds.shape=}\n")

        # Only return lm_logits
        return self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )[0]

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        return HybridAdam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
