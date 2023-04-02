import logging

import torch
from torch import nn
import torchaudio
from einops import rearrange
from transformers import GPT2Config, GPT2LMHeadModel, RobertaTokenizer
from colossalai.nn.optimizer import HybridAdam
import laion_clap

logger = logging.getLogger(__name__)

class CirceGPT_online(nn.Module):
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

        # CLAP: https://github.com/LAION-AI/CLAP
        self.clap = laion_clap.CLAP_Module(enable_fusion=True)
        self.clap.eval()
        self.clap.requires_grad_(False)


    def forward(self, codes, audio_waveform=None, inference_text=None, attention_mask=None):
        # CLAP: https://github.com/LAION-AI/CLAP/blob/main/src/training/infer_demo.py
        if audio_waveform is not None:
            with torch.inference_mode(), torch.cuda.amp.autocast():
                embed = torch.from_numpy(
                    self.clap.get_audio_embedding_from_data(audio_waveform.cpu().numpy())
                )
        elif inference_text is not None:
            raise NotImplementedError
            # load the text, can be a list (i.e. batch size)
            text_data = [inference_text]*codes.shape[0]
            # tokenize for roberta, if you want to tokenize for another text encoder, please refer to data.py#L43-90
            text_data = self._tokenizer(text_data)
            with torch.inference_mode():
                embed = self.clap.get_text_embedding(text_data)
        elif inference_text is not None and audio_waveform is not None:
            raise ValueError("Either audio_waveform or inference_text must be None")
        else:
            raise ValueError("Either audio_waveform or inference_text must not be None")

        # Project audio embeddings (and codes if necessary) to GPT dimension
        with torch.cuda.amp.autocast():
            embed = rearrange(self.wc(embed.clone().to("cuda")), "b h -> b 1 h")
            codes = self.we(codes)
        print(f"{embed.shape=}\n{codes.shape=}\n")
        inputs_embeds = torch.cat([embed, codes], 1)
        print(f"{inputs_embeds=}")
        
        # Only return lm_logits
        return self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )[0]

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        return HybridAdam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)