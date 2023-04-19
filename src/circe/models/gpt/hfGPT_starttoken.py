import torch
from transformers import GPT2Config, GPT2LMHeadModel
from colossalai.nn.optimizer import HybridAdam
from einops import repeat, rearrange

class GPTLMModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = GPT2LMHeadModel(
            GPT2Config(
                n_embd=cfg.n_embd,
                n_layer=cfg.n_layer,
                n_head=cfg.n_head,
                n_positions=cfg.n_positions,
                vocab_size=cfg.vocab_size + 1,
                return_dict=False
            )
        )
        self.start_token = torch.tensor([cfg.vocab_size])

    def forward(self, input_ids, attention_mask=None):
        # Only return lm_logits
        if input_ids.numel() > 0:
            input_ids_with_start = torch.cat(
                [
                    repeat(self.start_token.to(input_ids.device), "1 -> b 1", b=input_ids.shape[0]),
                    input_ids
                ],
                dim=1
            )
        else:
            input_ids_with_start = repeat(self.start_token.to(input_ids.device), "1 -> b 1", b=1)
        return self.model(input_ids=input_ids_with_start, attention_mask=attention_mask)[0]

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        return HybridAdam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
