from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from colossalai.nn.optimizer import HybridAdam

class GPTLMModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = GPT2LMHeadModel(
            GPT2Config(
                n_embd=cfg.n_embd,
                n_layer=cfg.n_layer,
                n_head=cfg.n_head,
                n_positions=cfg.n_positions,
                vocab_size=cfg.vocab_size,
                return_dict=False
            )
        )

    def forward(self, input_ids, attention_mask=None):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        return HybridAdam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
