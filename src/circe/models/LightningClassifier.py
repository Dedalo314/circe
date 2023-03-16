import logging

from torch import Tensor, nn, no_grad, topk, multinomial, cat
import pytorch_lightning as pl

from circe.utils.import_class import import_class

class LightningClassifier(pl.LightningModule):
    r"""
    Classifier abstraction for different models

    The model passed to the init is used in the forward.
    """
    def __init__(self, cfg) -> None:
        super(LightningClassifier, self).__init__()
        self._conf = cfg
        self.learning_rate = cfg.lr
        self.model_class = import_class(cfg.model_class)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)

    @no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self._conf.block_size else idx[:, -self._conf.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = cat((idx, idx_next), dim=1)

        return idx

    def configure_sharded_model(self) -> None:
        self.classifier = self.model_class(self._conf)

    def configure_optimizers(self):
        optimizer = self.classifier.configure_optimizers(
            self._conf.weight_decay,
            self.learning_rate,
            (self._conf.beta1, self._conf.beta2)
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        logits = self(X)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        self.log("Loss/train", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        self._shared_eval(val_batch, batch_idx, "val")

    def test_step(self, test_batch, batch_idx):
        self._shared_eval(test_batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        X, y = batch
        logits = self(X)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        self.log(f"Loss/{prefix}", loss)