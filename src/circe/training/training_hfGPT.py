"""
Training script using hydra.cc and PL.
"""
import logging

import hydra
import lightning.pytorch as pl
from lightning_colossalai import ColossalAIStrategy
import torch

from circe.utils.import_class import import_class
from circe.models.LightningClassifier import LightningClassifier
from circe.data.LightningDataModule import LightningDataModule

# Default hydra logger
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    data_module = LightningDataModule(cfg=cfg.data)

    logger.info(f"Batch size: {cfg.data.train.batch_size}")

    classifier = LightningClassifier(cfg=cfg.model)

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.epochs,
        default_root_dir=cfg.trainer.default_root_dir,
        strategy=ColossalAIStrategy(
            placement_policy="auto",
            initial_scale=32
        ),
        num_sanity_val_steps=0,
        precision=16,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        limit_val_batches=0
    )
    trainer.fit(classifier, data_module)


try:
    main()
except Exception as ex:
    logger.exception(ex)
    raise
