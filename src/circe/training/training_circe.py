"""
Training script using hydra.cc and PL.
"""
import logging

import hydra
import pytorch_lightning as pl

from circe.utils.import_class import import_class
from circe.models.LightningARCLAP import LightningARCLAP
from circe.data.LightningDataModule import LightningDataModule

# Default hydra logger
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    data_module = LightningDataModule(cfg=cfg.data)

    logger.info(f"Batch size: {cfg.data.train.batch_size}")

    classifier = LightningARCLAP(cfg=cfg.model)

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.epochs,
        default_root_dir=cfg.trainer.default_root_dir,
        strategy=pl.strategies.colossalai.ColossalAIStrategy(
            placement_policy="auto",
            initial_scale=32
        ),
        num_sanity_val_steps=0,
        precision=16,
        log_every_n_steps=cfg.trainer.log_every_n_steps
    )
    if "checkpoint" in cfg.model:
        trainer.fit(classifier, data_module, ckpt_path=cfg.model.checkpoint)
    else:
        trainer.fit(classifier, data_module)


try:
    main()
except Exception as ex:
    logger.exception(ex)
    raise
