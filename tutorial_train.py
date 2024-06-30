from share import *

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torch.utils.data import DataLoader, Subset
import random
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import types
import argparse


def init(is_debug):
    # Configs
    resume_path = './models/control_sd15_ini.ckpt'
    batch_size = 4
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Add configure_optimizers method to the model
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": 'train/loss_epoch',
                "frequency": 1
            },
        }

    model.use_scheduler = True
    model.configure_optimizers = configure_optimizers.__get__(model)


    def training_step(self, batch, batch_idx):
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss, loss_dict = self.shared_step(batch)

        # バッチサイズを取得
        if isinstance(batch, dict):
            # ディクショナリの最初の値を使用してバッチサイズを取得
            batch_size = next(iter(batch.values())).size(0)
        elif isinstance(batch, (tuple, list)):
            batch_size = batch[0].size(0)
        else:
            batch_size = batch.size(0)
        
        self.log_dict(loss_dict, prog_bar=True,
                        logger=True, on_step=True, on_epoch=True, batch_size=batch_size)

        self.log("global_step", self.global_step,
                    prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=batch_size)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=batch_size)

        return loss

    model.training_step = types.MethodType(training_step, model)


    full_dataset = MyDataset('sotai_sketch', augment=True)
    if is_debug:
        num_samples = 100
        indices = random.sample(range(len(full_dataset)), num_samples)
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset

    dataloader = DataLoader(
        dataset,
        num_workers=16,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True
    )


    image_logger = ImageLogger(".", batch_frequency=logger_freq)
    wandb_logger = WandbLogger()
    tb_logger = TensorBoardLogger("tb_logs")

    interval_checkpoint_callback = ModelCheckpoint(
        every_n_epochs=100,
        monitor=None,
        dirpath="./output",
        filename="Sotai_sketch_ControlNet_epoch={epoch:04d}_train_loss_epoch={train/loss_epoch:.4e}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        save_last=True,
        verbose=True,
    )

    min_checkpoint_callback = ModelCheckpoint(
        dirpath="./output",
        filename="Sotai_sketch_ControlNet_epoch={epoch:04d}_train_loss_epoch={train/loss_epoch:.4e}",
        auto_insert_metric_name=False,
        save_top_k=5,
        monitor='train/loss_epoch',
        mode='min',
        every_n_epochs=1,
        save_last=True,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        gpus=[1],
        precision=32,
        logger=[wandb_logger, tb_logger],
        callbacks=[interval_checkpoint_callback, min_checkpoint_callback, image_logger, lr_monitor],
    )
    # import inspect
    # print(inspect.getsource(model.__init__))
    
    return model, dataloader, trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, type=bool)

    model, dataloader, trainer = init(parser.parse_args().debug)
    trainer.fit(model, dataloader)