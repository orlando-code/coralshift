from coralshift.machine_learning.transformer.transformer import (
    TSTransformerEncoderClassiregressor,
)
from coralshift.machine_learning.trainer.trainer import Trainer
from coralshift.machine_learning import transformer_utils

import torch


def run_all():
    dh = transformer_utils.get_data()

    for lr in [1e-4]:
        model = TSTransformerEncoderClassiregressor(
            feat_dim=12,
            d_model=64,
            max_len=1000,
            n_heads=8,
            num_layers=6,
            dim_feedforward=512,
            num_classes=1,
            dropout=0.1,
            pos_encoding="learnable",
            activation="gelu",
            norm="BatchNorm",
            freeze=False,
        )
        trainer = Trainer(dh=dh, epochs=1)
        dh.create_dataset()
        dh.split_data(train_split=0.8)
        dataloader_train = dh.create_dataloader(dh.train_data, batch_size=8)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        trainer.fit(dataloader=dataloader_train, model=model, optimiser=optimiser)
        dataloader_test = dh.create_dataloader(dh.test_data, batch_size=8)
        accuracy = trainer.evaluate(dataloader=dataloader_test, model=model)
        print(accuracy)
