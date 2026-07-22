import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse

import lightning.pytorch as pl
import omegaconf
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, Timer
from prime_torch import SWRegressor

# Add the prime_torch file to the system path so we can import it
# import sys
# sys.path.append("/glade/u/home/cobrien/prime/prime_lib/primesw")
from data import SWDataModule


def main(config):
    torch.set_float32_matmul_precision("medium")
    cfg = omegaconf.OmegaConf.load(config)

    datamodule = SWDataModule(
        target_features=cfg.data.target_features,
        input_features=cfg.data.input_features,
        position_features=cfg.data.position_features,
        interp_flags=cfg.data.interp_flags,
        region=cfg.data.region,
        cuts=cfg.data.cuts,
        cadence=cfg.data.cadence,
        interpolate=cfg.data.interpolate,
        window=cfg.data.window,
        stride=cfg.data.stride,
        interp_frac=cfg.data.interp_frac,
        trn_bounds=cfg.data.trn_bounds,
        val_bounds=cfg.data.val_bounds,
        tst_bounds=cfg.data.tst_bounds,
        batch_size=cfg.opt.batch_size,
        num_workers=cfg.opt.num_workers,
        datastore=cfg.data.datastore,
        in_key=cfg.data.in_key,
        tar_key=cfg.data.tar_key,
        scaler_type=cfg.data.scaler_type,
    )

    model = SWRegressor(
        optimizer=cfg.opt.optimizer,
        lr=cfg.opt.lr,
        lr_scheduler=cfg.opt.lr_scheduler,
        patience=cfg.opt.patience,
        factor=cfg.opt.factor,
        weight_decay=cfg.opt.weight_decay,
        total_iters=cfg.opt.total_iters,
        in_dim=len(cfg.data.input_features),
        tar_dim=len(cfg.data.target_features),
        pos_dim=len(cfg.data.position_features),
        in_norm=datamodule.input_normalizations,
        tar_norm=datamodule.target_normalizations,
        pos_norm=datamodule.position_normalizations,
        window=cfg.data.window,
        stride=cfg.data.stride,
        interp_frac=cfg.data.interp_frac,
        decoder_type=cfg.model.decoder_type,
        encoder_type=cfg.model.encoder_type,
        decoder_hidden_layers=cfg.model.decoder_hidden_layers,
        encoder_hidden_dim=cfg.model.encoder_hidden_dim,
        encoder_num_layers=cfg.model.encoder_num_layers,
        p_drop=cfg.model.p_drop,
        pos_encoding_size=cfg.model.pos_encoding_size,
        loss=cfg.opt.loss,
        save_debug_ckpt=cfg.experiments.save_debug_ckpt,
        save_predictions=True,
    )
    # checkpoint_path = "../data/prime/checkpoints/final_model.ckpt"

    # model = SWRegressor.load_from_checkpoint(
    #     checkpoint_path,
    #     in_norm=datamodule.input_normalizations,
    #     tar_norm=datamodule.target_normalizations,
    #     pos_norm=datamodule.position_normalizations,
    #     weights_only=False,
    # )
    # TODO: implement tensorboard logger
    tb_logger = pl.loggers.TensorBoardLogger(cfg.experiments.trainer.tensorboard_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.experiments.checkpoint,  # Maps to "../data/prime/checkpoints"
        filename="prime-model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,  # Saves only the single best model
        monitor="Loss/val",  # Or whatever your validation metric key is named
        mode="min",  # Minimizing the loss
    )

    trainer = pl.Trainer(
        num_sanity_val_steps=0,  # Skip sanity check to speed up testing
        accelerator=cfg.experiments.trainer.accelerator,
        max_epochs=cfg.experiments.trainer.max_epochs,
        callbacks=[Timer(), RichProgressBar(), checkpoint_callback],
        logger=tb_logger,
        precision=cfg.experiments.trainer.precision,  # Lower the precision to not blow up memory
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.save_checkpoint("data/prime/checkpoints/final_model.ckpt")

    trainer.test(model=model, datamodule=datamodule)
    # predict_raw = model(datamodule.tst_ds.input_data, datamodule.tst_ds.position_data)
    # preds_array = predict_raw.detach().cpu().numpy()
    # df = pd.DataFrame(preds_array)
    # df.to_csv("predictions.csv", index=False)

    # for idx, feature in enumerate(datamodule.target_features):
    #     predict_scaled = (predict_raw[:, idx] * datamodule.tst_ds.target_normalizations[feature][1]) + datamodule.tst_ds.target_normalizations[
    #         feature
    #     ][0]
    #     obs_scaled = (
    #         datamodule.tst_ds.target_data[:, idx] * datamodule.tst_ds.target_normalizations[feature][1]
    #     ) + datamodule.tst_ds.target_normalizations[feature][0]
    #     mae = np.mean(np.abs(predict_scaled.detach().numpy() - obs_scaled.detach().numpy()))
    #     print(f"{feature} MAE: {mae}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Single training run of PRIME.")
    parser.add_argument(
        "--config",
        type=str,
        default="PRIME/testing_config.yaml",
        help="Path to config file defining training run.",
    )
    args = parser.parse_args()
    main(args.config)
