import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data_module import VigenereDataModule
from vigenere import VigenereBlindSolver

def main():
    MAX_K_LEN = 12
    MIN_K_LEN = 3
    BATCH_SIZE = 32
    
    dm = VigenereDataModule(
        text_path="data/processed/final_dataset_shuffled.txt",
        alphabet="abcçdefgğhıijklmnoöprsştuüvyz",
        batch_size=BATCH_SIZE,
        min_key_len=MIN_K_LEN,
        max_key_len=MAX_K_LEN
    )
    dm.setup()

    model = VigenereBlindSolver(
        vocab_size=dm.dataset.crypto_vocab_size,
        min_key_len=MIN_K_LEN,
        max_key_len=MAX_K_LEN,
        d_model=256,
        nhead=4,
        num_layers=4
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_blind",
        filename="predictor-{epoch:02d}-{val_acc:.3f}-{val_top2_acc:.3f}",
        save_top_k=2,
        monitor="val_acc",
        mode="max"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=10,
        mode="max"
    )

    logger = TensorBoardLogger("tb_logs", name="key_recovery")

    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        max_epochs=150,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        accumulate_grad_batches=4
    )

    #trainer.fit(model, datamodule=dm)
    trainer.fit(model, datamodule=dm, ckpt_path="checkpoints_blind/predictor-epoch=32-val_acc=0.758-val_top2_acc=0.000.ckpt")

if __name__ == '__main__':
    main()