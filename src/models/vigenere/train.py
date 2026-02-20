import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from vigenere_data_module import VigenereDataModule
from vigenere_transformer import VigenereLightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar

def main():
    MAX_K_LEN = 8
    MIN_K_LEN = 3
    MAX_LEN = 128
    BATCH_SIZE = 128
    D_MODEL = 256
    N_HEAD = 8
    NUM_LAYERS = 6
    LR = 1e-3

    # DataModule
    dm = VigenereDataModule(
        text_path="data/processed/final_dataset_shuffled.txt",
        alphabet="abcçdefgğhıijklmnoöprsştuüvyz",
        seq_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        min_key_len=MIN_K_LEN,
        max_key_len=MAX_K_LEN
    )

    dm.setup()

    # Model
    model = VigenereLightningModule(
        vocab_size=dm.dataset.total_vocab_size,
        max_len=MAX_LEN,
        max_key_len=MAX_K_LEN,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        lr=LR,
        pad_token_id=dm.dataset.PAD_TOKEN_IDX
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_vigenere2",
        filename="vigenere-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )

    progress_bar = RichProgressBar(
        leave=True
    )

    logger = TensorBoardLogger("tb_logs", name="vigenere2")

    # Trainer
    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        max_epochs=40,
        callbacks=[progress_bar, checkpoint_callback, early_stop_callback],
        logger=logger,
        precision="32-true",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train
    trainer.fit(model, datamodule=dm)

    print("Eğitim tamamlandı!")
    print(f"En iyi model: {checkpoint_callback.best_model_path}")

if __name__ == '__main__':
    main()