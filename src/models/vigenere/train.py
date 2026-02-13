import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from vigenere_data_module import VigenereDataModule
from vigenere_transformer import VigenereLightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar

# DataModule
dm = VigenereDataModule(
    text_path="data/processed/final_dataset_shuffled.txt",
    alphabet="abcçdefgğhıijklmnoöprsştuüvyz",
    seq_len=128,
    batch_size=64,
    min_key_len=3,
    max_key_len=12
)

dm.setup()

# Model
model = VigenereLightningModule(
    vocab_size=dm.dataset.total_vocab_size,
    max_len=128,
    max_key_len=8,
    d_model=256,
    nhead=8,
    num_layers=4,
    lr=1e-3,
    pad_token_id=dm.dataset.PAD_TOKEN_IDX
)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
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

logger = TensorBoardLogger("tb_logs", name="vigenere")

# Trainer
trainer = pl.Trainer(
    accelerator="mps",
    devices=1,
    max_epochs=20,
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