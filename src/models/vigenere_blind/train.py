import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger

from data_module import VigenereDataModule
from vigenere import VigenereBlindSolver

class CurriculumCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        
        if epoch < 30:
            stage = 0
        elif epoch < 60:
            stage = 1
        elif epoch < 90:
            stage = 2
        else:
            stage = 3
            
        if hasattr(trainer.datamodule, 'train_dataset'):
            trainer.datamodule.train_dataset.dataset.set_curriculum_stage(stage)
        
        if hasattr(trainer.datamodule, 'val_dataset'):
            trainer.datamodule.val_dataset.dataset.set_curriculum_stage(stage)

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
        d_model=512,
        nhead=4,
        num_layers=6
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_blind2",
        filename="predictor-{epoch:02d}-{val_acc:.3f}-{val_top2_acc:.3f}",
        save_top_k=10,
        monitor="val_acc",
        mode="max"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=10,
        mode="max"
    )

    curriculum_callback = CurriculumCallback()

    logger = TensorBoardLogger("tb_logs", name="key_recovery2")

    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        max_epochs=250,
        callbacks=[checkpoint_callback, early_stop_callback, curriculum_callback],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        accumulate_grad_batches=4
    )

    trainer.fit(model, datamodule=dm)
    #trainer.fit(model, datamodule=dm, ckpt_path="checkpoints_blind/predictor-epoch=138-val_acc=0.983-val_top2_acc=0.000.ckpt")

if __name__ == '__main__':
    main()