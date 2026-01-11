import os
import sys
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
src_path = os.path.abspath(os.path.join(current_dir, '../../'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config.constants import S_CONFIG_V8
from generators.substitution_generator import SubstutionDataGenerator
from models.substitution_v8.lightning_module_v8 import SubstitutionCipherSolverV8


def main():
    pl.seed_everything(42)
    full_dataset = SubstutionDataGenerator(
        text_path=S_CONFIG_V8['text_path'],
        alphabet=S_CONFIG_V8['alphabet'],
        seq_len=S_CONFIG_V8['seq_len']
    )

    # Train, Validation
    total_count = len(full_dataset)
    val_count = int(total_count * S_CONFIG_V8['val_split'])
    train_count = total_count - val_count

    train_set, val_set = random_split(full_dataset, [train_count, val_count])

    print(f"Toplam Chunk: {total_count}")
    print(f"Train: {train_count} | Validation: {val_count}")

    # DataLoaders
    train_loader = DataLoader(
        train_set, 
        batch_size=S_CONFIG_V8['batch_size'], 
        shuffle=True, 
        num_workers=S_CONFIG_V8['num_workers'],
        persistent_workers=True,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_set, 
        batch_size=S_CONFIG_V8['batch_size'], 
        shuffle=False, 
        num_workers=S_CONFIG_V8['num_workers'],
        persistent_workers=True,
        pin_memory=False
    )

    # Model
    vocab_size = len(full_dataset.full_alphabet)
    print(f"Alfabe Boyutu (Vocab Size): {vocab_size}") # özel karakterler dahil 33 olmalı

    model = SubstitutionCipherSolverV8(
        vocab_size=vocab_size,
        embed_dim=S_CONFIG_V8['embed_dim'],
        hidden_size=S_CONFIG_V8['hidden_size'],
        lr=S_CONFIG_V8['lr']
    )

    # Callback
    # callbacklerde gerçek sonuca göre devam etmek için val_gen_acc kullanıldı val_loss yerine
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_v8',
        filename='substitution-{epoch:02d}-{val_gen_acc:.3f}',
        monitor='val_gen_acc',
        mode='max',
        save_top_k=3,
        verbose=True
    )

    early_stopping = EarlyStopping(
        monitor='val_gen_acc',
        patience=10,
        mode='max',
        verbose=True,
        min_delta=0.002 
    )

    # TensorBoard Logları 
    logger = TensorBoardLogger("tb_logs", name="cipher_model_v8")

    # Model Eğitimi
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        accelerator="auto",    # GPU varsa kullanır, yoksa CPU
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"Eğitim tamamlandı! En iyi model şuraya kaydedildi: {checkpoint_callback.best_model_path}")

if __name__ == '__main__':
    main()