import os
import sys
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

# Hem proje kökünü (config için) hem de src klasörünü (generators/models için) path'e ekliyoruz
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
src_path = os.path.abspath(os.path.join(current_dir, '../../'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config.constants import S_CONFIG_V1
from generators.substitution_generator import SubstutionDataGenerator
from models.substitution.lightning_module import SubstitutionCipherSolver

def main():
    pl.seed_everything(42)
    full_dataset = SubstutionDataGenerator(
        text_path=S_CONFIG_V1['text_path'],
        alphabet=S_CONFIG_V1['alphabet'],
        seq_len=S_CONFIG_V1['seq_len']
    )

    # Train ve Validation olarak bölüyoruz
    # Datasetin __len__ metodu chunk vericek şekilde yazdığımızdan bu şekilde kullanılabiliyor
    total_count = len(full_dataset)
    val_count = int(total_count * S_CONFIG_V1['val_split'])
    train_count = total_count - val_count

    train_set, val_set = random_split(full_dataset, [train_count, val_count])

    print(f"Toplam Chunk: {total_count}")
    print(f"Train: {train_count} | Validation: {val_count}")

    # DataLoaders
    train_loader = DataLoader(
        train_set, 
        batch_size=S_CONFIG_V1['batch_size'], 
        shuffle=True, 
        num_workers=S_CONFIG_V1['num_workers'],
        pin_memory=True # GPU transferi için
    )

    val_loader = DataLoader(
        val_set, 
        batch_size=S_CONFIG_V1['batch_size'], 
        shuffle=False, 
        num_workers=S_CONFIG_V1['num_workers'],
        pin_memory=True
    )

    # Model
    vocab_size = len(full_dataset.full_alphabet)
    print(f"Alfabe Boyutu (Vocab Size): {vocab_size}") # özel karakterler dahil 33 olmalı

    model = SubstitutionCipherSolver(
        vocab_size=vocab_size,
        embed_dim=S_CONFIG_V1['embed_dim'],
        hidden_size=S_CONFIG_V1['hidden_size'],
        lr=S_CONFIG_V1['lr']
    )

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_v4',
        filename='substitution-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True
    )

    # Loss 3 epoch boyunca düşmezse eğitimi durdur
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        verbose=True
    )

    # TensorBoard Logları 
    logger = TensorBoardLogger("tb_logs", name="cipher_model")

    #RESUME_CHECKPOINT_PATH = "checkpoints/substitution-epoch=39-val_loss=0.22.ckpt"

    # Model Eğitimi
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        accelerator="auto",    # GPU varsa kullanır, yoksa CPU
        devices=1,             # 1 GPU kullan
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=10   # Her 10 batch'te bir logla
    )

    trainer.fit(model, train_loader, val_loader)
    # print(f"Eğitim '{RESUME_CHECKPOINT_PATH}' dosyasından devam ediyor...")
    # trainer.fit(model, train_loader, val_loader, ckpt_path=RESUME_CHECKPOINT_PATH)

    print(f"Eğitim tamamlandı! En iyi model şuraya kaydedildi: {checkpoint_callback.best_model_path}")

if __name__ == '__main__':
    main()