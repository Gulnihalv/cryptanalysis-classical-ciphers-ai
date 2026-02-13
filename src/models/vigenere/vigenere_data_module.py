import os
import sys
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
src_path = os.path.abspath(os.path.join(current_dir, '../../'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from generators.vigenere_generator import VigenereKeystreamGenerator

class VigenereDataModule(pl.LightningDataModule):
    def __init__(self, text_path, alphabet, seq_len=128, batch_size=64, 
                 min_key_len=3, max_key_len=12, val_split=0.1):
        super().__init__()
        self.text_path = text_path
        self.alphabet = alphabet
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.min_key_len = min_key_len
        self.max_key_len = max_key_len
        self.val_split = val_split
        
    def setup(self, stage=None):
        # Full dataset oluştur
        full_dataset = VigenereKeystreamGenerator(
            text_path=self.text_path,
            alphabet=self.alphabet,
            seq_len=self.seq_len,
            min_key_len=self.min_key_len,
            max_key_len=self.max_key_len
        )
        
        # Train/Val split
        total_len = len(full_dataset)
        val_len = int(total_len * self.val_split)
        train_len = total_len - val_len
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_len, val_len]
        )
        
        self.dataset = full_dataset
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )