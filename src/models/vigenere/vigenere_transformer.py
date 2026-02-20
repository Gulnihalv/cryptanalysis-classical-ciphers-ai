import torch
import torch.nn as nn
import pytorch_lightning as pl

class VigenereLightningModule(pl.LightningModule):
    def __init__(self, vocab_size=31, max_len=512, max_key_len=12, d_model=128, nhead=4, num_layers=2, lr=1e-3, pad_token_id=29):
        super().__init__()
        self.save_hyperparameters()

        self.char_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.cycle_emb = nn.Embedding(max_key_len + 1, d_model)
        self.key_len_emb = nn.Embedding(max_key_len + 1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, norm_first=True, dim_feedforward=d_model*4, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_key = nn.Linear(d_model, vocab_size)
        
        # Loss (Padding ve Boşlukları yok sayıyoruz)
        self.pad_idx = pad_token_id 
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def forward(self, src, cycle_pos, key_len):
        batch_size, seq_len = src.shape
        device = src.device

        # Position indices
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        char_emb = self.char_emb(src)  # [B, L, D]
        pos_emb = self.pos_emb(pos)    # [B, L, D]
        cycle_emb = self.cycle_emb(cycle_pos)  # [B, L, D]

        # Key length embedding (global)
        key_emb = self.key_len_emb(key_len).unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, D]

        # Combine
        x = char_emb + pos_emb + cycle_emb + key_emb

        encoded = self.transformer(x)
        return self.fc_key(encoded)

    def training_step(self, batch, batch_idx):
        src = batch["src"]
        key_target = batch["key_target"]
        cycle_pos = batch["cycle_pos"]
        key_len = batch["key_len"]

        logits = self(src, cycle_pos, key_len)
        
        # Loss [Batch*Seq, Vocab]
        loss = self.criterion(logits.view(-1, self.hparams.vocab_size), key_target.view(-1))
        
        preds = logits.argmax(dim=-1)
        mask = (key_target != self.pad_idx)
        correct = (preds == key_target) & mask
        acc = correct.sum().float() / mask.sum().float()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src = batch["src"]
        key_target = batch["key_target"]
        cycle_pos = batch["cycle_pos"]
        key_len = batch["key_len"]

        logits = self(src, cycle_pos, key_len)
        loss = self.criterion(logits.view(-1, self.hparams.vocab_size), key_target.view(-1))
        
        # Doğruluk (Accuracy)
        preds = logits.argmax(dim=-1)
        mask = (key_target != self.pad_idx)
        correct = (preds == key_target) & mask
        acc = correct.sum().float() / mask.sum().float()
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # AdamW
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=0.01,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Warmup + Cosine Decay Scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,  # İlk %10 warmup
                anneal_strategy='cos',  # Cosine decay
                div_factor=25,  # Başlangıç LR = max_lr / 25
                final_div_factor=1e4  # Son LR = max_lr / 10000
            ),
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]
    
    def decrypt(self, cipher_indices, key_len_val, dataset):
        """Şifreli metni çözer"""
        self.eval()
        
        # Liste ise tensor'e çevir
        if isinstance(cipher_indices, list):
            src = torch.tensor(cipher_indices, dtype=torch.long).unsqueeze(0).to(self.device)
        else:
            src = cipher_indices.unsqueeze(0).to(self.device) if cipher_indices.dim() == 1 else cipher_indices.to(self.device)
        
        # Cycle position hesapla
        cycle_pos_list = []
        current_k_ptr = 0
        src_list = src[0].cpu().tolist()
        
        for char_idx in src_list:
            if char_idx == dataset.SPACE_TOKEN_IDX or char_idx == dataset.PAD_TOKEN_IDX:
                cycle_pos_list.append(key_len_val)
            else:
                cycle_pos_list.append(current_k_ptr % key_len_val)
                current_k_ptr += 1
        
        cycle_pos = torch.tensor(cycle_pos_list, dtype=torch.long).unsqueeze(0).to(self.device)
        key_len_tensor = torch.tensor([key_len_val], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits = self(src, cycle_pos, key_len_tensor)
            predicted_key = logits.argmax(dim=-1)
        
        # Decrypt: P = (C - K) % Mod
        plain_indices = (src - predicted_key) % dataset.crypto_vocab_size
        
        # String'e çevir
        decoded_text = []
        plain_list = plain_indices[0].cpu().tolist()
        
        for p_idx, s_idx in zip(plain_list, src_list):
            if s_idx == dataset.PAD_TOKEN_IDX:
                break
            if s_idx == dataset.SPACE_TOKEN_IDX:
                decoded_text.append(" ")
            elif p_idx in dataset.int2char:
                decoded_text.append(dataset.int2char[p_idx])
            else:
                decoded_text.append("?")
        
        return "".join(decoded_text)