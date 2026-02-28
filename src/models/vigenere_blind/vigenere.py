import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class VigenereBlindSolver(pl.LightningModule):
    def __init__(self, vocab_size=29, pad_idx=29, max_len=128, min_key_len=3, max_key_len=12, d_model=256, nhead=8, num_layers=6, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.pad_idx = pad_idx
        self.min_key_len = min_key_len
        self.max_key_len = max_key_len
        self.num_classes = max_key_len - min_key_len + 1
        self.char_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)
        #self.ic_temperature = nn.Parameter(torch.tensor([2.0]))

        self.cycle_embs = nn.ModuleList([
            nn.Embedding(k, d_model) for k in range(min_key_len, max_key_len + 1)
        ])

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _compute_ic(self, src):
        B, L = src.shape
        scores = []
        for k in range(self.min_key_len, self.max_key_len + 1):
            shifted  = src[:, k:]      # [B, L-k]
            original = src[:, :-k]     # [B, L-k]
            pad_mask = (original == self.pad_idx) | (shifted == self.pad_idx)
            matches  = ((shifted == original) & ~pad_mask).float()
            valid    = (~pad_mask).float().sum(dim=1).clamp(min=1)
            scores.append((matches.sum(dim=1) / valid))  # [B]
        return torch.stack(scores, dim=1)

    def forward(self, src):
        batch_size, seq_len = src.shape
        device = src.device
        ic_scores = self._compute_ic(src)
        #scaled_ic = ic_scores * self.ic_temperature.clamp(min=0.1)
        length_weights = F.softmax(ic_scores * 200, dim=-1)

        self._last_length_weights = length_weights.detach()

        soft_cycle_emb = torch.zeros(batch_size, seq_len, self.char_emb.embedding_dim, device=device)
        positions = torch.arange(seq_len, device=device).unsqueeze(0) # [1, Seq]

        for i, k in enumerate(range(self.min_key_len, self.max_key_len + 1)):
            cycle_pos_k = positions % k  # [1, Seq]
            emb_k = self.cycle_embs[i](cycle_pos_k) # [1, Seq, d_model]
            weight_k = length_weights[:, i].unsqueeze(1).unsqueeze(2) # [B, 1, 1]
            soft_cycle_emb += emb_k * weight_k

        x = self.char_emb(src) + self.pos_emb(positions) + soft_cycle_emb
        encoded = self.transformer(x)

        return self.fc_out(encoded)
    
    def _shared_step(self, batch):
        src = batch["src"]               # [B, L]
        tgt = batch["key_target"]        # [B, L] - Modelin bulmaya çalıştığı anahtar akışı
        
        # 1. İleri Besleme (Forward)
        logits = self(src)               # [B, L, vocab_size]
        
        # 2. Loss Hesaplama 
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            tgt.view(-1), 
            ignore_index=self.pad_idx
        )
        
        # 3. Doğruluk (Accuracy) Hesaplama
        preds = logits.argmax(dim=-1)                 # [B, L]
        pad_mask = (tgt != self.pad_idx)              # Gerçek harflerin/anahtarların olduğu yerler
        
        # Sadece gerçek anahtar değerlerini karşılaştır
        correct = (preds == tgt) & pad_mask
        acc = correct.sum().float() / pad_mask.sum().float().clamp(min=1e-9)
        
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)

        lw = self._last_length_weights
        entropy = -(lw * (lw + 1e-9).log()).sum(dim=-1).mean()
        self.log("length_entropy", entropy, prog_bar=True, on_step=False, on_epoch=True) # loss'a katmıyoruz sadece loglama için

        #self.log("ic_temp_value", self.ic_temperature.item(), prog_bar=True, on_step=False, on_epoch=True)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1, 
                anneal_strategy="cos", # Cosine decay ile yumuşak iniş
                div_factor=25,
                final_div_factor=1e4,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]