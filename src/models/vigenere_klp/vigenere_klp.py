import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class VigenereKeyLengthCNN(pl.LightningModule):
    def __init__(
        self,
        vocab_size=29,
        pad_token_id=29,
        d_model=128,
        min_key_len=3,
        max_key_len=12,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pad_idx    = pad_token_id
        self.min_key_len = min_key_len
        self.max_key_len = max_key_len
        self.num_classes = max_key_len - min_key_len + 1

        self.char_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_token_id)

        # MULTI-SCALE CNN
        self.kernel_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, 64, kernel_size=k, padding="same"),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
            for k in self.kernel_sizes
        ])

        num_conv = len(self.kernel_sizes)
        input_dim = (64 * num_conv * 2) + self.num_classes

        # SINIFLANDIRICI
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.num_classes),
        )

        # HARMONİK SOFT LABEL MATRİSİ 
        # key_len=4 ise, key_len=8 de kısmen doğru sayılıyor katı olduğu için
        harmonic_matrix = torch.zeros(self.num_classes, self.num_classes)
        for target_val in range(min_key_len, max_key_len + 1):
            target_idx = target_val - min_key_len
            harmonic_matrix[target_idx, target_idx] = 1.0
            carpan = 2
            while target_val * carpan <= max_key_len:
                kat_val = target_val * carpan
                kat_idx = kat_val - min_key_len
                harmonic_matrix[target_idx, kat_idx] = 0.15 / carpan
                carpan += 1
            harmonic_matrix[target_idx] = (
                harmonic_matrix[target_idx] / harmonic_matrix[target_idx].sum()
            )

        self.register_buffer("harmonic_matrix", harmonic_matrix)

    def forward(self, src):
        pad_mask = (src == self.pad_idx).unsqueeze(1)  # [B, 1, L]
        x = self.char_emb(src).transpose(1, 2)         # [B, D, L]
        pooled = []
        for conv in self.convs:
            out = conv(x)                              # [B, 64, L]

            # MAX POOLING (En güçlü sinyali yakalamak için)
            out_max = out.masked_fill(pad_mask, float("-inf"))
            max_p = F.adaptive_max_pool1d(out_max, 1).squeeze(-1)  # [B, 64]

            # AVERAGE POOLING (Tekrar sıklığını yakalamak için)
            out_avg = out.masked_fill(pad_mask, 0.0)
            valid_len = (~pad_mask).sum(dim=2).clamp(min=1) # Sadece gerçek harfleri sayıyoruz
            avg_p = out_avg.sum(dim=2) / valid_len           # [B, 64]
            
            pooled.append(max_p)
            pooled.append(avg_p)

        ic_scores = self._compute_ic(src) 
        combined = torch.cat([*pooled, ic_scores], dim=-1)
        return self.classifier(combined)
    
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

    def _shared_step(self, batch):
        src        = batch["src"]
        key_len    = batch["key_len"]

        logits         = self(src)
        target_indices = key_len - self.min_key_len          # class index'e çevir
        soft_targets   = self.harmonic_matrix[target_indices] # [B, num_classes]

        loss = F.cross_entropy(logits, soft_targets)

        preds = logits.argmax(dim=-1)
        acc   = (preds == target_indices).float().mean()

        # Top-2 accuracy: modelin ilk 2 tahmini içinde doğru key_len var mı bakıyoruz
        top2_preds      = logits.topk(2, dim=-1).indices     # [B, 2]
        target_expanded = target_indices.unsqueeze(-1).expand_as(top2_preds)
        top2_acc        = (top2_preds == target_expanded).any(dim=-1).float().mean()

        return loss, acc, top2_acc

    def training_step(self, batch, batch_idx):
        loss, acc, top2_acc = self._shared_step(batch)
        self.log("train_loss",     loss,     prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_acc",      acc,      prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_top2_acc", top2_acc, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, top2_acc = self._shared_step(batch)
        self.log("val_loss",     loss,     prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val_acc",      acc,      prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val_top2_acc", top2_acc, prog_bar=True,  on_step=False, on_epoch=True)
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
                anneal_strategy="cos",
                div_factor=25,
                final_div_factor=1e4,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def predict_key_length(self, src: torch.Tensor, top_k: int = 1):
        """
        Inference yardımcısı.
        src: [L] veya [B, L]
        return: tahmin edilen key_len değeri (int) veya listesi
        """
        self.eval()
        with torch.no_grad():
            if src.dim() == 1:
                src = src.unsqueeze(0)   # [1, L]
            src = src.to(self.device)
            logits = self(src)           # [B, num_classes]
            if top_k == 1:
                preds = logits.argmax(dim=-1) + self.min_key_len
                return preds.squeeze().item() if preds.numel() == 1 else preds.tolist()
            else:
                top_indices = logits.topk(top_k, dim=-1).indices + self.min_key_len
                return top_indices.tolist()