import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class VigenereBlindSolver(pl.LightningModule):
    WRONG_KEY_PROB  = 0.20
    VALIDITY_WEIGHT = 0.30

    def __init__(
        self,
        vocab_size=29,
        pad_idx=29,
        max_len=512,
        min_key_len=3,
        max_key_len=12,
        d_model=256,
        nhead=8,
        num_layers=6,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pad_idx     = pad_idx
        self.min_key_len = min_key_len
        self.max_key_len = max_key_len

        # --- Embedding katmanları ---
        self.char_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_idx)
        self.pos_emb  = nn.Embedding(max_len, d_model)

        # Her olası key_len için ayrı cycle embedding tablosu
        # cycle_embs[i] → key_len = min_key_len + i periyodunun pozisyon tablosu
        self.cycle_embs = nn.ModuleList([
            nn.Embedding(k, d_model)
            for k in range(min_key_len, max_key_len + 1)
        ])

        # --- Transformer ---
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Output head'ler ---
        self.fc_out       = nn.Linear(d_model, vocab_size)   # plaintext tahmini
        self.validity_head = nn.Linear(d_model, 1)           # bu key_len doğru mu?

    def _get_cycle_emb(self, seq_len: int, key_len: torch.Tensor, device) -> torch.Tensor:
        B = key_len.size(0)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, L]

        cycle_emb = torch.zeros(B, seq_len, self.hparams.d_model, device=device)

        for i, k in enumerate(range(self.min_key_len, self.max_key_len + 1)):
            mask = (key_len == k)           # [B] — bu key_len'e sahip örnekler
            if not mask.any():
                continue
            cycle_pos = positions % k       # [1, L]
            emb = self.cycle_embs[i](cycle_pos)             # [1, L, d_model]
            cycle_emb[mask] = emb.expand(mask.sum(), -1, -1)

        return cycle_emb
    
    def forward(self, src: torch.Tensor, key_len: torch.Tensor):
        B, L   = src.shape
        device = src.device

        positions  = torch.arange(L, device=device).unsqueeze(0)   # [1, L]
        cycle_emb  = self._get_cycle_emb(L, key_len, device)       # [B, L, d_model]

        # Padding mask — transformer'ın pad tokenları görmezden gelmesi için
        pad_mask = (src == self.pad_idx)  # [B, L]  True = ignore

        x = self.char_emb(src) + self.pos_emb(positions) + cycle_emb
        encoded = self.transformer(x, src_key_padding_mask=pad_mask)  # [B, L, d_model]

        plaintext_logits = self.fc_out(encoded)                        # [B, L, vocab_size]

        # Sequence pooling (pad'leri hariç tut) → validity skoru
        non_pad   = (~pad_mask).unsqueeze(-1).float()                  # [B, L, 1]
        pooled    = (encoded * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1)  # [B, d_model]
        validity_logit = self.validity_head(pooled).squeeze(-1)        # [B]

        return plaintext_logits, validity_logit

    def _shared_step(self, batch, inject_wrong_key: bool = False):
        src       = batch["src"]        # [B, L]
        tgt_plain = batch["tgt_plain"]  # [B, L] — hedef: düz metin
        true_key_len = batch["key_len"] # [B]

        B = src.size(0)
        device = src.device

        # --- Yanlış key_len enjeksiyonu (sadece training) ---
        if inject_wrong_key:
            # Her örnek için bağımsız karar ver
            wrong_mask = torch.rand(B, device=device) < self.WRONG_KEY_PROB  # [B]

            if wrong_mask.any():
                # Yanlış key_len üret: true_key_len'den farklı rastgele bir değer
                rand_keys = torch.randint(
                    self.min_key_len, self.max_key_len + 1, (B,), device=device
                )
                # Eğer rastgele değer true ile aynıysa kaydır
                same = rand_keys == true_key_len
                rand_keys[same] = (rand_keys[same] - self.min_key_len + 1) % \
                                   (self.max_key_len - self.min_key_len + 1) + self.min_key_len

                key_len_input = torch.where(wrong_mask, rand_keys, true_key_len)
            else:
                key_len_input = true_key_len

            # Validity label: doğru key_len verildi mi?
            validity_label = (~wrong_mask).float()  # [B]
        else:
            key_len_input  = true_key_len
            validity_label = torch.ones(B, device=device)

        # --- Forward ---
        plaintext_logits, validity_logit = self(src, key_len_input)  # [B,L,V], [B]

        # --- Plaintext CE Loss ---
        # Yanlış key_len verilen örneklerde plaintext loss hesaplama
        # (model çöp üretmeli, ama biz onu plaintext'e zorlamayalım)
        if inject_wrong_key and wrong_mask.any():
            correct_mask = ~wrong_mask  # [B]
            if correct_mask.any():
                ce_loss = F.cross_entropy(
                    plaintext_logits[correct_mask].reshape(-1, plaintext_logits.size(-1)),
                    tgt_plain[correct_mask].reshape(-1),
                    ignore_index=self.pad_idx,
                )
            else:
                ce_loss = torch.tensor(0.0, device=device)
        else:
            ce_loss = F.cross_entropy(
                plaintext_logits.reshape(-1, plaintext_logits.size(-1)),
                tgt_plain.reshape(-1),
                ignore_index=self.pad_idx,
            )

        # --- Validity BCE Loss ---
        bce_loss = F.binary_cross_entropy_with_logits(validity_logit, validity_label)

        # --- Toplam Loss ---
        loss = ce_loss + self.VALIDITY_WEIGHT * bce_loss

        # --- Accuracy (sadece doğru key_len örneklerinde) ---
        with torch.no_grad():
            if inject_wrong_key:
                correct_mask = ~wrong_mask
            else:
                correct_mask = torch.ones(B, dtype=torch.bool, device=device)

            if correct_mask.any():
                preds    = plaintext_logits[correct_mask].argmax(dim=-1)   # [B', L]
                tgt_c    = tgt_plain[correct_mask]
                pad_mask = (tgt_c != self.pad_idx)
                correct  = (preds == tgt_c) & pad_mask
                acc = correct.sum().float() / pad_mask.sum().float().clamp(min=1e-9)
            else:
                acc = torch.tensor(0.0, device=device)

            # Validity accuracy
            val_pred = (validity_logit > 0).float()
            val_acc  = (val_pred == validity_label).float().mean()

        return loss, ce_loss, bce_loss, acc, val_acc

    def training_step(self, batch, batch_idx):
        loss, ce_loss, bce_loss, acc, val_acc = self._shared_step(
            batch, inject_wrong_key=True
        )
        self.log("train_loss",     loss,     prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_ce_loss",  ce_loss,  prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_bce_loss", bce_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_acc",      acc,      prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_val_acc",  val_acc,  prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validasyonda yanlış key_len yok — gerçek performansı ölç
        loss, ce_loss, bce_loss, acc, val_acc = self._shared_step(
            batch, inject_wrong_key=False
        )
        self.log("val_loss",     loss,    prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_ce_loss",  ce_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_acc",      acc,     prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_val_acc",  val_acc, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def decode(self, ciphertext_indices: torch.Tensor) -> dict:
        self.eval()
        device = ciphertext_indices.device

        num_keys = self.max_key_len - self.min_key_len + 1
        L = ciphertext_indices.size(0)

        # [1, L] → [num_keys, L] kopyala
        src = ciphertext_indices.unsqueeze(0).expand(num_keys, -1)  # [K, L]

        # key_len = [3, 4, ..., 12]
        key_lens = torch.arange(
            self.min_key_len, self.max_key_len + 1, device=device
        )  # [K]

        # Tek forward pass — K örnek paralel
        plaintext_logits, validity_logits = self(src, key_lens)  # [K,L,V], [K]

        validity_scores = torch.sigmoid(validity_logits)  # [K] — 0..1
        best_idx        = validity_scores.argmax().item()
        best_key_len    = self.min_key_len + best_idx

        all_plaintexts = plaintext_logits.argmax(dim=-1)  # [K, L]
        best_plaintext = all_plaintexts[best_idx]         # [L]

        return {
            "best_key_len"   : best_key_len,
            "plaintext"      : best_plaintext,
            "validity_scores": validity_scores,
            "all_plaintexts" : all_plaintexts,
        }

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