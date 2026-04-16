import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class VigenereBlindSolver(pl.LightningModule):
    def __init__(
        self,
        vocab_size=29,
        pad_idx=29,
        max_len=512,
        min_key_len=3,
        max_key_len=12,
        d_model=256,
        nhead=4,
        num_layers=4,
        lr=3e-4,
        key_len_loss_weight=0.3,
        plain_loss_weight=0.5,
        ic_hard_epochs=10,
        total_steps_override=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.total_steps_override = total_steps_override
        self.pad_idx     = pad_idx
        self.min_key_len = min_key_len
        self.max_key_len = max_key_len
        self.num_classes = max_key_len - min_key_len + 1

        self.char_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_idx)
        self.pos_emb  = nn.Embedding(max_len, d_model)

        self.cycle_embs = nn.ModuleList([
            nn.Embedding(k, d_model) for k in range(min_key_len, max_key_len + 1)
        ])

        self.ic_temperature = nn.Parameter(torch.tensor([20.0]))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,       # Post-LN → Pre-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        self.fc_keystream = nn.Linear(d_model, vocab_size)

        self.fc_plain = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

        self.key_len_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, self.num_classes),
        )

    def _compute_ic(self, src):
        """
        Her k için Index of Coincidence
        [B, num_classes]
        """
        B, L = src.shape
        scores = []
        for k in range(self.min_key_len, self.max_key_len + 1):
            shifted  = src[:, k:]
            original = src[:, :-k]
            pad_mask = (original == self.pad_idx) | (shifted == self.pad_idx)
            matches  = ((shifted == original) & ~pad_mask).float()
            valid    = (~pad_mask).float().sum(dim=1).clamp(min=1)
            confidence = (valid / L).clamp(max=1.0)
            scores.append((matches.sum(dim=1) / valid) * confidence)
        return torch.stack(scores, dim=1)   # [B, num_classes]

    def _get_length_weights(self, ic_scores):
        temp = self.ic_temperature.clamp(min=1.0)
        soft = F.softmax(ic_scores * temp, dim=-1)   # [B, num_classes]

        hard = torch.zeros_like(ic_scores)
        hard.scatter_(1, ic_scores.argmax(dim=1, keepdim=True), 1.0)

        if self.trainer is not None:
            alpha = min(self.current_epoch / self.hparams.ic_hard_epochs, 1.0)
        else:
            alpha = 1.0

        return (1.0 - alpha) * soft + alpha * hard   # [B, num_classes]

    def forward(self, src):
        batch_size, seq_len = src.shape
        device = src.device

        padding_mask = (src == self.pad_idx)

        ic_scores      = self._compute_ic(src)
        length_weights = self._get_length_weights(ic_scores)   # [B, num_classes]
        self._last_length_weights = length_weights.detach()

        positions      = torch.arange(seq_len, device=device).unsqueeze(0)   # [1, L]
        soft_cycle_emb = torch.zeros(batch_size, seq_len, self.char_emb.embedding_dim, device=device)

        for i, k in enumerate(range(self.min_key_len, self.max_key_len + 1)):
            cycle_pos_k = positions % k                                    # [1, L]
            emb_k       = self.cycle_embs[i](cycle_pos_k)                 # [1, L, d_model]
            weight_k    = length_weights[:, i].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            soft_cycle_emb += emb_k * weight_k

        x       = self.char_emb(src) + self.pos_emb(positions) + soft_cycle_emb
        encoded = self.transformer(x, src_key_padding_mask=padding_mask)   # [B, L, d_model]

        # Head çıkışları

        # 1. Keystream logitleri
        keystream_logits = self.fc_keystream(encoded)   # [B, L, vocab_size]

        # 2. Plaintext logitleri
        plain_logits = self.fc_plain(encoded)           # [B, L, vocab_size]

        # 3. Key length — padding-aware global avg pool → MLP
        token_mask     = (~padding_mask).float().unsqueeze(-1)             # [B, L, 1]
        valid_counts   = token_mask.sum(dim=1).clamp(min=1)               # [B, 1]
        pooled         = (encoded * token_mask).sum(dim=1) / valid_counts  # [B, d_model]
        key_len_logits = self.key_len_head(pooled)                         # [B, num_classes]

        return keystream_logits, plain_logits, key_len_logits

    def _shared_step(self, batch):
        src        = batch["src"]         # [B, L]  — şifreli metin
        tgt_key    = batch["key_target"]  # [B, L]  — keystream ground truth
        tgt_plain  = batch["tgt_plain"]   # [B, L]  — plaintext ground truth
        key_len_gt = batch["key_len"]     # [B]     — gerçek key uzunluğu

        keystream_logits, plain_logits, key_len_logits = self(src)

        # 1. Keystream kaybı
        keystream_loss = F.cross_entropy(
            keystream_logits.view(-1, keystream_logits.size(-1)),
            tgt_key.view(-1),
            ignore_index=self.pad_idx,
        )

        # 2. Plaintext kaybı
        plain_loss = F.cross_entropy(
            plain_logits.view(-1, plain_logits.size(-1)),
            tgt_plain.view(-1),
            ignore_index=self.pad_idx,
        )

        # 3. Key length kaybı
        key_len_class = key_len_gt - self.min_key_len   # 0-indexed
        key_len_loss  = F.cross_entropy(key_len_logits, key_len_class)

        # 4. Toplam ağırlıklı kayıp
        loss = (
            1.0 * keystream_loss
            + self.hparams.plain_loss_weight   * plain_loss
            + self.hparams.key_len_loss_weight * key_len_loss
        )

        pad_mask = (tgt_key != self.pad_idx)

        # Keystream accuracy
        ks_preds = keystream_logits.argmax(dim=-1)
        ks_acc   = ((ks_preds == tgt_key) & pad_mask).sum().float() / pad_mask.sum().float().clamp(min=1e-9)

        # fc_plain'den direkt plaintext accuracy
        pl_preds = plain_logits.argmax(dim=-1)
        pl_mask  = (tgt_plain != self.pad_idx)
        pl_acc   = ((pl_preds == tgt_plain) & pl_mask).sum().float() / pl_mask.sum().float().clamp(min=1e-9)

        # Keystream'den türetilen plaintext accuracy: P = (C - K) % mod
        derived_plain = (src - ks_preds) % self.hparams.vocab_size
        dp_acc = ((derived_plain == tgt_plain) & pl_mask).sum().float() / pl_mask.sum().float().clamp(min=1e-9)

        # Key length accuracy
        kl_preds = key_len_logits.argmax(dim=-1)
        kl_acc   = (kl_preds == key_len_class).float().mean()

        return loss, ks_acc, pl_acc, dp_acc, keystream_loss, plain_loss, key_len_loss, kl_acc

    def training_step(self, batch, batch_idx):
        loss, ks_acc, pl_acc, dp_acc, ks_loss, pl_loss, kl_loss, kl_acc = self._shared_step(batch)

        lw      = self._last_length_weights
        entropy = -(lw * (lw + 1e-9).log()).sum(dim=-1).mean()
        alpha   = min(self.current_epoch / self.hparams.ic_hard_epochs, 1.0)

        self.log("train_loss",     loss,    prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_ks_acc",   ks_acc,  prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_pl_acc",   pl_acc,  prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_dp_acc",   dp_acc,  prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_kl_acc",   kl_acc,  prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_ks_loss",  ks_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_pl_loss",  pl_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_kl_loss",  kl_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("length_entropy", entropy, prog_bar=True,  on_step=False, on_epoch=True)
        self.log("ic_alpha",       alpha,   prog_bar=True,  on_step=False, on_epoch=True)
        self.log("ic_temperature", self.ic_temperature.item(), prog_bar=False, on_step=False, on_epoch=True)

        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", current_lr, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, ks_acc, pl_acc, dp_acc, ks_loss, pl_loss, kl_loss, kl_acc = self._shared_step(batch)

        self.log("val_loss",    loss,    prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_ks_acc",  ks_acc,  prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_pl_acc",  pl_acc,  prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_dp_acc",  dp_acc,  prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_kl_acc",  kl_acc,  prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_kl_loss", kl_loss, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        total_steps = (
            self.total_steps_override 
            if self.total_steps_override 
            else self.trainer.estimated_stepping_batches
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy="cos",
                div_factor=25,
                final_div_factor=1e4,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def decrypt(self, cipher_indices, dataset):
        self.eval()

        if isinstance(cipher_indices, list):
            src = torch.tensor(cipher_indices, dtype=torch.long).unsqueeze(0).to(self.device)
        else:
            src = cipher_indices.unsqueeze(0).to(self.device) if cipher_indices.dim() == 1 else cipher_indices.to(self.device)

        with torch.no_grad():
            ks_logits, pl_logits, kl_logits = self(src)
            predicted_keystream = ks_logits.argmax(dim=-1)
            predicted_key_len   = kl_logits.argmax(dim=-1).item() + self.min_key_len

        plain_from_key = (src - predicted_keystream) % dataset.crypto_vocab_size
        plain_direct   = pl_logits.argmax(dim=-1)

        def decode(indices):
            result = []
            for p, s in zip(indices[0].cpu().tolist(), src[0].cpu().tolist()):
                if s == dataset.PAD_TOKEN_IDX:
                    break
                result.append(dataset.int2char.get(p, "?"))
            return "".join(result)

        return {
            "keystream_path":    decode(plain_from_key),
            "direct_path":       decode(plain_direct),
            "predicted_key_len": predicted_key_len,
        }