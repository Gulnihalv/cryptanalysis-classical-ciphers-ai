import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from substitution_transformer import SubstitutionEncoderModel


class SubstitutionCipherSolverV9(pl.LightningModule):

    def __init__(
        self,
        vocab_size: int = 33,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        warmup_steps: int = 400,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SubstitutionEncoderModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            max_len=max_len,
        )

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=label_smoothing
        )
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=vocab_size, ignore_index=0
        )

    def forward(self, src):
        return self.model(src)

    # ── Training ─────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        src, _tgt_input, tgt_output = batch   # tgt_input artık kullanılmıyor

        logits = self.model(src)               # [B, S, V]
        loss = self.loss_fn(
            logits.reshape(-1, self.hparams.vocab_size),
            tgt_output.reshape(-1),
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # ── Validation ───────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        src, _tgt_input, tgt_output = batch

        logits = self.model(src)
        loss = self.loss_fn(
            logits.reshape(-1, self.hparams.vocab_size),
            tgt_output.reshape(-1),
        )

        # Direkt tahmin accuracy (= "gen_acc", exposure bias yok)
        preds = logits.argmax(dim=-1)
        acc = self.accuracy(preds.reshape(-1), tgt_output.reshape(-1))

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc",  acc,  prog_bar=True, on_epoch=True)
        return loss

    # ── Optimizer: Noam Schedule ─────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1.0,            # Noam schedule LR'yi kendisi yönetir
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=1e-2,
        )

        d_model = self.hparams.embed_dim
        warmup = self.hparams.warmup_steps

        def noam_lambda(step: int) -> float:
            step = max(1, step)
            return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }