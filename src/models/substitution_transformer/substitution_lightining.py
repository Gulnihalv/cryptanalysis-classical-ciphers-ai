import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from substitution_transformer import SubstitutionTransformer

class SubstitutionCipherSolverV8(pl.LightningModule):

    def __init__(
        self,
        vocab_size: int = 33,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        lr: float = 1e-4,
        warmup_steps: int = 4000,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SubstitutionTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
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

    def forward(self, src, tgt_input):
        return self.model(src, tgt_input)

    @property
    def teacher_forcing_ratio(self) -> float:
        epoch_shifted = max(0, self.current_epoch - 5)
        return max(0.0, 1.0 - epoch_shifted * 0.05)

    def training_step(self, batch, batch_idx):
        src, tgt_input, tgt_output = batch
        tf_ratio = self.teacher_forcing_ratio

        if torch.rand(1).item() < tf_ratio:
            # ── Teacher Forcing ─────────────────────────────────────────────────
            logits = self.model(src, tgt_input)  # [B, S, V]
        else:
            # ── Scheduled Sampling (free-run) ────────────────────────────────────
            # Run the decoder autoregressively; gradients flow through the whole
            # sequence so the model learns to recover from its own errors.
            B, S = src.shape
            device = src.device

            memory, memory_pad_mask = self.model._encode(src)

            generated = torch.full(
                (B, 1), self.model.SOS_TOKEN, dtype=torch.long, device=device
            )
            all_logits = []

            for t in range(S):
                T_cur = generated.size(1)
                tgt_emb = self.model.pos_encoding(self.model.tgt_embedding(generated))
                causal_mask = self.model._make_causal_mask(T_cur, device)
                tgt_pad_mask = generated == self.model.PAD_IDX

                dec_out = self.model.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    memory_key_padding_mask=memory_pad_mask,
                )
                step_logits = self.model.fc_out(dec_out[:, -1:, :])  # [B, 1, V]
                all_logits.append(step_logits)

                next_token = step_logits.detach().argmax(dim=-1)  # [B, 1]
                generated = torch.cat([generated, next_token], dim=1)

            logits = torch.cat(all_logits, dim=1)  # [B, S, V]

        loss = self.loss_fn(
            logits.reshape(-1, self.hparams.vocab_size), tgt_output.reshape(-1)
        )
        self.log("train_loss", loss, prog_bar=True)
        self.log("tf_ratio", tf_ratio, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt_input, tgt_output = batch

        # Teacher-forced loss & accuracy
        logits = self.model(src, tgt_input)
        loss = self.loss_fn(
            logits.reshape(-1, self.hparams.vocab_size), tgt_output.reshape(-1)
        )
        preds = logits.argmax(dim=-1)
        tf_acc = self.accuracy(preds.reshape(-1), tgt_output.reshape(-1))

        # Autoregressive (greedy) accuracy — the metric we really care about
        with torch.no_grad():
            generated = self.model.generate(src)
            gen_acc = self.accuracy(generated.reshape(-1), tgt_output.reshape(-1))

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_tf_acc", tf_acc, prog_bar=True, on_epoch=True)
        self.log("val_gen_acc", gen_acc, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
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