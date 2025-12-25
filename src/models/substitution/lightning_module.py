import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from models.substitution.substitution_lstm import SubstitutionLSTM

class SubstitutionCipherSolver(pl.LightningModule):
    def __init__(self, vocab_size=33, embed_dim=128, hidden_size=256, lr= 0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = SubstitutionLSTM(
            vocab_size=vocab_size, 
            embed_dim=embed_dim, 
            hidden_size=hidden_size
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=0) # PAD indexini ignore ediyoruz
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=0)
        self.lr = lr

    def forward(self, src, tgt_input):
        """burda alınan parametreler işleme sonrasında substitution modele devredilecek"""
        logits = self.model(src, tgt_input)
        return logits

    def training_step(self, batch, batch_idx):
        src, tgt_input, tgt_output = batch

        prob = 0.3 
        mask = torch.rand(tgt_input.shape, device=self.device) < prob

        mask[:, 0] = False 

        random_tokens = torch.randint(
            low=4, 
            high=self.hparams.vocab_size, 
            size=tgt_input.shape, 
            device=self.device
        )

        noisy_tgt_input = torch.where(mask, random_tokens, tgt_input)

        logits = self(src, noisy_tgt_input) # [batch, seq_len, vocab]
        loss = self.loss(logits.view(-1, self.hparams.vocab_size), tgt_output.view(-1)) # loss hesaplama (CrossEntropyLoss için flattening yapılıyor)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt_input, tgt_output = batch
        logits = self(src, tgt_input)
        loss = self.loss(logits.view(-1, self.hparams.vocab_size), tgt_output.view(-1))

        preds = torch.argmax(logits, dim=-1) #
        acc = self.accuracy(preds.view(-1), tgt_output.view(-1))

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.lr)
        schedular = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=2, 
            ),
            'monitor': 'val_loss', #val_loss izlenecek
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [schedular]
    
    # Mevcut sınıfının içine ekle
    def generate_beam(self, src, beam_width=3):
        """
         inference aşamasında kullanılacak
        """
        return self.model.generate_beam(src, beam_width)




