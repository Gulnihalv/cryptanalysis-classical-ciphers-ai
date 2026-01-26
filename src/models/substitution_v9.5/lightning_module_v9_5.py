import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from substitution_lstm_v9_5 import SubstitutionLSTM 

class SubstitutionCipherSolver(pl.LightningModule):
    def __init__(self, vocab_size=33, embed_dim=128, hidden_size=256, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = SubstitutionLSTM(
            vocab_size=vocab_size, 
            embed_dim=embed_dim, 
            hidden_size=hidden_size
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=0)
        self.lr = lr

    def forward(self, src, tgt_input=None):
        return self.model(src, tgt_input=tgt_input)

    def training_step(self, batch, batch_idx):
        src, tgt_input, tgt_output = batch

        # Teacher Forcing Decay
        epoch_shifted = max(0, self.current_epoch - 5)
        teacher_forcing_ratio = max(0.2, 1.0 - (epoch_shifted * 0.03)) 
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

        if use_teacher_forcing:
            # Teacher Forcing modunda forward zaten her şeyi hallediyor
            logits = self(src, tgt_input=tgt_input)
        else:
            
            # 1. Static Features (Döngü öncesi hesaplamalar)
            src_emb = self.model.embedding(src) # [Batch, Seq, Embed]
            
            cipher_context, _ = self.model.cipher_context_encoder(src_emb) 
            
            # B. İstatistikler
            # 1. Unigram
            unigram_freqs = self.model.compute_global_stats(src)
            unigram_feat = self.model.unigram_encoder(unigram_freqs) # [Batch, 32]
            
            # 2. N-Gram
            ngram_feat = self.model.ngram_encoder(src) # [Batch, 64]
            
            # İstatistikleri birleştir [Batch, 96]
            global_stats = torch.cat([unigram_feat, ngram_feat], dim=1)
            
            # Döngü Hazırlığı
            batch_size, seq_len = src.size()
            outputs = []
            
            current_input = tgt_input[:, 0:1] # İlk token (SOS)
            h = None 
            
            for t in range(seq_len):
                # O anki verileri hazırla
                cipher_char_emb = src_emb[:, t:t+1, :]     # [B, 1, Embed]
                tgt_emb = self.model.embedding(current_input) # [B, 1, Embed]
                context = cipher_context[:, t:t+1, :]      # [B, 1, Hidden]

                # Global istatistiği bu adım için boyutlandır
                global_stats_t = global_stats.unsqueeze(1) # [B, 1, 96]
                
                # Hepsini Birleştir (Toplam boyut 1120 olmalı)
                combined_input = torch.cat((cipher_char_emb, tgt_emb, context, global_stats_t), dim=2)
                
                # LSTM Step
                out, h = self.model.rnn(combined_input, h)
                
                # Predict
                logit = self.model.fc(out)
                outputs.append(logit)
                
                # Bir sonraki adım için kendi tahminini kullan
                predicted_token = torch.argmax(logit, dim=-1)
                current_input = predicted_token.detach() 

            logits = torch.cat(outputs, dim=1)

        loss = self.loss(logits.view(-1, self.hparams.vocab_size), tgt_output.view(-1))
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("tf_ratio", teacher_forcing_ratio, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt_input, tgt_output = batch

        # 1. Loss (Teacher Forcing ile)
        logits = self(src, tgt_input=tgt_input)
        loss = self.loss(logits.view(-1, self.hparams.vocab_size), tgt_output.view(-1))

        # 2. Accuracy (Teacher Forcing ile - Hocanın yardımıyla ne yaptı?)
        preds = torch.argmax(logits, dim=-1)
        tf_acc = self.accuracy(preds.view(-1), tgt_output.view(-1))

        # 3. Accuracy (Generation - Tek başına ne yaptı?)
        with torch.no_grad():
            generated = self.model.generate(src)
            gen_acc = self.accuracy(generated.view(-1), tgt_output.view(-1))

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_tf_acc", tf_acc, prog_bar=True, on_epoch=True)
        self.log("val_gen_acc", gen_acc, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_gen_acc',
                'interval': 'epoch',
                'frequency': 1
            }
        }