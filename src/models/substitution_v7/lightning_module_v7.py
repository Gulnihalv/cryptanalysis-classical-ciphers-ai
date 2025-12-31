import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from models.substitution_v7.substitution_lstm_v7 import SubstitutionLSTM

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

        teacher_forcing_ratio = max(0.0, 1.0 - (self.current_epoch * 0.05)) # Daha yavaş düşsün ama 0'a insin
        
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

        if use_teacher_forcing:
            # Klasik hızlı yol (Teacher Forcing)
            logits = self(src, tgt_input=tgt_input)
        else:
            # --- YENİ YÖNTEM: Autoregressive Rollout ---
            # Burada modelin kendi ürettiği çıktıyı (hatalı olsa bile) 
            # bir sonraki adıma besleyerek zincirleme etkiyi öğretiyoruz.
            
            # Encoder kısmını bir kere çalıştır
            src_emb = self.model.embedding(src)
            cipher_context, _ = self.model.cipher_context_encoder(src_emb)
            
            batch_size, seq_len = src.size()
            outputs = []
            
            # İlk girdi SOS token
            current_input = tgt_input[:, 0:1] # [B, 1]
            h = None # Hidden state
            
            for t in range(seq_len):
                # 1. Adım verilerini hazırla
                cipher_char_emb = src_emb[:, t:t+1, :]
                tgt_emb = self.model.embedding(current_input)
                context = cipher_context[:, t:t+1, :]
                
                # 2. Birleştir
                combined_input = torch.cat((cipher_char_emb, tgt_emb, context), dim=2)
                
                # 3. LSTM Adımı
                out, h = self.model.rnn(combined_input, h)
                
                # 4. Tahmin
                logit = self.model.fc(out) # [B, 1, V]
                outputs.append(logit)
                
                # 5. Bir sonraki adımın girdisi ne olacak?
                # İşte burası kilit nokta: Modelin kendi seçimi!
                predicted_token = torch.argmax(logit, dim=-1) # [B, 1]
                
                # prediction'ı gradient'ten kopar (yoksa hata alırsın)
                current_input = predicted_token.detach() 

            # Tüm adımları birleştir: [B, S, V]
            logits = torch.cat(outputs, dim=1)

        loss = self.loss(logits.view(-1, self.hparams.vocab_size), tgt_output.view(-1))
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("tf_ratio", teacher_forcing_ratio, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt_input, tgt_output = batch

        # Teacher forcing ile loss
        logits = self(src, tgt_input=tgt_input)
        loss = self.loss(logits.view(-1, self.hparams.vocab_size), tgt_output.view(-1))

        # Teacher forcing accuracy
        preds = torch.argmax(logits, dim=-1)
        tf_acc = self.accuracy(preds.view(-1), tgt_output.view(-1))

        # Gerçek inference accuracy (burası her batchte çalışıyor)
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
            optimizer, mode='max', factor=0.5, patience=5
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