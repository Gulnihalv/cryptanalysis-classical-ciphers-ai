import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from models.substitution_v8.substitution_lstm_v8 import SubstitutionLSTM

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

        # Decay teacher forcing
        epoch_shifted = max(0, self.current_epoch - 5)
        teacher_forcing_ratio = max(0.0, 1.0 - (epoch_shifted * 0.05))
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

        if use_teacher_forcing:
            logits = self(src, tgt_input=tgt_input)
        else:
            # 1. Precompute static features once
            src_emb = self.model.embedding(src)
            
            # A. CNN
            cnn_in = src_emb.permute(0, 2, 1)
            cnn_out = self.model.local_cnn(cnn_in)
            
            # B. LSTM
            lstm_in = cnn_out.permute(0, 2, 1)
            cipher_context, _ = self.model.global_lstm(lstm_in) 

            global_freqs = self.model.compute_global_stats(src)
            freq_features = self.model.freq_encoder(global_freqs) # [B, 32]
            
            batch_size, seq_len = src.size()
            outputs = []
            
            current_input = tgt_input[:, 0:1] 
            h = None 
            
            for t in range(seq_len):
                # Slice inputs for this step
                cipher_char_emb = src_emb[:, t:t+1, :]
                tgt_emb = self.model.embedding(current_input)
                context = cipher_context[:, t:t+1, :]

                # We simply add a time dimension: [B, 32] -> [B, 1, 32]
                freq_t = freq_features.unsqueeze(1)
                
                # Combine (Sizes match the new LSTM definition)
                combined_input = torch.cat((cipher_char_emb, tgt_emb, context, freq_t), dim=2)
                
                # LSTM Step
                out, h = self.model.rnn(combined_input, h)
                
                # Predict
                logit = self.model.fc(out)
                outputs.append(logit)
                
                # Feed own prediction to next step
                predicted_token = torch.argmax(logit, dim=-1)
                current_input = predicted_token.detach() 

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