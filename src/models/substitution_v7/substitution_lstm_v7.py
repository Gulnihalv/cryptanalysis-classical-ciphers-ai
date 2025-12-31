import torch
import torch.nn as nn

class SubstitutionLSTM(nn.Module):
    def __init__(self, vocab_size=33, embed_dim=128, hidden_size=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.SOS_TOKEN = 1
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Global istatikleri öğrenmek için
        self.cipher_context_encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size // 2, # Çift yönlü olduğundan 
            num_layers=1,
            batch_first=True,
            bidirectional=True  # Tüm cipher'ı görüyor
        )
        
        # Ana RNN girdi boyutu artıyor
        self.rnn = nn.LSTM(
            input_size=embed_dim * 2 + hidden_size,  # cipher_char + prev_char + global_context 128 + 128 + 256 = 512 # v7.2de embeddim 256 hidden 512
            hidden_size=hidden_size,
            num_layers=3,
            dropout=0.2, #overfitting önlemek için
            batch_first=True,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, tgt_input=None):      
        # 1. Global cipher context (tüm cipher'ı bir kez işle)
        src_emb = self.embedding(src) #  (Batch, Seq, embedding = 128)
        cipher_context, _ = self.cipher_context_encoder(src_emb)  # [Batch, Seq_len, Hidden]
        
        # Önceki gerçek plain char
        tgt_emb = self.embedding(tgt_input)
        
        # girdi: cipher_char + prev_plain_char + cipher_context 128 + 128 + 256
        combined_input = torch.cat((src_emb, tgt_emb, cipher_context), dim=2)
        
        outputs, _ = self.rnn(combined_input)
        logits = self.fc(outputs) #[B, S, V] (Batch, Seq, 33)
        
        return logits

    def generate(self, src):
        batch_size, seq_len = src.size()
        device = src.device
        
        # Global cipher context'i bir kez hesapla
        src_emb = self.embedding(src) # [B, S, E]
        cipher_context, _ = self.cipher_context_encoder(src_emb) # [B, S, H]
        
        current_input_token = torch.full((batch_size, 1), self.SOS_TOKEN, dtype=torch.long, device=device)
        h = None
        predictions = []
        
        for t in range(seq_len):
            cipher_char_emb = src_emb[:, t:t+1, :] # [B, 1, E] t deki şifreli karakter
            tgt_emb = self.embedding(current_input_token) # [B, 1, E] bir önceki tahmin, trainde burası için belirli oranda gerçek value geliyor
            context = cipher_context[:, t:t+1, :] # [B, 1, H] contextteki o anki parça
            
            combined_input = torch.cat((cipher_char_emb, tgt_emb, context), dim=2)
            output, h = self.rnn(combined_input, h) # hidden state sonraki adıma aktarılıyor
            
            logit = self.fc(output) # [B, 1, V] tahmin
            predicted_token = torch.argmax(logit, dim=-1) # En yüksek olasılıklı harfi seçiyoruz
            
            predictions.append(predicted_token)
            current_input_token = predicted_token

        return torch.cat(predictions, dim=1)