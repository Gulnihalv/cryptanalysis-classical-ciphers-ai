import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalNGramEncoder(nn.Module):
    """
    Metindeki harf ikililerini (Bigram) ve üçlülerini (Trigram) sayarak
    dilin 'parmak izini' çıkaran modül. CNN yerine bu kullanılır.
    """
    def __init__(self, vocab_size=33, output_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        
        # 1. Bigram Encoder (İkililer)
        # Giriş: 33x33 = 1089 olası kombinasyon
        self.bigram_encoder = nn.Sequential(
            nn.Linear(vocab_size * vocab_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32)
        )
        
        # 2. Trigram Encoder (Üçlüler) - Hash tabanlı sıkıştırma
        # 33^3 çok büyük olduğu için 512 bucket içine hashliyoruz
        self.num_buckets = 512
        self.trigram_encoder = nn.Sequential(
            nn.Linear(self.num_buckets, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32)
        )
        
    def compute_bigram_freqs(self, src):
        """
        Global bigram frekanslarını hesaplar.
        Loop yerine vektörize işlem yapmaya çalışırız ama basitlik için sliding window mantığı:
        """
        batch_size, seq_len = src.size()
        device = src.device
        
        # [Batch, 1089]
        bigram_counts = torch.zeros(batch_size, self.vocab_size * self.vocab_size, device=device)
        
        # Sliding window (Vektörize edilmiş)
        # t anındaki harf ve t+1 anındaki harf
        current = src[:, :-1]  # [B, L-1]
        next_char = src[:, 1:] # [B, L-1]
        
        # İndex hesapla: c1 * 33 + c2
        indices = current * self.vocab_size + next_char
        
        # Mask: 0 (padding) olanları sayma
        mask = (current > 0) & (next_char > 0)
        
        # Scatter add ile sayım (Loopsuz hızlı sayım)
        # indices[mask] -> geçerli ikililer
        # bigram_counts'a ekle
        
        for b in range(batch_size):
            valid_indices = indices[b][mask[b]]
            if len(valid_indices) > 0:
                # bincount ile hızlı sayım
                counts = torch.bincount(valid_indices, minlength=self.vocab_size**2).float()
                bigram_counts[b] = counts

        # Normalize (Sequence length'e böl)
        seq_lengths = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        return bigram_counts / seq_lengths
    
    def compute_trigram_freqs(self, src):
        """
        Global trigram frekansları (Hash collision mantığı ile)
        """
        batch_size, seq_len = src.size()
        device = src.device
        
        trigram_counts = torch.zeros(batch_size, self.num_buckets, device=device)
        
        c1 = src[:, :-2]
        c2 = src[:, 1:-1]
        c3 = src[:, 2:]
        
        # Hash fonksiyonu
        hash_vals = (c1 * (self.vocab_size**2) + c2 * self.vocab_size + c3) % self.num_buckets
        
        mask = (c1 > 0) & (c2 > 0) & (c3 > 0)
        
        for b in range(batch_size):
            valid_hashes = hash_vals[b][mask[b]]
            if len(valid_hashes) > 0:
                counts = torch.bincount(valid_hashes, minlength=self.num_buckets).float()
                trigram_counts[b] = counts
                
        seq_lengths = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        return trigram_counts / seq_lengths
    
    def forward(self, src):
        bigram_freqs = self.compute_bigram_freqs(src)      # [B, 1089]
        bigram_features = self.bigram_encoder(bigram_freqs)  # [B, 32]
        
        trigram_freqs = self.compute_trigram_freqs(src)    # [B, 512]
        trigram_features = self.trigram_encoder(trigram_freqs)  # [B, 32]
        
        return torch.cat([bigram_features, trigram_features], dim=1)  # [B, 64]


class SubstitutionLSTM(nn.Module):
    def __init__(self, vocab_size=33, embed_dim=256, hidden_size=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.SOS_TOKEN = 1
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 1. Context Encoder (Saf LSTM'e geri dönüş - v7 stili ama güçlü)
        self.cipher_context_encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 2. İstatistik Encoderlar (Dedektifler)
        
        # A. Unigram (Tek harf) Frekansı - v7'den miras
        self.unigram_encoder = nn.Sequential(
            nn.Linear(vocab_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # B. N-Gram (İkili/Üçlü) Frekansı - YENİ SİLAHIMIZ
        self.ngram_encoder = GlobalNGramEncoder(vocab_size, output_dim=64)
        
        # 3. Main Solver RNN
        # Girdiler:
        # - Cipher Char Embed (256)
        # - Plain Char Embed (256)
        # - Context (512)
        # - Unigram Freq (32)
        # - NGram Freq (64) <--- YENİ
        # Toplam: 1120
        self.rnn = nn.LSTM(
            input_size=embed_dim * 2 + hidden_size + 32 + 64,
            hidden_size=hidden_size,
            num_layers=3, # 3 katmanlı derinlik korundu
            dropout=0.2,
            batch_first=True,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_size, vocab_size)

    def compute_global_stats(self, src):
        """Unigram (Tek harf) frekansları"""
        one_hots = F.one_hot(src, num_classes=self.vocab_size).float()
        mask = (src != 0).float().unsqueeze(2)
        total_counts = (one_hots * mask).sum(dim=1)
        seq_lengths = mask.sum(dim=1)
        freq_dist = total_counts / seq_lengths.clamp(min=1)
        return freq_dist

    def forward(self, src, tgt_input=None):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt_input)
        
        # A. Context (LSTM)
        # Artık CNN yok, harfleri karıştırmadan bağlamı öğreniyoruz
        cipher_context, _ = self.cipher_context_encoder(src_emb) # [B, S, 512]
        
        # B. İstatistikler
        # 1. Tekli Frekans
        unigram_freqs = self.compute_global_stats(src)
        unigram_feat = self.unigram_encoder(unigram_freqs) # [B, 32]
        
        # 2. Çoklu Frekans (Bigram/Trigram)
        ngram_feat = self.ngram_encoder(src) # [B, 64]
        
        # İstatistikleri birleştir: [B, 96]
        global_stats = torch.cat([unigram_feat, ngram_feat], dim=1)
        
        # Zamana yay (Expand)
        seq_len = src.size(1)
        global_stats_expanded = global_stats.unsqueeze(1).expand(-1, seq_len, -1) # [B, S, 96]
        
        # C. Hepsini Birleştir
        combined_input = torch.cat((src_emb, tgt_emb, cipher_context, global_stats_expanded), dim=2)
        
        outputs, _ = self.rnn(combined_input)
        logits = self.fc(outputs)
        
        return logits

    def generate(self, src):
        # Generate fonksiyonunda da aynı birleştirmeleri yapmalısın!
        batch_size, seq_len = src.size()
        device = src.device
        
        src_emb = self.embedding(src)
        cipher_context, _ = self.cipher_context_encoder(src_emb)
        
        # İstatistikleri hesapla
        unigram_freqs = self.compute_global_stats(src)
        unigram_feat = self.unigram_encoder(unigram_freqs)
        
        ngram_feat = self.ngram_encoder(src)
        
        global_stats = torch.cat([unigram_feat, ngram_feat], dim=1)
        global_stats_expanded = global_stats.unsqueeze(1) # [B, 1, 96]
        
        current_input_token = torch.full((batch_size, 1), self.SOS_TOKEN, dtype=torch.long, device=device)
        h = None
        predictions = []
        
        for t in range(seq_len):
            cipher_char_emb = src_emb[:, t:t+1, :]
            tgt_emb = self.embedding(current_input_token)
            context = cipher_context[:, t:t+1, :]
            
            # Global istatistik her adımda aynı
            combined_input = torch.cat((cipher_char_emb, tgt_emb, context, global_stats_expanded), dim=2)
            
            output, h = self.rnn(combined_input, h)
            logit = self.fc(output)
            predicted_token = torch.argmax(logit, dim=-1)
            
            predictions.append(predicted_token)
            current_input_token = predicted_token

        return torch.cat(predictions, dim=1)
    
    def generate_beam(self, src, beam_width=3):
        """
        Beam Search Decoding for v9.5
        v9.5 Mimarisine uygun: CNN yok, Global N-Gram istatistikleri var.
        """
        batch_size, seq_len = src.size()
        device = src.device
        
        # A. Harf ve Bağlam
        src_emb = self.embedding(src) # [Batch, Seq, Embed]
        cipher_context, _ = self.cipher_context_encoder(src_emb) # [Batch, Seq, Hidden]
        
        # B. Global İstatistikler
        # Unigram
        unigram_freqs = self.compute_global_stats(src)
        unigram_feat = self.unigram_encoder(unigram_freqs)
        # N-Gram
        ngram_feat = self.ngram_encoder(src)
        
        # İstatistikleri birleştir: [Batch, 96]
        global_stats = torch.cat([unigram_feat, ngram_feat], dim=1)

        results = []
        
        # Batch içindeki her örnek için Beam Search yap
        for b in range(batch_size):
            # Hipotez Yapısı: (Score, Last_Token, Hidden_State, Sequence_So_Far)
            # Score 0.0 ile başlıyoruz (Log probability toplamı olacak)
            hypotheses = [(0.0, self.SOS_TOKEN, None, [])]
            
            # O anki batch örneğine ait verileri çek (Slice alıyoruz)
            curr_cipher_emb = src_emb[b]      # [Seq, Embed]
            curr_context = cipher_context[b]  # [Seq, Hidden]
            curr_stats = global_stats[b]      # [96]
            
            # Sequence boyunca ilerle
            for t in range(seq_len):
                all_candidates = []
                
                # Mevcut Beam'deki her yaşayan hipotezi genişlet
                for score, inp_token, h_state, seq_so_far in hypotheses:
                    
                    # 1. Cipher Char (t anındaki şifreli harf) -> [1, 1, Embed]
                    cipher_char_t = curr_cipher_emb[t:t+1].unsqueeze(0)
                    
                    # 2. Target Input (Bir önceki tahmin edilen harf) -> [1, 1, Embed]
                    tgt_token_tensor = torch.tensor([[inp_token]], device=device)
                    tgt_emb = self.embedding(tgt_token_tensor)
                    
                    # 3. Context (t anındaki bağlam) -> [1, 1, Hidden]
                    context_t = curr_context[t:t+1].unsqueeze(0)
                    
                    # 4. Global Stats (Metin boyunca sabit) -> [1, 1, 96]
                    stats_t = curr_stats.view(1, 1, -1)
                    
                    # Sıralama: Src_Emb + Tgt_Emb + Context + Global_Stats
                    combined = torch.cat((cipher_char_t, tgt_emb, context_t, stats_t), dim=2)
                    
                    # --- LSTM ADIMI ---
                    out, new_h = self.rnn(combined, h_state)
                    
                    # --- SKORLAMA ---
                    log_probs = F.log_softmax(self.fc(out), dim=-1).squeeze() # [Vocab]
                    
                    # En iyi 'beam_width' kadar adayı seç
                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                    
                    for k in range(beam_width):
                        new_score = score + topk_log_probs[k].item()
                        new_token = topk_indices[k].item()
                        # Yeni adayı listeye ekle
                        all_candidates.append((new_score, new_token, new_h, seq_so_far + [new_token]))
                
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                hypotheses = all_candidates[:beam_width]
            
            best_seq = hypotheses[0][3]
            results.append(torch.tensor(best_seq, device=device))
            
        return torch.stack(results)
    
    def generate_consistent(self, src):
        raw_prediction = self.generate_beam(src) # or self.generate(src)
        
        batch_size, seq_len = src.size()
        device = src.device
        
        refined_outputs = []
        
        for b in range(batch_size):
            cipher_seq = src[b].tolist()
            pred_seq = raw_prediction[b].tolist()
            
            # votes[cipher_char][plain_char] = count
            votes = {} 
            
            for c_char, p_char in zip(cipher_seq, pred_seq):
                # Ignore special tokens
                if c_char < 4: continue 
                
                if c_char not in votes:
                    votes[c_char] = {}
                if p_char not in votes[c_char]:
                    votes[c_char][p_char] = 0
                
                votes[c_char][p_char] += 1
            
            # Step 3: Build the "Winner" Dictionary
            final_key_map = {}
            used_plain_chars = set()
            
            # Sort cipher chars by total frequency (most common first to claim best plain chars)
            sorted_cipher_chars = sorted(votes.keys(), key=lambda k: sum(votes[k].values()), reverse=True)
            
            for c_char in sorted_cipher_chars:
                # Find the plain char with the highest vote count
                candidates = votes[c_char]
                # Sort candidates by vote count
                sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
                
                best_plain = sorted_candidates[0][0]
                
                # If 'S' is already taken by a stronger cipher char, take the next best guess
                for cand_plain, count in sorted_candidates:
                    if cand_plain not in used_plain_chars:
                        best_plain = cand_plain
                        break
                
                final_key_map[c_char] = best_plain
                used_plain_chars.add(best_plain)
            
            # If a char wasn't in the map (e.g. rare char), keep the raw prediction or use a placeholder
            new_seq = []
            for i, c_char in enumerate(cipher_seq):
                if c_char in final_key_map:
                    new_seq.append(final_key_map[c_char])
                else:
                    new_seq.append(pred_seq[i]) # Fallback to raw prediction
            
            refined_outputs.append(torch.tensor(new_seq, device=device))
            
        return torch.stack(refined_outputs)