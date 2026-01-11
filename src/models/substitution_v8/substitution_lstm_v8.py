import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConv1D(nn.Module):
    """
    Residual bağlantıya sahip Conv1D bloğu. 
    LSTM yerine bu yapı kullanılarak lokal bağlam (komşu harfler) öğrenilir.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        # Padding, giriş ve çıkış uzunluğunun (Seq Len) aynı kalmasını sağlar.
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.InstanceNorm1d(out_channels) # Batch bağımsızlığı için InstanceNorm
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm1d(out_channels)
        
        # Eğer giriş ve çıkış kanalları farklıysa residual bağlantı için projeksiyon gerekir
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [Batch, Channel(Embed), Length]
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        return self.relu(out + residual)

class SubstitutionLSTMV8(nn.Module):
    def __init__(self, vocab_size=33, embed_dim=256, hidden_size=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.SOS_TOKEN = 1
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Conv1D input'u [Batch, Embed, Seq] bekler.
        # Kernel size 3: (Sol, Merkez, Sağ) -> Trigram penceresi görür.
        self.context_encoder = nn.Sequential(
            # Katman 1: Embed -> Hidden
            ResidualConv1D(embed_dim, hidden_size, kernel_size=3, dropout=0.2),
            # Katman 2: Hidden -> Hidden (Derinlik katar, daha geniş alanı görür)
            ResidualConv1D(hidden_size, hidden_size, kernel_size=3, dropout=0.2)
        )
        
        # 2. Frequency Encoder
        self.freq_encoder = nn.Sequential(
            nn.Linear(vocab_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 3. Main Solver RNN
        # Inputs: 
        #   [Cipher Embed (256)] + 
        #   [Prev Plain Embed (256)] + 
        #   [Local Conv Context (512)] +
        #   [Global Freq Stats (32)]
        # Total: 256 + 256 + 512 + 32 = 1056
        
        rnn_input_size = embed_dim * 2 + hidden_size + 32
        
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=3,
            dropout=0.2,
            batch_first=True,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_size, vocab_size)

    def compute_global_stats(self, src):
        """Batch bazlı frekans histogramı çıkarır."""
        one_hots = F.one_hot(src, num_classes=self.vocab_size).float()
        mask = (src != 0).float().unsqueeze(2)
        total_counts = (one_hots * mask).sum(dim=1)
        seq_lengths = mask.sum(dim=1)
        freq_dist = total_counts / seq_lengths.clamp(min=1)
        return freq_dist

    def forward(self, src, tgt_input=None):
        # Embeddings: [Batch, Seq, Embed]
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt_input)
        
        # Conv1D [Batch, Channel, Length] ister. Bizde [Batch, Length, Channel] var.
        # Permute ile yer değiştiriyoruz: (0, 2, 1)
        conv_input = src_emb.permute(0, 2, 1) 
        
        # Encoder'dan geçir
        conv_out = self.context_encoder(conv_input) # Çıktı: [Batch, Hidden, Length]
        
        # Tekrar eski haline getir: [Batch, Length, Hidden]
        cipher_context = conv_out.permute(0, 2, 1)
        
        global_freqs = self.compute_global_stats(src) # [B, V]
        freq_features = self.freq_encoder(global_freqs) # [B, 32]
        
        seq_len = src.size(1)
        freq_features_expanded = freq_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine everything
        combined_input = torch.cat((src_emb, tgt_emb, cipher_context, freq_features_expanded), dim=2)
        
        outputs, _ = self.rnn(combined_input)
        logits = self.fc(outputs)
        
        return logits

    def generate(self, src):
        batch_size, seq_len = src.size()
        device = src.device
        
        src_emb = self.embedding(src)
        
        # --- Precompute Context (Conv1D Parallel) ---
        conv_input = src_emb.permute(0, 2, 1)
        conv_out = self.context_encoder(conv_input)
        cipher_context = conv_out.permute(0, 2, 1) # [B, S, Hidden]
        
        # Precompute Frequencies
        global_freqs = self.compute_global_stats(src)
        freq_features = self.freq_encoder(global_freqs) # [B, 32]
        
        current_input_token = torch.full((batch_size, 1), self.SOS_TOKEN, dtype=torch.long, device=device)
        h = None
        predictions = []
        
        for t in range(seq_len):
            cipher_char_emb = src_emb[:, t:t+1, :]
            tgt_emb = self.embedding(current_input_token)
            
            # Conv Context önceden hesaplandı, t anını çekiyoruz
            context = cipher_context[:, t:t+1, :]
            
            freq_t = freq_features.unsqueeze(1)
            
            combined_input = torch.cat((cipher_char_emb, tgt_emb, context, freq_t), dim=2)
            output, h = self.rnn(combined_input, h)
            
            logit = self.fc(output)
            predicted_token = torch.argmax(logit, dim=-1)
            
            predictions.append(predicted_token)
            current_input_token = predicted_token

        return torch.cat(predictions, dim=1)
    
    def generate_beam(self, src, beam_width=3):
        batch_size, seq_len = src.size()
        device = src.device
        
        # --- Precompute Encoders ---
        src_emb = self.embedding(src)
        
        # Conv1D için Permute
        conv_input = src_emb.permute(0, 2, 1)
        conv_out = self.context_encoder(conv_input)
        cipher_context = conv_out.permute(0, 2, 1) # [B, S, Hidden]

        global_freqs = self.compute_global_stats(src)
        freq_features = self.freq_encoder(global_freqs).unsqueeze(1)

        results = []
        
        for b in range(batch_size):
            hypotheses = [(0.0, self.SOS_TOKEN, None, [])]
            
            curr_cipher_emb = src_emb[b]    # [Seq, Embed]
            curr_context = cipher_context[b] # [Seq, Hidden] (Precomputed Conv Features)
            curr_freq = freq_features[b]     # [1, 32]
            
            for t in range(seq_len):
                candidates = []
                for score, inp_token, h_state, seq_so_far in hypotheses:
                    cipher_char_t = curr_cipher_emb[t:t+1].unsqueeze(0)
                    tgt_emb = self.embedding(torch.tensor([[inp_token]], device=device))
                    context_t = curr_context[t:t+1].unsqueeze(0)
                    freq_t = curr_freq.unsqueeze(0)
                    
                    combined = torch.cat((cipher_char_t, tgt_emb, context_t, freq_t), dim=2)
                    out, new_h = self.rnn(combined, h_state)
                    
                    log_probs = F.log_softmax(self.fc(out), dim=-1).squeeze()
                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                    
                    for k in range(beam_width):
                        new_score = score + topk_log_probs[k].item()
                        new_token = topk_indices[k].item()
                        candidates.append((new_score, new_token, new_h, seq_so_far + [new_token]))
                
                candidates.sort(key=lambda x: x[0], reverse=True)
                hypotheses = candidates[:beam_width]
            
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