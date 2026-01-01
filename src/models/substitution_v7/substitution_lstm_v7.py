import torch
import torch.nn as nn
import torch.nn.functional as F

class SubstitutionLSTM(nn.Module):
    def __init__(self, vocab_size=33, embed_dim=256, hidden_size=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.SOS_TOKEN = 1
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 1. Local Context (Bi-LSTM) - Keeps track of "what letter is next to me?"
        self.cipher_context_encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 2. Frequency Encoder - Compresses the "Language Fingerprint"
        # Input: 33 (vocab size) -> Output: 32 (condensed stats)
        self.freq_encoder = nn.Sequential(
            nn.Linear(vocab_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 3. Main Solver RNN
        # Inputs: [Cipher Char] + [Prev Plain Char] + [Local Context] + [Global Freq Stats]
        # Size: 256 + 256 + 512 + 32 = 1056
        self.rnn = nn.LSTM(
            input_size=embed_dim * 2 + hidden_size + 32,
            hidden_size=hidden_size,
            num_layers=3,
            dropout=0.2,
            batch_first=True,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_size, vocab_size)

    def compute_global_stats(self, src):
        """
        Calculates the frequency histogram of the batch using fast matrix operations.
        Returns: [Batch, Vocab_Size] (Normalized 0-1)
        """
        # Create One-Hot: [Batch, Seq, Vocab] -> e.g. [32, 250, 33]
        one_hots = F.one_hot(src, num_classes=self.vocab_size).float()
        
        # Mask padding (Index 0)
        mask = (src != 0).float().unsqueeze(2) # [Batch, Seq, 1]
        
        total_counts = (one_hots * mask).sum(dim=1) # [Batch, Vocab]
        
        # Normalize by sequence length (avoid div by zero)
        seq_lengths = mask.sum(dim=1) # [Batch, 1]
        freq_dist = total_counts / seq_lengths.clamp(min=1)
        
        return freq_dist

    def forward(self, src, tgt_input=None):
        # Embeddings
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt_input)
        
        # A. Local Bi-Directional Context
        cipher_context, _ = self.cipher_context_encoder(src_emb) # [B, S, H]
        
        # B. Global Frequency Context
        global_freqs = self.compute_global_stats(src) # [B, V]
        freq_features = self.freq_encoder(global_freqs) # [B, 32]
        
        # Expand global features to match sequence length (Repeat for every timestep)
        seq_len = src.size(1)
        freq_features_expanded = freq_features.unsqueeze(1).expand(-1, seq_len, -1) # [B, S, 32]
        
        # Combine everything
        combined_input = torch.cat((src_emb, tgt_emb, cipher_context, freq_features_expanded), dim=2)
        
        outputs, _ = self.rnn(combined_input)
        logits = self.fc(outputs)
        
        return logits

    def generate(self, src):
        batch_size, seq_len = src.size()
        device = src.device
        
        # Precompute Contexts ONCE
        src_emb = self.embedding(src)
        cipher_context, _ = self.cipher_context_encoder(src_emb)
        
        # Precompute Frequencies ONCE
        global_freqs = self.compute_global_stats(src)
        freq_features = self.freq_encoder(global_freqs) # [B, 32]
        
        current_input_token = torch.full((batch_size, 1), self.SOS_TOKEN, dtype=torch.long, device=device)
        h = None
        predictions = []
        
        for t in range(seq_len):
            cipher_char_emb = src_emb[:, t:t+1, :]
            tgt_emb = self.embedding(current_input_token)
            context = cipher_context[:, t:t+1, :]
            
            # Use the precomputed global stats for this step (same for all t)
            freq_t = freq_features.unsqueeze(1) # [B, 1, 32]
            
            combined_input = torch.cat((cipher_char_emb, tgt_emb, context, freq_t), dim=2)
            output, h = self.rnn(combined_input, h)
            
            logit = self.fc(output)
            predicted_token = torch.argmax(logit, dim=-1)
            
            predictions.append(predicted_token)
            current_input_token = predicted_token

        return torch.cat(predictions, dim=1)