import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, E]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class SubstitutionEncoderModel(nn.Module):
    """
    Cipher tokens
        │
        ├─► Embedding + Positional Encoding
        │
        ├─► FREQ TOKEN: global frekans istatistiği → [B, 1, E] olarak prepend
        │   (encoder'daki her pozisyon buna attend edebilir)
        │
        ├─► Transformer Encoder (bi-directional, N layer)
        │
        └─► Linear projection → [B, S, vocab_size] logits
    """

    def __init__(
        self,
        vocab_size: int = 33,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.vocab_size = vocab_size
        self.PAD_IDX = 0

        # ── Embeddings ───────────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)

        # ── Frequency token encoder ──────────────────────────────────────────
        # Global karakter frekansı → embed_dim boyutunda bir "özet token"
        self.freq_encoder = nn.Sequential(
            nn.Linear(vocab_size, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # ── Transformer Encoder ──────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: daha stabil
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # ── Output projection ────────────────────────────────────────────────
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_global_stats(self, src: torch.Tensor) -> torch.Tensor:
        """Normalize edilmiş karakter frekans histogramı. → [B, vocab_size]"""
        one_hots = F.one_hot(src, num_classes=self.vocab_size).float()
        mask = (src != self.PAD_IDX).float().unsqueeze(2)
        total_counts = (one_hots * mask).sum(dim=1)
        seq_lengths = mask.sum(dim=1).clamp(min=1)
        return total_counts / seq_lengths

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src : [B, S]  cipher token indices
        Returns:
            logits : [B, S, vocab_size]
        """
        B, S = src.shape
        device = src.device

        # Padding mask: True olan pozisyonlar ignore edilir
        pad_mask = src == self.PAD_IDX  # [B, S]

        # Frekans token'ı: [B, 1, E]
        global_freqs = self.compute_global_stats(src)
        freq_token = self.freq_encoder(global_freqs).unsqueeze(1)

        # Cipher embedding + pozisyonel encoding: [B, S, E]
        src_emb = self.pos_encoding(self.embedding(src))

        # Freq token'ı başa ekle → [B, S+1, E]
        encoder_input = torch.cat([freq_token, src_emb], dim=1)

        # Padding mask'i genişlet (freq token hiçbir zaman padding değil)
        freq_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)
        full_pad_mask = torch.cat([freq_pad, pad_mask], dim=1)  # [B, S+1]

        # Encoder çalıştır: [B, S+1, E]
        encoder_out = self.encoder(encoder_input, src_key_padding_mask=full_pad_mask)

        # Freq token'ı at, sadece orijinal S pozisyonu al: [B, S, E]
        token_out = encoder_out[:, 1:, :]

        # Logit projeksiyon: [B, S, vocab_size]
        return self.fc_out(token_out)

    @torch.no_grad()
    def generate(self, src: torch.Tensor) -> torch.Tensor:
        logits = self.forward(src)           # [B, S, V]
        return logits.argmax(dim=-1)         # [B, S]

    @torch.no_grad()
    def generate_consistent(self, src: torch.Tensor) -> torch.Tensor:
        raw_prediction = self.generate(src)

        results = []
        for b in range(src.size(0)):
            cipher_seq = src[b].tolist()
            pred_seq = raw_prediction[b].tolist()

            votes: dict[int, dict[int, int]] = {}
            for c_char, p_char in zip(cipher_seq, pred_seq):
                if c_char < 4:
                    continue
                votes.setdefault(c_char, {}).setdefault(p_char, 0)
                votes[c_char][p_char] += 1

            final_key_map: dict[int, int] = {}
            used_plain_chars: set[int] = set()

            sorted_cipher_chars = sorted(
                votes.keys(), key=lambda k: sum(votes[k].values()), reverse=True
            )
            for c_char in sorted_cipher_chars:
                sorted_cands = sorted(
                    votes[c_char].items(), key=lambda x: x[1], reverse=True
                )
                best = sorted_cands[0][0]
                for cand, _ in sorted_cands:
                    if cand not in used_plain_chars:
                        best = cand
                        break
                final_key_map[c_char] = best
                used_plain_chars.add(best)

            new_seq = [
                final_key_map.get(c, pred_seq[i])
                for i, c in enumerate(cipher_seq)
            ]
            results.append(torch.tensor(new_seq, device=src.device))

        return torch.stack(results)