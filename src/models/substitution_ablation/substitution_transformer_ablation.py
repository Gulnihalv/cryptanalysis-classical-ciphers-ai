import torch
import torch.nn as nn
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
    Ablation variant: NO frequency token — pure Transformer Encoder.

    Cipher tokens
        │
        ├─► Embedding + Positional Encoding
        │
        ├─► Transformer Encoder (bi-directional, N layers)
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

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src    : [B, S]  cipher token indices
        Returns:
            logits : [B, S, vocab_size]
        """
        # Padding mask: True olan pozisyonlar ignore edilir
        pad_mask = src == self.PAD_IDX  # [B, S]

        # Embedding + positional encoding: [B, S, E]
        src_emb = self.pos_encoding(self.embedding(src))

        # Encoder: [B, S, E]
        encoder_out = self.encoder(src_emb, src_key_padding_mask=pad_mask)

        # Logit projeksiyon: [B, S, vocab_size]
        return self.fc_out(encoder_out)

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
