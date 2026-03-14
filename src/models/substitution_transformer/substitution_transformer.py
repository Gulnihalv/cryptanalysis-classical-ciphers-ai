import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
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
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, E]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SubstitutionTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for substitution cipher solving.

    Key design decisions vs LSTM v7:
    - Global frequency features injected as a learned "FREQ" token prepended to the
      encoder sequence (similar to [CLS]).  This lets every encoder/decoder position
      attend to the global statistics via self/cross-attention.
    - Causal mask on the decoder to stay autoregressive (compatible with teacher forcing).
    - Padding mask propagated everywhere so the model never attends to <PAD>.
    """

    def __init__(
        self,
        vocab_size: int = 33,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.SOS_TOKEN = 1
        self.PAD_IDX = 0

        # ── Embeddings ──────────────────────────────────────────────────────────
        self.src_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.tgt_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)

        # ── Frequency Encoder ────────────────────────────────────────────────────
        # Compresses global char-frequency histogram → single embed_dim vector
        # that is prepended to the encoder sequence as a special "FREQ" token.
        self.freq_encoder = nn.Sequential(
            nn.Linear(vocab_size, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # ── Transformer ──────────────────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # ── Output projection ────────────────────────────────────────────────────
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    # ────────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────────

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_global_stats(self, src: torch.Tensor) -> torch.Tensor:
        """Normalised character frequency histogram.  [B] → [B, vocab_size]"""
        one_hots = F.one_hot(src, num_classes=self.vocab_size).float()
        mask = (src != self.PAD_IDX).float().unsqueeze(2)
        total_counts = (one_hots * mask).sum(dim=1)
        seq_lengths = mask.sum(dim=1).clamp(min=1)
        return total_counts / seq_lengths

    def _make_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (additive, -inf for masked positions)."""
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def _encode(self, src: torch.Tensor):
        """
        Encode cipher text + prepend frequency token.

        Returns:
            memory        : [B, S+1, E]   (+1 for the FREQ token)
            memory_key_padding_mask : [B, S+1]  (True = ignore)
        """
        B, S = src.shape
        device = src.device

        # Padding mask for source tokens
        src_pad_mask = src == self.PAD_IDX  # [B, S]

        # Frequency token: [B, 1, E]
        global_freqs = self.compute_global_stats(src)
        freq_token = self.freq_encoder(global_freqs).unsqueeze(1)  # [B, 1, E]

        # Source embeddings + positional encoding
        src_emb = self.pos_encoding(self.src_embedding(src))  # [B, S, E]

        # Prepend freq token → [B, S+1, E]
        encoder_input = torch.cat([freq_token, src_emb], dim=1)

        # Extend padding mask: freq token is never padding → [B, S+1]
        freq_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        memory_pad_mask = torch.cat([freq_mask, src_pad_mask], dim=1)

        memory = self.encoder(encoder_input, src_key_padding_mask=memory_pad_mask)
        return memory, memory_pad_mask

    # ────────────────────────────────────────────────────────────────────────────
    # Forward (teacher-forced)
    # ────────────────────────────────────────────────────────────────────────────

    def forward(self, src: torch.Tensor, tgt_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src       : [B, S]  cipher token indices
            tgt_input : [B, S]  decoder input (SOS + plain tokens, shifted right)
        Returns:
            logits    : [B, S, vocab_size]
        """
        T = tgt_input.size(1)
        device = src.device

        memory, memory_pad_mask = self._encode(src)

        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt_input))  # [B, T, E]
        tgt_pad_mask = tgt_input == self.PAD_IDX                     # [B, T]
        causal_mask = self._make_causal_mask(T, device)               # [T, T]

        decoder_out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )  # [B, T, E]

        return self.fc_out(decoder_out)  # [B, T, vocab_size]

    # ────────────────────────────────────────────────────────────────────────────
    # Greedy autoregressive generation
    # ────────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, src: torch.Tensor) -> torch.Tensor:
        """
        Greedy decoding — matches the LSTM v7 interface exactly.

        Returns: [B, S]  predicted plain token indices
        """
        B, S = src.shape
        device = src.device

        memory, memory_pad_mask = self._encode(src)

        # Start with SOS for all samples in the batch
        generated = torch.full((B, 1), self.SOS_TOKEN, dtype=torch.long, device=device)

        for _ in range(S):
            T_cur = generated.size(1)
            tgt_emb = self.pos_encoding(self.tgt_embedding(generated))
            causal_mask = self._make_causal_mask(T_cur, device)
            tgt_pad_mask = generated == self.PAD_IDX

            dec_out = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=memory_pad_mask,
            )  # [B, T_cur, E]

            # Take only the last time-step logit
            next_logits = self.fc_out(dec_out[:, -1, :])  # [B, vocab_size]
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)

        # Strip leading SOS token → [B, S]
        return generated[:, 1:]

    # ────────────────────────────────────────────────────────────────────────────
    # Beam search
    # ────────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_beam(self, src: torch.Tensor, beam_width: int = 3) -> torch.Tensor:
        """
        Beam search decoding.  Processes one sample at a time (no batch parallelism)
        to keep the beam bookkeeping simple — same trade-off as LSTM v7.
        """
        B, S = src.shape
        device = src.device

        results = []

        for b in range(B):
            src_b = src[b : b + 1]  # [1, S]
            memory, memory_pad_mask = self._encode(src_b)

            # hypotheses: list of (score, token_sequence_tensor)
            hypotheses = [(0.0, torch.tensor([self.SOS_TOKEN], dtype=torch.long, device=device))]

            for _ in range(S):
                candidates = []

                for score, seq in hypotheses:
                    T_cur = seq.size(0)
                    tgt_in = seq.unsqueeze(0)  # [1, T_cur]
                    tgt_emb = self.pos_encoding(self.tgt_embedding(tgt_in))
                    causal_mask = self._make_causal_mask(T_cur, device)

                    dec_out = self.decoder(
                        tgt_emb,
                        memory,
                        tgt_mask=causal_mask,
                        memory_key_padding_mask=memory_pad_mask,
                    )
                    log_probs = F.log_softmax(self.fc_out(dec_out[:, -1, :]), dim=-1).squeeze(0)

                    topk_lp, topk_idx = torch.topk(log_probs, beam_width)
                    for k in range(beam_width):
                        new_score = score + topk_lp[k].item()
                        new_seq = torch.cat([seq, topk_idx[k : k + 1]])
                        candidates.append((new_score, new_seq))

                candidates.sort(key=lambda x: x[0], reverse=True)
                hypotheses = candidates[:beam_width]

            best_seq = hypotheses[0][1][1:]  # Strip SOS
            results.append(best_seq)

        return torch.stack(results)

    # ────────────────────────────────────────────────────────────────────────────
    # Post-processing: enforce one-to-one cipher→plain mapping via majority vote
    # (identical logic to LSTM v7 — reusable regardless of model backbone)
    # ────────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_consistent(self, src: torch.Tensor) -> torch.Tensor:
        raw_prediction = self.generate_beam(src)

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
                sorted_candidates = sorted(
                    votes[c_char].items(), key=lambda x: x[1], reverse=True
                )
                best_plain = sorted_candidates[0][0]
                for cand_plain, _ in sorted_candidates:
                    if cand_plain not in used_plain_chars:
                        best_plain = cand_plain
                        break
                final_key_map[c_char] = best_plain
                used_plain_chars.add(best_plain)

            new_seq = [
                final_key_map.get(c_char, pred_seq[i])
                for i, c_char in enumerate(cipher_seq)
            ]
            results.append(torch.tensor(new_seq, device=src.device))

        return torch.stack(results)