import random
import torch
from torch.utils.data import Dataset

class VigenereKeystreamGenerator(Dataset):
    def __init__(self, text_path, alphabet, seq_len, min_key_len=3, max_key_len=12):
        super().__init__()

        self.crypto_alphabet = list(alphabet) 
        self.crypto_vocab_size = len(self.crypto_alphabet)
        
        self.total_vocab_size = self.crypto_vocab_size + 2
        self.int2char = {i: c for i, c in enumerate(self.crypto_alphabet)}
        self.char2int = {c: i for i, c in enumerate(self.crypto_alphabet)}
        
        # Token ID'leri
        self.PAD_TOKEN_IDX = self.crypto_vocab_size       # Padding ve Ignore Index
        self.SPACE_TOKEN_IDX = self.crypto_vocab_size + 1 # Input'taki boşluk
        
        self.seq_len = seq_len
        self.min_key_len = min_key_len
        self.max_key_len = max_key_len
        self.space_char = " "

        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.read().replace("\n", " ").strip()
            allowed = set(self.crypto_alphabet + [self.space_char])
            self.text = "".join([c for c in raw_text if c in allowed])

        self.text_len = len(self.text)
        self.chunks = self._compute_optimized_chunks()

    def _compute_optimized_chunks(self):
        chunks = []
        pos = 0
        while pos < self.text_len:
            end_pos = min(pos + self.seq_len, self.text_len)
            if end_pos >= self.text_len:
                chunks.append((pos, end_pos))
                break
            chunk_str = self.text[pos: end_pos]
            last_space_idx = chunk_str.rfind(' ')
            if last_space_idx != -1:
                end_pos = pos + last_space_idx + 1
            chunks.append((pos, end_pos))
            pos = end_pos
        return chunks

    def __len__(self):
        return len(self.chunks)

    def generate_random_key(self):
        k_len = random.randint(self.min_key_len, self.max_key_len)
        key_indices = [random.randint(0, self.crypto_vocab_size - 1) for _ in range(k_len)]
        return key_indices, k_len

    def encrypt_vigenere_skip_space(self, text_chunk, key_indices):
        """
        Hem şifreler hem de modelin tahmin etmesi gereken 'keystream'i oluşturur.
        """
        src_indices = []       # Model Input (Ciphertext)
        tgt_indices = []       # Opsiyonel (Plaintext - Sadece kontrol için)
        keystream_target = []  # Model Output Hedefi (Key ID'leri)
        cyclic_positions = []  # Embedding için (0, 1, 2...)
        
        key_len = len(key_indices)
        key_ptr = 0 

        for char in text_chunk:
            if char == self.space_char:
                # Input: Boşluk Tokeni
                src_indices.append(self.SPACE_TOKEN_IDX)
                tgt_indices.append(self.SPACE_TOKEN_IDX)
                
                # Hedef: PAD (Loss hesaplanmasın, model burayı unutsun)
                keystream_target.append(self.PAD_TOKEN_IDX) 
                
                # Cycle Pos: Nötr (key_len)
                cyclic_positions.append(key_len)
            
            elif char in self.char2int:
                p_val = self.char2int[char]
                
                # O anki anahtar değeri
                current_key = key_indices[key_ptr % key_len]
                
                # Şifreleme: C = (P + K) % Mod
                c_val = (p_val + current_key) % self.crypto_vocab_size
                
                src_indices.append(c_val)
                tgt_indices.append(p_val)
                
                # Modelin bu 'current_key'i (0-28) tahmin etmesi gerekiyor
                keystream_target.append(current_key)
                
                # Cycle Pos: 0, 1, 2...
                cyclic_positions.append(key_ptr % key_len)
                
                key_ptr += 1

        return src_indices, tgt_indices, keystream_target, cyclic_positions

    def __getitem__(self, index):
        text_chunk = self.text[self.chunks[index][0]: self.chunks[index][1]]

        # Dinamik Kırpma modelin her boyutta çalışabilrmesi için
        if len(text_chunk) > 40 and random.random() < 0.15:
            new_len = random.randint(20, len(text_chunk))
            text_chunk = text_chunk[:new_len]

        # Şifreleme ve Hedef Oluşturma
        key_indices, k_len = self.generate_random_key()
        
        src, tgt, key_target, cyc_pos = self.encrypt_vigenere_skip_space(text_chunk, key_indices)

        # Padding İşlemleri
        pad_len = self.seq_len - len(src)
        
        # Input Padding
        src_padded = src + [self.PAD_TOKEN_IDX] * pad_len
        
        # Target Padding
        key_target_padded = key_target + [self.PAD_TOKEN_IDX] * pad_len
        
        # Plaintext Padding
        tgt_padded = tgt + [self.PAD_TOKEN_IDX] * pad_len
        
        # Cycle Position Padding (Nötr pozisyon)
        cyc_pos_padded = cyc_pos + [k_len] * pad_len

        return {
            "src": torch.tensor(src_padded, dtype=torch.long),
            "key_target": torch.tensor(key_target_padded, dtype=torch.long), 
            "tgt_plain": torch.tensor(tgt_padded, dtype=torch.long),
            "cycle_pos": torch.tensor(cyc_pos_padded, dtype=torch.long),
            "key_len": torch.tensor(k_len, dtype=torch.long)
        }