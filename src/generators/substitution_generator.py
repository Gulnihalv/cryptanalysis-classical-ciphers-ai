import random
import torch
from torch.utils.data import Dataset

class SubstutionDataGenerator(Dataset):
    def __init__(self, text_path, alphabet, seq_len):
        super().__init__()

        self.PAD_TOKEN = "<PAD>"
        self.SOS_TOKEN = "<SOS>"
        self.EOS_TOKEN = "<EOS>"
        self.space = " "

        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.space]
        self.full_alphabet = self.special_tokens + list(alphabet)

        self.seq_len = seq_len
        self.char2idx = {c: i for i, c in enumerate(self.full_alphabet)}
        self.idx2char = {i: c for i, c in enumerate(self.full_alphabet)}

        with open(text_path, "r", encoding="utf-8") as f:
            self.text = f.read().replace("\n", " ")

        self.text_len = len(self.text)
        self.content_len = self.seq_len - 2 # SOS ile EOS u seq_len den çıkarıyorum

        self.chunks = self._compute_optimized_chunks() # her chunk başlangıç noktasının hesaplanması için
        print(f"Toplam Chunk Sayısı: {len(self.chunks)}")

    def _compute_optimized_chunks(self):
        chunks = []
        pos = 0

        while pos < self.text_len:
            end_pos = min(pos + self.content_len, self.text_len)

            # son chunk
            if end_pos >= self.text_len:
                chunks.append((pos, end_pos))
                break

            is_char_valid = (end_pos < self.text_len)

            if (is_char_valid and self.text[end_pos] != ' ') and (self.text[end_pos-1] != ' '):
                chunk_str = self.text[pos: end_pos]
                last_space_idx = chunk_str.rfind(' ')

                if (last_space_idx != -1):
                    end_pos = pos + last_space_idx + 1
            
            chunks.append((pos, end_pos))
            pos = end_pos

        return chunks


    def __len__(self):
        return len(self.chunks)
    
    def generate_random_key(self):
        #harfleri karıştırıp dict olarak dönüyoruz. Ve özel tokenları çıkarıyoruz.
        real_chars = self.full_alphabet[4:]
        shuffled = list(real_chars)
        random.shuffle(shuffled)
        return {src: target for src, target in zip(real_chars, shuffled)}
    
    def str_to_indices(self, text):
        # char2idx içeriğini doldurucaz
        return [self.char2idx.get(c, 0) for c in text]
    
    def get_chunk(self, index):
        start, end = self.chunks[index]
        return self.text[start:end]
    
    def __getitem__(self, index):
        plain_text = self.get_chunk(index)

        # Şifreleme bölümü
        keymap = self.generate_random_key()
        table = str.maketrans(keymap)
        cipher_text = plain_text.translate(table)

        # vektörleştirme
        plain_text_indices = self.str_to_indices(plain_text)
        cipher_text_indices = self.str_to_indices(cipher_text)

        # encoder input için
        src = cipher_text_indices + [self.char2idx[self.PAD_TOKEN]] * (self.seq_len - len(cipher_text_indices))

        # decoder için (teacher)
        tgt_input = [self.char2idx[self.SOS_TOKEN]] + plain_text_indices
        tgt_input += [self.char2idx[self.PAD_TOKEN]] * (self.seq_len - len(tgt_input))

        # target : modelin tahmin çıktısı
        tgt_output = plain_text_indices + [self.char2idx[self.EOS_TOKEN]]
        tgt_output += [self.char2idx[self.PAD_TOKEN]] * (self.seq_len - len(tgt_output))

        # pytorch için indexten tensora çevirdik
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt_input, dtype=torch.long), torch.tensor(tgt_output, dtype=torch.long)