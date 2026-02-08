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
        # SOS ve EOS için yer ayırıyoruz
        self.content_len = self.seq_len - 2 

        self.chunks = self._compute_optimized_chunks() 
        print(f"Toplam Chunk Sayısı: {len(self.chunks)}")

    def _compute_optimized_chunks(self):
        chunks = []
        pos = 0

        while pos < self.text_len:
            end_pos = min(pos + self.content_len, self.text_len)

            # Son parçaya geldiysek direkt ekle
            if end_pos >= self.text_len:
                chunks.append((pos, end_pos))
                break

            # Kelimeyi ortadan bölmemek için son boşluğu bul
            chunk_str = self.text[pos: end_pos]
            last_space_idx = chunk_str.rfind(' ')

            if last_space_idx != -1:
                # Boşluğun olduğu yere kadar al (+1 boşluğu da dahil etmek için)
                end_pos = pos + last_space_idx + 1
            
            chunks.append((pos, end_pos))
            pos = end_pos

        return chunks
    
    def __len__(self):
        return len(self.chunks)
    
    def generate_random_key(self):
        real_chars = self.full_alphabet[4:]
        shuffled = list(real_chars)
        random.shuffle(shuffled)
        return {src: target for src, target in zip(real_chars, shuffled)}
    
    def str_to_indices(self, text):
        return [self.char2idx.get(c, 0) for c in text]
    
    def get_chunk(self, index):
        start, end = self.chunks[index]
        return self.text[start:end]
    
    def __getitem__(self, index):
        # 1. Orijinal uzun (ve kelime bütünlüğü korunmuş) parçayı al
        plain_text_raw = self.get_chunk(index)

        # --- DİNAMİK KIRPMA (Kelime Bütünlüğünü Koruyarak) ---
        # Strateji: %50 ihtimalle metni kısalt, ama kelimeleri parçalama.
        
        # Sadece yeterince uzun metinlerde yap
        if len(plain_text_raw) > 60 and random.random() < 0.5:
            # Metni kelimelere ayır
            words = plain_text_raw.split(self.space)
            
            # Eğer parçalayacak kadar kelime varsa
            if len(words) > 3:
                # Rastgele bir kelime sayısı seç (Örn: Toplam 20 kelime varsa, 5 ile 15 arası bir sayı seç)
                # Amacımız 40-150 karakter arası bir şey tutturmak ama kelime bazlı gidiyoruz.
                min_words = 3
                max_words = len(words)
                
                # Rastgele pencere boyutu (kelime sayısı olarak)
                num_words_to_take = random.randint(min_words, max_words - 1)
                
                # Rastgele başlangıç kelimesi
                start_word_idx = random.randint(0, len(words) - num_words_to_take)
                
                # Seçilen kelimeleri birleştir
                selected_words = words[start_word_idx : start_word_idx + num_words_to_take]
                plain_text = self.space.join(selected_words)
                
                # Eğer kırpma sonucu çok çok kısa kaldıysa (örn 2 harf), orijinali kullan (Güvenlik önlemi)
                if len(plain_text) < 10:
                    plain_text = plain_text_raw
            else:
                plain_text = plain_text_raw
        else:
            # Kısaltma yok, orijinali kullan
            plain_text = plain_text_raw
        # ----------------------------------------------------

        # Şifreleme bölümü
        keymap = self.generate_random_key()
        table = str.maketrans(keymap)
        cipher_text = plain_text.translate(table)

        # Vektörleştirme
        plain_text_indices = self.str_to_indices(plain_text)
        cipher_text_indices = self.str_to_indices(cipher_text)

        # Encoder input (Padding ekle)
        # Dinamik kırpmada seq_len sabit kaldığı için padding artacaktır, bu normal.
        src = cipher_text_indices + [self.char2idx[self.PAD_TOKEN]] * (self.seq_len - len(cipher_text_indices))

        # Decoder input (Teacher Forcing)
        tgt_input = [self.char2idx[self.SOS_TOKEN]] + plain_text_indices
        tgt_input += [self.char2idx[self.PAD_TOKEN]] * (self.seq_len - len(tgt_input))

        # Target (Output)
        tgt_output = plain_text_indices + [self.char2idx[self.EOS_TOKEN]]
        tgt_output += [self.char2idx[self.PAD_TOKEN]] * (self.seq_len - len(tgt_output))

        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt_input, dtype=torch.long), torch.tensor(tgt_output, dtype=torch.long)