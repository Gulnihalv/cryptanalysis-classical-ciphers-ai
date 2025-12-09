import numpy as np
import random
import math
# daha sonrasında frekanslar bozulmasın diye batch sonlarında kelime bölünmüş mü kontrolü ekelenecek.
class SubstutionDataGenerator():
    def __init__(self, raw_text_path, alphabet, sequence_len, batch_size):
        self.alphabet = alphabet
        self.seq_len = sequence_len
        self.batch_size = batch_size

        with open(raw_text_path, "r" , encoding='utf-8') as f:
            self.full_text = f.read().replace('\n', ' ')

        self.text_len = len(self.full_text) 

        self.char2idx = {char: i for i, char in enumerate(alphabet)}
        self.indices = list(range(0, self.text_len - self.seq_len, self.seq_len))

        self.on_epoch_end()
        print("Data is ready. Number of chunks:", {len(self.indices)})

    def on_epoch_end(self):
        random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)
    
    def generate_random_key(self):
        shuffled = list(self.alphabet)
        random.shuffle(shuffled)
        return {src: target for src, target in zip(self.alphabet, shuffled)}
    
    def str_to_indices(self, text):
        return [self.char2idx.get(c, 0) for c in text]
    
    def get_batch(self, batch_index):
        start = batch_index * self.batch_size
        end = start + self.batch_size
        batch_indices = self.indices[start : end]

        batch_X = []
        batch_y = []

        for idx in batch_indices:
            plaintext_chunk = self.full_text[idx : idx + self.seq_len]

            if len(plaintext_chunk) < self.seq_len:
                plaintext_chunk = plaintext_chunk.ljust(self.seq_len, ' ')

            key_map = self.generate_random_key()

            #şifreleme
            table = str.maketrans(key_map)
            ciphertext = plaintext_chunk.translate(table)

            batch_X.append(self.str_to_indices(ciphertext))
            batch_y.append(self.str_to_indices(plaintext_chunk))

        return np.array(batch_X), np.array(batch_y)



    