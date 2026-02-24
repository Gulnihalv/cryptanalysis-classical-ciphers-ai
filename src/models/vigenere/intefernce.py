import torch
from vigenere_transformer import VigenereLightningModule

checkpoint_path = "checkpoints_vigenere2/vigenere-epoch=24-val_loss=0.04.ckpt"
model = VigenereLightningModule.load_from_checkpoint(checkpoint_path)
model.eval()

# M1 (MPS) veya CPU seçimi
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

alphabet = "abcçdefgğhıijklmnoöprsştuüvyz"
char2int = {c: i for i, c in enumerate(list(alphabet))}
int2char = {i: c for i, c in enumerate(list(alphabet))}

plaintext = "bugün gezdim yedim"
key = [2, 5]
key_len_val = len(key)

ciphertext_indices = []
vocab_size = len(alphabet)

print(f"Orijinal: {plaintext}")

cleaned_plain = [c for c in plaintext if c in char2int or c == " "]
key_ptr = 0

# Şifreleme simülasyonu
for char in cleaned_plain:
    if char == " ":
        ciphertext_indices.append(30)
    else:
        p_val = char2int[char]
        k_val = key[key_ptr % key_len_val]
        c_val = (p_val + k_val) % vocab_size
        ciphertext_indices.append(c_val)
        key_ptr += 1

class DummyDataset:
    def __init__(self):
        self.SPACE_TOKEN_IDX = 30
        self.PAD_TOKEN_IDX = 29
        self.crypto_vocab_size = 29
        self.int2char = int2char

cipher_tensor = torch.tensor(ciphertext_indices, dtype=torch.long)
decrypted_text = model.decrypt(cipher_tensor, key_len_val, DummyDataset())

print(f"Model Çıktısı: {decrypted_text}")