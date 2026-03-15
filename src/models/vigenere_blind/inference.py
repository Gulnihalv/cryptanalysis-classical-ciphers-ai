import torch
from vigenere import VigenereBlindSolver

model = VigenereBlindSolver.load_from_checkpoint("checkpoints_blind2/predictor-epoch=79-val_acc=0.951-val_top2_acc=0.000.ckpt")
model2 = VigenereBlindSolver.load_from_checkpoint("checkpoints_blind/predictor-epoch=137-val_acc=0.983-val_top2_acc=0.000.ckpt")
model.eval()
model2.eval()

# M1 (MPS) veya CPU seçimi
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model2.to(device)

alphabet = "abcçdefgğhıijklmnoöprsştuüvyz"
char2int = {c: i for i, c in enumerate(list(alphabet))}
int2char = {i: c for i, c in enumerate(list(alphabet))}

plaintext = "düşünce sistemindeki temel görüş insan aklının aydınlattığı kesin doğrulara ve bilgiye doğru ilerlemektir"
plaintext = plaintext.replace(" ", "")
key = [2, 5]
key_len_val = len(key)

ciphertext_indices = []
vocab_size = len(alphabet)

print(f"Orijinal: {plaintext}\n")

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

result = model.decrypt(ciphertext_indices, DummyDataset())
result2 = model2.decrypt(ciphertext_indices, DummyDataset())
print(f"Decrypted: {result}\n")
print(f"Decrypted2: {result2}\n")