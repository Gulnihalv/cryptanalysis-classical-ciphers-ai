import torch
from vigenere_klp import VigenereKeyLengthCNN


model = VigenereKeyLengthCNN.load_from_checkpoint("checkpoints_length_predictor/predictor-epoch=57-val_acc=0.827-val_top2_acc=0.952.ckpt").eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

alphabet = "abcçdefgğhıijklmnoöprsştuüvyz"
char2int = {c: i for i, c in enumerate(list(alphabet))}
int2char = {i: c for i, c in enumerate(list(alphabet))}

plaintext = "güneşli ve sıcak bir yaz günüydü şehrin meydanındaki s"
plaintext = plaintext.replace(" ", "")
key = [2, 5, 6, 12, 3]
key_len_val = len(key)

ciphertext_indices = []
vocab_size = len(alphabet)

print(f"Orijinal: {plaintext}")

cleaned_plain = [c for c in plaintext if c in char2int or c == " "]
key_ptr = 0

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

def find_key_length_math(cipher_tensor, max_k=12):
    L = len(cipher_tensor)
    scores = []
    
    for k in range(1, max_k + 1):
        if L <= k:
            scores.append(0.0)
            continue
        
        shifted = cipher_tensor[k:]
        original = cipher_tensor[:-k]
        
        matches = (shifted == original).sum().float()
        score = (matches / len(shifted)).item()
        scores.append(score)
    
    best_k = scores.index(max(scores)) + 1
    
    scores_copy = scores.copy()
    scores_copy[best_k - 1] = 0.0
    second_best_k = scores_copy.index(max(scores_copy)) + 1
    
    return best_k, second_best_k, scores

print(find_key_length_math(cipher_tensor))

print("--- UZUNLUK TAHMİNİ ---")
top2_lengths = model.predict_key_length(cipher_tensor, top_k=2)
print(f"Gerçek Uzunluk : {len(key)}")
print(f"Model 1. Tercih: {top2_lengths[0][0]}")
print(f"Model 2. Tercih: {top2_lengths[0][1]}")

print(f"Çözülen  : {top2_lengths}")