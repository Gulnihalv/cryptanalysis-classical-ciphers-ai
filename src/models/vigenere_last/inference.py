import os
import sys
import torch
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
src_path = os.path.abspath(os.path.join(current_dir, '../../'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from vigenere import VigenereBlindSolver

# --- AYARLAR ---
ALPHABET_CHARS = "abcçdefgğhıijklmnoöprsştuüvyz"
CHECKPOINT_PATH = "checkpoints_vigenere/last.ckpt" # Kendi model yolunu buraya yaz

char2idx = {c: i for i, c in enumerate(ALPHABET_CHARS)}
idx2char = {i: c for i, c in enumerate(ALPHABET_CHARS)}
PAD_IDX = len(ALPHABET_CHARS) # Vocab size (29) pad_idx'tir
MAX_LEN = 512 # Modelin maksimum desteklediği uzunluk

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def encrypt_vigenere(plain_text: str, key: str):
    """Metni Vigenere ile şifreler (Sadece alfabedeki harfleri tutar)."""
    # Sadece alfabede olanları filtrele (Eğitim verisi gibi)
    clean_text = "".join([c for c in plain_text.lower() if c in char2idx])
    
    cipher_text = []
    key_indices = [char2idx[k] for k in key]
    
    for i, char in enumerate(clean_text):
        p_val = char2idx[char]
        k_val = key_indices[i % len(key_indices)]
        c_val = (p_val + k_val) % len(ALPHABET_CHARS)
        cipher_text.append(idx2char[c_val])
        
    return "".join(cipher_text), clean_text

def run_inference(model, input_text: str, test_key: str):
    # Metni şifrele ve temizle
    cipher_text, clean_text = encrypt_vigenere(input_text, test_key)
    
    # Modelin sınırı olan 512 karakteri aşmamak için kes
    cipher_text = cipher_text[:MAX_LEN]
    clean_text = clean_text[:MAX_LEN]

    print(f"Orjinal Metin : {clean_text[:100]}...")
    print(f"Şifreli Metin : {cipher_text[:100]}...")
    print(f"Metin uzunluğu: {len(cipher_text)} karakter")
    print(f"Kullanılan Anahtar: '{test_key}' (Uzunluk: {len(test_key)})")

    # Tensora çevir
    indices = [char2idx[c] for c in cipher_text]
    src_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE)

    start = time.time()
    with torch.no_grad():
        result = model.decode(src_tensor)
    elapsed = time.time() - start

    pred_indices = result["plaintext"].tolist()
    decoded_text = "".join([idx2char.get(idx, "") for idx in pred_indices])
    
    pred_key_len = result["best_key_len"]

    print(f"Çalışma süresi: {elapsed:.4f}s")
    print(f"Tahmin Edilen Key Uzunluğu: {pred_key_len}")
    print(f"Model Tahmini : {decoded_text[:100]}...")

    # Doğruluk hesapla (karakter bazlı)
    correct = sum(a == b for a, b in zip(clean_text, decoded_text))
    total   = len(clean_text)
    print(f"Karakter doğruluğu: {correct}/{total} = %{100*correct/total:.1f}")
    print("-" * 80)

if __name__ == "__main__":
    # Modeli bir kez yükle
    print(f"Model yükleniyor... ({DEVICE})")
    model = VigenereBlindSolver.load_from_checkpoint(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )
    model.to(DEVICE)
    model.eval()
    model.freeze()
    print("Model hazır.\n" + "=" * 80)

    # Senin verdiğin test metinleri
    test_texts = {
        "kısa (106 kar)"  : "düşünce sistemindeki temel görüş insan aklının aydınlattığı kesin doğrulara ve bilgiye doğru ilerlemektir",
        "orta (261 kar)"  : "türkiye avrupa ve asya kıtalarında yer alan bir ülkedir başkenti ankara olup en büyük şehri istanbul dır osmanlı imparatorluğunun yıkılmasının ardından mustafa kemal atatürk önderliğinde cumhuriyet kurulmuştur ülke demokratik parlamenter bir sisteme sahiptir",
        "uzun (300+ kar)" : "dünya üzerinde milyonlarca farklı canlı türü yaşamaktadır her biri ekosistemin önemli bir parçasıdır bitkiler hayvanlar mikroorganizmalar ve mantarlar doğal dengeyi sağlar ormanlar okyanuslar nehirler ve göller biyoçeşitliliğin korunduğu alanlardır iklim değişikliği ve insan faaliyetleri bu dengeyi tehdit etmektedir sürdürülebilir yaşam için doğayı korumak gerekir geri dönüşüm enerji tasarrufu ve bilinçli tüketim önemlidir",
        "çok uzun (433 kar)": "bilim tarihi sürecinde bu tip sahnelere sürekli tanık olmuş deney ve gözlem sonucunda çöken kanunların yerini başkaları almıştır gerçek ve varlığın amacını soruşturan felsefe sistematik düşünmeyi gerektirmektedir din odaklı orta çağ felsefesinde hristiyanlığın kendisine bir aracı olarak kullandığı felsefe tanrı bilgi inanç eksenlerinde yoğun şekilde kullanılmıştır aydınlanma çağında yapılan felsefede akıl ön plana çıkmıştır hayatın en önemli gerçeği samimiliktir",
    }

    test_key = "ankara" # Test için kullanılacak şifreleme anahtarı (6 harfli)

    for label, text in test_texts.items():
        print(f"\n>>> {label}")
        run_inference(model, text, test_key)