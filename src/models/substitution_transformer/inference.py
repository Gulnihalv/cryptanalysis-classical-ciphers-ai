import os
import random
import sys
import torch
import time
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
src_path = os.path.abspath(os.path.join(current_dir, '../../'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from substitution_lightining import SubstitutionCipherSolverV9

ALPHABET_CHARS = "abcçdefgğhıijklmnoöprsştuüvyz"
PAD_TOKEN  = "<PAD>"
SOS_TOKEN  = "<SOS>"
EOS_TOKEN  = "<EOS>"
SPACE_TOKEN = " "

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, SPACE_TOKEN]
FULL_ALPHABET  = SPECIAL_TOKENS + list(ALPHABET_CHARS)
CHECKPOINT_PATH = "checkpoint_subs_t/best-epoch=59-val_acc=0.9811.ckpt"

char2idx = {c: i for i, c in enumerate(FULL_ALPHABET)}
idx2char  = {i: c for i, c in enumerate(FULL_ALPHABET)}

PAD_IDX = char2idx[PAD_TOKEN]
SEQ_LEN = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def encrypt_text(plain_text: str):
    real_chars     = list(ALPHABET_CHARS)
    shuffled_chars = list(ALPHABET_CHARS)
    random.shuffle(shuffled_chars)
    key_map = {s: t for s, t in zip(real_chars, shuffled_chars)}
    cipher_text = plain_text.translate(str.maketrans(key_map))
    return cipher_text, key_map


def text_to_padded_tensor(text: str) -> torch.Tensor:
    """Metni SEQ_LEN uzunluğunda padding'li tensor'a çevirir."""
    indices = [char2idx.get(c, PAD_IDX) for c in text]
    indices = indices[:SEQ_LEN]
    indices += [PAD_IDX] * (SEQ_LEN - len(indices))
    return torch.tensor(indices, dtype=torch.long)


def decode_tensor(pred_tensor: torch.Tensor, src_tensor: torch.Tensor) -> str:
    """
    Tahmin tensor'ını stringe çevirir.
    src_tensor'daki padding başladığı anda durur — 'zzz...' sorununu önler.
    """
    preds = pred_tensor.squeeze().tolist()
    srcs  = src_tensor.squeeze().tolist()

    decoded = []
    for pred_idx, src_idx in zip(preds, srcs):
        if src_idx == PAD_IDX:
            break                          # src padding → dur
        char = idx2char.get(pred_idx, "")
        if char in (PAD_TOKEN, EOS_TOKEN):
            break
        if char == SOS_TOKEN:
            continue
        decoded.append(char)

    return "".join(decoded)


def sliding_window_inference(model, cipher_text: str, stride: int = 200) -> str:
    n = len(cipher_text)

    # Kısa metin: tek pencere, doğrudan işle
    if n <= SEQ_LEN:
        src_tensor = text_to_padded_tensor(cipher_text).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_tensor = model.model.generate(src_tensor)
        return decode_tensor(pred_tensor, src_tensor)

    # Uzun metin: sliding window + global voting
    # votes[cipher_char_idx][plain_char_idx] = toplam oy sayısı
    votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    start = 0
    while start < n:
        end    = min(start + SEQ_LEN, n)
        window = cipher_text[start:end]

        src_tensor = text_to_padded_tensor(window).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_tensor = model.model.generate(src_tensor)  # [1, SEQ_LEN]

        src_list  = src_tensor.squeeze().tolist()
        pred_list = pred_tensor.squeeze().tolist()

        for c_idx, p_idx in zip(src_list, pred_list):
            if c_idx == PAD_IDX or c_idx < 4:  # padding veya özel token → atla
                continue
            votes[c_idx][p_idx] += 1

        if end == n:
            break
        start += stride

    # Global majority vote → one-to-one mapping
    final_map: dict[int, int] = {}
    used: set[int] = set()

    # En sık geçen cipher char'ları önce işle (daha güvenilir vote'ları var)
    sorted_cipher = sorted(votes.keys(), key=lambda k: sum(votes[k].values()), reverse=True)

    for c_idx in sorted_cipher:
        candidates = sorted(votes[c_idx].items(), key=lambda x: x[1], reverse=True)
        for p_idx, _ in candidates:
            if p_idx not in used:
                final_map[c_idx] = p_idx
                used.add(p_idx)
                break

    # Mapping'i metne uygula
    result = []
    for char in cipher_text:
        c_idx = char2idx.get(char, PAD_IDX)
        if c_idx in final_map:
            result.append(idx2char.get(final_map[c_idx], char))
        else:
            result.append(char)  # mapping yoksa orijinali koy

    return "".join(result)

def run_inference(model, input_text: str):
    cipher_text, key_map = encrypt_text(input_text)

    print(f"\nOrjinal Metin : {input_text}")
    print(f"Şifreli Metin : {cipher_text}")
    print(f"Metin uzunluğu: {len(cipher_text)} karakter")

    start = time.time()
    decoded_text = sliding_window_inference(model, cipher_text)
    elapsed = time.time() - start

    print(f"Çalışma süresi: {elapsed:.4f}s")
    print(f"Model Tahmini : {decoded_text}")

    # Doğruluk hesapla (karakter bazlı)
    correct = sum(a == b for a, b in zip(input_text, decoded_text))
    total   = len(input_text)
    print(f"Karakter doğruluğu: {correct}/{total} = %{100*correct/total:.1f}")
    print("-" * 80)

if __name__ == "__main__":
    # Modeli bir kez yükle
    print(f"Model yükleniyor... ({DEVICE})")
    model = SubstitutionCipherSolverV9.load_from_checkpoint(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )
    model.eval()
    model.freeze()
    print("Model hazır.\n" + "=" * 80)

    test_texts = {
        "kısa (106 kar)"  : "düşünce sistemindeki temel görüş insan aklının aydınlattığı kesin doğrulara ve bilgiye doğru ilerlemektir",
        "orta (261 kar)"  : "türkiye avrupa ve asya kıtalarında yer alan bir ülkedir başkenti ankara olup en büyük şehri istanbul dır osmanlı imparatorluğunun yıkılmasının ardından mustafa kemal atatürk önderliğinde cumhuriyet kurulmuştur ülke demokratik parlamenter bir sisteme sahiptir",
        "uzun (500+ kar)" : "dünya üzerinde milyonlarca farklı canlı türü yaşamaktadır her biri ekosistemin önemli bir parçasıdır bitkiler hayvanlar mikroorganizmalar ve mantarlar doğal dengeyi sağlar ormanlar okyanuslar nehirler ve göller biyoçeşitliliğin korunduğu alanlardır iklim değişikliği ve insan faaliyetleri bu dengeyi tehdit etmektedir sürdürülebilir yaşam için doğayı korumak gerekir geri dönüşüm enerji tasarrufu ve bilinçli tüketim önemlidir",
        "çok uzun"        : "bilim tarihi sürecinde bu tip sahnelere sürekli tanık olmuş deney ve gözlem sonucunda çöken kanunların yerini başkaları almıştır gerçek ve varlığın amacını soruşturan felsefe sistematik düşünmeyi gerektirmektedir din odaklı orta çağ felsefesinde hristiyanlığın kendisine bir aracı olarak kullandığı felsefe tanrı bilgi inanç eksenlerinde yoğun şekilde kullanılmıştır aydınlanma çağında yapılan felsefede akıl ön plana çıkmıştır hayatın en önemli gerçeği samimiliktir bu itibarla hayat ile bağı olan edebiyat mutlaka samimi bir edebiyattır",
    }

    for label, text in test_texts.items():
        print(f"\n>>> {label}")
        run_inference(model, text)