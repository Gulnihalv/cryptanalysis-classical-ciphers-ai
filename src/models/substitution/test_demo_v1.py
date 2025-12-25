import os
import random
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
src_path = os.path.abspath(os.path.join(current_dir, '../../'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.substitution.lightning_module import SubstitutionCipherSolver

ALPHABET_CHARS = "abcçdefgğhıijklmnoöprsştuüvyz"
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
SPACE_TOKEN = " "

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, SPACE_TOKEN]
FULL_ALPHABET = SPECIAL_TOKENS + list(ALPHABET_CHARS)
CHECKPOINT_PATH = "checkpoints_v3/substitution-epoch=43-val_loss=0.20.ckpt"

# Haritalar
char2idx = {c: i for i, c in enumerate(FULL_ALPHABET)}
idx2char = {i: c for i, c in enumerate(FULL_ALPHABET)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def encrypt_text(plain_text):
    real_chars = list(ALPHABET_CHARS)
    shuffled_chars = list(ALPHABET_CHARS)
    random.shuffle(shuffled_chars)
    
    key_map = {src: target for src, target in zip(real_chars, shuffled_chars)}
    
    table = str.maketrans(key_map)
    cipher_text = plain_text.translate(table)
    
    return cipher_text, key_map

def decode_tensor(indices_tensor):
    indices = indices_tensor.squeeze().tolist()
    decoded_chars = []
    
    for idx in indices:
        char = idx2char.get(idx, "")
        
        if char == SOS_TOKEN: continue
        if char == PAD_TOKEN: continue
        if char == EOS_TOKEN: break
        
        decoded_chars.append(char)
        
    return "".join(decoded_chars)


def run_inference(input_text):
    model = SubstitutionCipherSolver.load_from_checkpoint(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )
    model.eval()
    model.freeze()

    cipher_text, key_map = encrypt_text(input_text)
    print(f"\nOrjinal Metin:  {input_text}\n")
    print(f"Şifreli Metin:  {cipher_text}\n")

    
    input_indices = [char2idx.get(c, char2idx[PAD_TOKEN]) for c in cipher_text]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)

    predicted_tensor = model.model.generate(input_tensor) 
    
    decoded_text = decode_tensor(predicted_tensor)
    
    print(f"Model Tahmini:  {decoded_text}")


if __name__ == "__main__":
    test_text_1 = "bilim tarihi sürecinde bu tip sahnelere sürekli tanık olmuş deney ve gözlem sonucunda çöken kanunların yerini başkaları almıştır gerçek ve varlığın amacını soruşturan felsefe sistematik düşünmeyi gerektirmektedir din odaklı orta çağ felsefesinde hristiyanlığın kendisine bir aracı olarak kullandığı felsefe tanrı bilgi inanç eksenlerinde yoğun şekilde kullanılmıştır aydınlanma çağında yapılan felsefede akıl ön plana çıkmıştır"
    test_text_2 = "türkiye avrupa ve asya kıtalarında yer alan bir ülkedir başkenti ankara olup en büyük şehri istanbul dır osmanlı imparatorluğunun yıkılmasının ardından mustafa kemal atatürk önderliğinde cumhuriyet kurulmuştur ülke demokratik parlamenter bir sisteme sahiptir"
    test_text_3 = "bir çok insan için hayat çok güzel olabilir ancak bazı insanlar için bu durum böyle değildir her gün yeni bir şeyler öğrenmek ve kendimizi geliştirmek çok önemlidir bunun için kitap okumak spor yapmak ve arkadaşlarımızla vakit geçirmek gerekir bu sayede daha mutlu olabiliriz"
    test_text_4 = "bugün hava çok güzeldi arkadaşlarımla parka gittik orada çok eğlendik futbol oynadık ve piknik yaptık akşam eve döndükten sonra biraz dinlendim sonra televizyon izledim yarın yine dışarı çıkmayı planlıyorum belki sinemaya gideriz veya alışveriş yaparız henüz karar vermedik ama mutlaka güzel bir gün olacak"
    test_text_5 = "dün akşam eve geldiğimde çok yorgundum bütün gün çalışmıştım biraz dinlendikten sonra yemek yedim sonra kitap okumaya başladım roman çok ilginçti sabaha kadar uyuyamadım bugün işe giderken çok uykum vardı ama yine de işlerimi hallettim akşam erken yatacağım yarın için enerjik olmak istiyorum çünkü önemli bir toplantım var"
    test_text_6 = "düşünce sistemindeki temel görüş insan aklının aydınlattığı kesin doğrulara ve bilgiye doğru ilerlemektir"
    test_text_7 = "dünya üzerinde milyonlarca farklı canlı türü yaşamaktadır her biri ekosistemin önemli bir parçasıdır bitkiler hayvanlar mikroorganizmalar ve mantarlar doğal dengeyi sağlar ormanlar okyanuslar nehirler ve göller biyoçeşitliliğin korunduğu alanlardır iklim değişikliği ve insan faaliyetleri bu dengeyi tehdit etmektedir sürdürülebilir yaşam için doğayı korumak gerekir geri dönüşüm enerji tasarrufu ve bilinçli tüketim önemlidir"
    test_text_8 = "küçük bir kasabada yaşayan genç bir kız her gün deniz kenarında yürüyüş yapmayı severdi dalgaların sesi ona huzur verirdi bazen saatlerce oturup ufku seyrederdi hayallerini kurardı bir gün büyük şehre gidip yazar olmak istiyordu hikayeleri çok güzeldi arkadaşları ona hep destek olurdu ailesi de onu destekliyordu günlerden bir gün büyük bir yarışma kazandı ve ödül olarak kitabını yayınlama fırsatı buldu çok mutluydu"
    test_text_9 = "öğrenciler üniversiteye başlamadan önce çok çalışmalıdır güzel notlar almak için düzenli çalışmak şarttır öğretmenler öğrencilere yardımcı olur bütün dersler önemlidir coğrafya müzik beden eğitimi gibi dersler de çok öğreticidir böylece öğrenciler çok şey öğrenmiş olur üniversitede çok güzel arkadaşlıklar kurulur öğrenci kulüpleri çeşitli etkinlikler düzenler böylece öğrenciler sosyalleşir ve kendilerini geliştirir"


    try:
        run_inference(test_text_9)
    except FileNotFoundError:
        print("\nDosya bulunamadı")
    except Exception as e:
        print(f"\nHata: {e}")