import os
import random
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

from models.substitution_v8.lightning_module_v8 import SubstitutionCipherSolver

ALPHABET_CHARS = "abcçdefgğhıijklmnoöprsştuüvyz"
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
SPACE_TOKEN = " "

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, SPACE_TOKEN]
FULL_ALPHABET = SPECIAL_TOKENS + list(ALPHABET_CHARS)
CHECKPOINT_PATH = "checkpoints_v7.4/substitution-epoch=47-val_gen_acc=0.930.ckpt"

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

    start = time.time()
    predicted_tensor = model.model.generate_consistent(input_tensor) 
    
    decoded_text = decode_tensor(predicted_tensor)
    end = time.time()
    print(f"Çalışma süresi: {end- start}")
    
    print(f"Model Tahmini:  {decoded_text}")

if __name__ == "__main__":
    test_text_1 = "bilim tarihi sürecinde bu tip sahnelere sürekli tanık olmuş deney ve gözlem sonucunda çöken kanunların yerini başkaları almıştır gerçek ve varlığın amacını soruşturan felsefe sistematik düşünmeyi gerektirmektedir din odaklı orta çağ felsefesinde hristiyanlığın kendisine bir aracı olarak kullandığı felsefe tanrı bilgi inanç eksenlerinde yoğun şekilde kullanılmıştır aydınlanma çağında yapılan felsefede akıl ön plana çıkmıştır"
    test_text_2 = "türkiye avrupa ve asya kıtalarında yer alan bir ülkedir başkenti ankara olup en büyük şehri istanbul dır osmanlı imparatorluğunun yıkılmasının ardından mustafa kemal atatürk önderliğinde cumhuriyet kurulmuştur ülke demokratik parlamenter bir sisteme sahiptir"
    test_text_3 = "bir çok insan için hayat çok güzel olabilir ancak bazı insanlar için bu durum böyle değildir her gün yeni bir şeyler öğrenmek ve kendimizi geliştirmek çok önemlidir bunun için kitap okumak spor yapmak ve arkadaşlarımızla vakit geçirmek gerekir bu sayede daha mutlu olabiliriz"
    test_text_4 = "bugün hava çok güzeldi arkadaşlarımla parka gittik orada çok eğlendik futbol oynadık ve piknik yaptık akşam eve döndükten sonra biraz dinlendim daha sonra televizyon izledim yarın yine dışarı çıkmayı planlıyorum belki sinemaya gideriz veya alışveriş yaparız henüz karar vermedik ama mutlaka güzel bir gün olacak"
    test_text_5 = "dün akşam eve geldiğimde çok yorgundum bütün gün çalışmıştım biraz dinlendikten sonra yemek yedim sonra kitap okumaya başladım roman çok ilginçti sabaha kadar uyuyamadım bugün işe giderken çok uykum vardı ama yine de işlerimi hallettim akşam erken yatacağım yarın için enerjik olmak istiyorum çünkü önemli bir toplantım var"
    test_text_6 = "düşünce sistemindeki temel görüş insan aklının aydınlattığı kesin doğrulara ve bilgiye doğru ilerlemektir"
    test_text_7 = "dünya üzerinde milyonlarca farklı canlı türü yaşamaktadır her biri ekosistemin önemli bir parçasıdır bitkiler hayvanlar mikroorganizmalar ve mantarlar doğal dengeyi sağlar ormanlar okyanuslar nehirler ve göller biyoçeşitliliğin korunduğu alanlardır iklim değişikliği ve insan faaliyetleri bu dengeyi tehdit etmektedir sürdürülebilir yaşam için doğayı korumak gerekir geri dönüşüm enerji tasarrufu ve bilinçli tüketim önemlidir"
    test_text_8 = "küçük bir kasabada yaşayan genç bir kız her gün deniz kenarında yürüyüş yapmayı severdi dalgaların sesi ona huzur verirdi bazen saatlerce oturup ufku seyrederdi hayallerini kurardı bir gün büyük şehre gidip yazar olmak istiyordu hikayeleri çok güzeldi arkadaşları ona hep destek olurdu ailesi de onu destekliyordu günlerden bir gün büyük bir yarışma kazandı ve ödül olarak kitabını yayınlama fırsatı buldu çok mutluydu"
    test_text_9 = "öğrenciler üniversiteye başlamadan önce çok çalışmalıdır güzel notlar almak için düzenli çalışmak şarttır öğretmenler öğrencilere yardımcı olur bütün dersler önemlidir coğrafya müzik beden eğitimi gibi dersler de çok öğreticidir böylece öğrenciler çok şey öğrenmiş olur üniversitede çok güzel arkadaşlıklar kurulur öğrenci kulüpleri çeşitli etkinlikler düzenler böylece öğrenciler sosyalleşir ve kendilerini geliştirir"
    test_text_10 = "köyün en yaşlısı olan bilge cüce tüm cüceleri bir araya toplar ve onlara huzuru korumanın ne kadar önemli olduğu konusunda konuşmalar yaparmış günlerden bir gün diyarın tüm ışığı kaybolmuş cüceler ne yapacaklarını bilmez bir haldelermiş içlerinden en cesur olan cüceyi ışığı geri getirme konusunda görevlendirmişler ve mağaranın dışına çıkarmışlar"
    test_text_11 = "cengiz han doğum adıyla temuçin ağustos moğol imparatorluğunun kurucusu ve ilk kağanı olan moğol komutan ve hükümdardır hükümdarlığı döneminde gerçekleştirdiği hiçbir savaşı kaybetmeyen cengiz han dünya tarihinin en büyük askeri liderlerinden birisi olarak kabul edilmektedir"
    test_text_12 = "hayatın en önemli gerçeği samimiliktir bu itibarla hayat ile bağı olan edebiyat mutlaka samimi bir edebiyattır denilebilir hayatı en gizli en karışık yönleriyle anlatmayan duygularımızı tıpkı hayatta olduğu gibi saf ve derin bir şekilde duyurmayan elemlerimizi felaketlerimizi açık açık yansıtmayan bir edebiyat hayat ile ilgisiz ve sahte bir edebiyattır"
    test_text_13 = "insan elinde ne illet var ki dokunduğunu değiştiriyor kendiliğinden iyi ve güzel olan şeyleri bozuyor iyi olmak arzusu bazen öyle azgın bir tutku oluyor ki iyi olalım derken kötü oluyoruz bazıları der ki iyinin aşırısı olmaz çünkü aşırı oldu mu zaten iyi değil demektir kelimelerle oynamak diyeceği gelir insanın buna felsefenin böyle ince oyunları vardır"
    test_text_14 = "güneşli ve sıcak bir yaz günüydü şehrin meydanındaki saat öğle sonu ikiyi gösteriyordu k nin dağ köylerinden on on bir yaşlarında iri kara gözlü işlemeli sarı bir mintanla şayak bir şalvar giyinmiş genç irisi bir çocuk elinde tabancası herkesin şaşkın bakışları arasından hükümet konağına doğru koşuyordu çocuğun iki üç yüz metre kadar gerisinden babasıyla amcası onu hızlı adımlarla izliyorlardı"
    test_text_15 = "o günü bir vaka gibi bütün canlılığıyla pek yakından hatırlarım büyük merasim salonu bayraklarla süslenmişti biz talebe elbisesini son defa giyinmiş ve bu salonda sıra sıra yerlerimizi almıştık hazırlık kıtası ve mektep devresi olarak ihtiyat zabit mektebinde geçirdiğimiz bir sene şimdi yapılacak merasimle sona ermiş olacaktı"
    
    try:
        run_inference(test_text_14)
    except FileNotFoundError:
        print("\nDosya bulunamadı")
    except Exception as e:
        print(f"\nHata: {e}")