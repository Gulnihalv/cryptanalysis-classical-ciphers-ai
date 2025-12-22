S_CONFIG_V1 = {
    'text_path': 'data/processed/wiki_processed_tr2.txt',     # Veri setinin yolu
    'alphabet': "abcçdefgğhıijklmnoöprsştuüvyz", # Şifrelenecek karakterler
    'seq_len': 250,                   # Chunk uzunluğu
    'batch_size': 32,
    'hidden_size': 256,
    'embed_dim': 128,
    'lr': 0.001,
    'max_epochs': 40,
    'num_workers': 4,                 # CPU çekirdek sayısı (Hız için kritik!)
    'val_split': 0.1                  # Verinin %10'u doğrulama için
}

special_tokens = ["<PAD>", "<SOS>", "<EOS>", " "]
alphabet = list("abcçdefgğhıijklmnoöprsştuüvyz")

# buraya special tokenlar ve listesi vs de gelecek