import torch
import torch.nn as nn

class SubstitutionLSTM(nn.Module):
    def __init__(self, vocab_size=33, embed_dim=128, hidden_size=256):
        """not: vocab_size dediğim 29 + 4 yani modele girdi özel karakterler dahil karakter sayısı."""
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        #encoder: Bu geleceği görebilir BiLSTM (bidirectional olan kısım)
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=2, # 1 yeterli gelmediği için 2 ye çıkarrtım
            dropout= 0.2, # yeni eklendi
            batch_first=True,
            bidirectional=True
        )

        # decoder girdisi için encoder çıktısı lazım. (hidden*2) + o anki harfin embed_dim'i
        decoder_input_size = (hidden_size * 2) + embed_dim

        # decoder geleceği göremiyor UniLSTM
        self.decoder = nn.LSTM(
            input_size= decoder_input_size,
            hidden_size=hidden_size,
            batch_first= True,
            bidirectional=False # geleceği görememesi için
        )

        fc_input_dim = hidden_size + (hidden_size * 2) # Artık sadece hidden size (Decoder Hidden) + (Encoder Hidden * 2) alıcak
        # output
        self.fc = nn.Linear(fc_input_dim, vocab_size)

    def forward(self, src, tgt_input):
        """Eğitim sırasında kullanılacak. Burda hem src (bu şifreli veriye karşılık geliyor) hem de teacher input veriliyor"""    
        # iki girdiyi de hem source hem de teacher girdisini embedding'e çeviriyorum.
        src_embedding = self.embedding(src) # girdi: [batch, seq], çıktı:[batch, seq, embed]
        tgt_embedding = self.embedding(tgt_input)

        encoder_out, _ = self.encoder(src_embedding) # src'yi encode ediyoruz. output: [batch, seq, hidden*2]
        
        # iki bilgiyi birleştiriyoruz
        combined = torch.cat((encoder_out, tgt_embedding), dim=2) # output: [batch, seq, hidden*2+embed]

        #decoder
        decoder_out, _ = self.decoder(combined) #tuple dönüyor ama biz sadece output kısmını alıyoruz. hidden_state almıyoruz 512 +128 = 640

        # decoder çıktısı ile encoder yani şifreli veri birleştiriliyor
        final_input = torch.cat((decoder_out, encoder_out), dim=2)

        #logits
        logits = self.fc(final_input) # output: [batch, seq, vocab] burda artık karakter çıkıyor(vocab)
        return logits

    def generate(self, src):
        """Bu method inference için kullanılacak. Sadece src modele veriliyor. Yani şifreli veri"""
        src_embedding = self.embedding(src)
        encoder_out, _ = self.encoder(src_embedding) # sadece [batch_size, seq_len, hidden* 2 ] kısmını alıyoruz

        batch_size = src.size(0)
        sos_id = 1 # sonradan constantstan alıcam
        input_t = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=src.device) # [batch, 1]
        hidden = None
        max_len = src.shape[1]
        output = [] #çıkan sonucu burada saklayacağız

        with torch.no_grad():
            for i in range(max_len):
                encoder_step = encoder_out[:, i, :] # [batch, hidden *2]
                encoder_step = encoder_step.unsqueeze(1) # [batch, 1, hidden*2]

                tgt_embed = self.embedding(input_t) # [batch, 1, embed]

                decoder_input = torch.cat((encoder_step, tgt_embed), dim=2)
                decoder_out, hidden = self.decoder(decoder_input, hidden) # hidden başlangıçta None.

                final_input = torch.cat((decoder_out, encoder_step), dim=2)
                logits = self.fc(final_input) # [batch, 1, vocab_size]
                best = torch.argmax(logits, dim=2) # [batch, 1]
                output.append(best)

                eos_id = 2
                if (best == eos_id).all(): # 1den fazla batch olursa çalışmıyor ileride düzeltilebilir.
                    break

                input_t = best

        return torch.cat(output, dim=1)
    