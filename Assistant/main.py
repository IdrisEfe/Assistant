'''
Yapay Zeka Tabanlı Sesli Asistan Geliştirme
# Ses dosyalarını işlemek için
pip install torchaudio
pip install soundfile

# Mikrofondan ses almak için
pip install pvrecorder
'''

import torch
import torchaudio
import matplotlib.pyplot as plt

# Ön işlemeli veri modelimizi alalım

from transformers import AutoProcessor, AutoModelForCTC

# Modelimizin ön işleme bileşenlerini alalım

islem = AutoProcessor.from_pretrained('m3hrdadfi/wav2vec2-large-xlsr-turkish')

# Model dahil etme

model = AutoModelForCTC.from_pretrained('m3hrdadfi/wav2vec2-large-xlsr-turkish')

# Mikrofon üzerinden sesli komut dinlenilmesi

from pvrecorder import PvRecorder

# wav formatında, okuma-yazma işlemler için gerekli kütüphane

import wave

# Ses dosyaları ikili format:
    
import struct

# Ses kaydedici bir nesne oluşturalım

recorder=PvRecorder(device_index=0,  frame_length=512)# Mikrofon seçme
                      
'''
recorder.start(): sesi kaydet
recorder.stop(): sesi durdur
recorder.delete(): sesi sil
recorder.read(): sesli oku
'''

# seslerin kaydedileceği liste

audio = []

# Ses alma bloğu

try:
    recorder.start()
    while True:
        print('Dinliyorum...')
        frame = recorder.read() 
        audio.extend(frame)
    
except KeyboardInterrupt: # Ctrl + C
    recorder.stop()
    with wave.open('ses.wav','w') as f:
        f.setparams((1, # Kaç kaynak
                     2, # Her bir örnekten 2 byte
                     16000, # Frekans
                     512, 'NONE', 'NONE')) # Sesin fiziksel özellikleri
        f.writeframes(struct.pack('h'*len(audio), *audio)) # Gelecek olan ses için yer ayarlıyor
        
        
finally:
    recorder.delete()

# Ses dosya eşleme (Kliket olayı): resampled örneği

waveform, sample_rate = torchaudio.load('ses.wav')
waveform_resampled = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq = 16000)(waveform)

# Ses dosyasının görselleştirilmesi

plt.figure(figsize=(10,6))
plt.subplot(2, 1, 1)

plt.plot(waveform.t().numpy())

plt.title('Ses dalgasının dalgaform grafiği')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)

spe = torchaudio.transforms.Spectrogram()(waveform_resampled)

spe_ch = spe[0, :, :]
plt.imshow(spe_ch.log2().numpy(), aspect='auto', cmap = 'viridis')

plt.title('Spektogram')
plt.xlabel('Zaman')
plt.ylabel('Frekans')

plt.tight_layout()
plt.show()

# Modelden tahmin alma

with torch.no_grad():
    logits = model(waveform_resampled).logits
    
output = torch.argmax(logits, dim = -1)
command = islem.batch_decocode(output)

print('Komutunuz:', command)
