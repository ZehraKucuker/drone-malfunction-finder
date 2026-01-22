"""
Ses Dosyalarından Mel-Spektrogram Görüntüsü Oluşturma
====================================================
Bu script, ses dosyalarını Mel spektrogram görüntülerine dönüştürür.
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


class SpectrogramConfig:
    """Spektrogram oluşturma konfigürasyonu"""
    SAMPLE_RATE = 22050
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 20
    FMAX = 8000
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    DPI = 100
    COLORMAP = 'magma'


class AudioToSpectrogram:
    """Ses dosyalarını mel-spektrogram görüntülerine dönüştürücü"""
    
    def __init__(self):
        self.config = SpectrogramConfig()
    
    def convert_audio_to_spectrogram(self, audio_path: str, output_path: str) -> bool:
        """
        Ses dosyasını mel-spektrogram görüntüsüne dönüştür
        
        Args:
            audio_path: Ses dosyası yolu
            output_path: Çıktı görüntü yolu
            
        Returns:
            bool: Başarılı ise True
        """
        try:
            # Ses dosyasını yükle
            y, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE)
            
            # Mel spektrogram hesapla
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=self.config.N_FFT,
                hop_length=self.config.HOP_LENGTH,
                n_mels=self.config.N_MELS,
                fmin=self.config.FMIN,
                fmax=self.config.FMAX
            )
            
            # dB'ye dönüştür
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Çıktı dizinini oluştur
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
            # Figure oluştur
            fig, ax = plt.subplots(figsize=(
                self.config.IMAGE_WIDTH / self.config.DPI,
                self.config.IMAGE_HEIGHT / self.config.DPI
            ), dpi=self.config.DPI)
            
            # Spektrogramı çiz
            librosa.display.specshow(
                mel_spec_db,
                sr=sr,
                hop_length=self.config.HOP_LENGTH,
                x_axis='time',
                y_axis='mel',
                cmap=self.config.COLORMAP,
                ax=ax,
                fmin=self.config.FMIN,
                fmax=self.config.FMAX
            )
            
            # Eksenleri kaldır
            ax.axis('off')
            ax.set_frame_on(False)
            plt.tight_layout(pad=0)
            
            # Kaydet
            plt.savefig(
                output_path,
                format='jpg',
                dpi=self.config.DPI,
                bbox_inches='tight',
                pad_inches=0,
                facecolor='black'
            )
            plt.close(fig)
            
            # Görüntüyü yeniden boyutlandır
            img = Image.open(output_path)
            img = img.resize(
                (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT),
                Image.Resampling.LANCZOS
            )
            img.save(output_path, 'JPEG', quality=95)
            
            return True
            
        except Exception as e:
            print(f"Dönüştürme hatası: {audio_path} -> {output_path} - {e}")
            return False

