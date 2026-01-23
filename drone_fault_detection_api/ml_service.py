"""
DroneAI Sense - ML Service API
Flask/FastAPI tabanlı ML model servisi
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Model klasörünü path'e ekle
CURRENT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = CURRENT_DIR.parent / "drone_fault_detection_model"

print(f"📁 Model dizini: {MODEL_DIR}")
print(f"📁 Model dizini mevcut mu: {MODEL_DIR.exists()}")

if not MODEL_DIR.exists():
    # Alternatif path dene
    MODEL_DIR = CURRENT_DIR / ".." / "drone_fault_detection_model"
    MODEL_DIR = MODEL_DIR.resolve()
    print(f"📁 Alternatif model dizini: {MODEL_DIR}")

sys.path.insert(0, str(MODEL_DIR))

try:
    from audio_to_spectrogram import AudioToSpectrogram
    from drone_type import DroneTypeDetector
    from fault_type import FaultTypeDetector
    print("✅ Model modülleri başarıyla import edildi")
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    print(f"📁 sys.path: {sys.path[:3]}")
    raise

# FastAPI uygulaması
app = FastAPI(
    title="DroneAI Sense ML Service",
    description="Drone ses analizi için ML model servisi",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global değişkenler
spectrogram_converter = None
drone_detector = None
fault_detector = None


def load_models():
    """Modelleri yükle"""
    global spectrogram_converter, drone_detector, fault_detector
    
    print("🔄 Modeller yükleniyor...")
    
    # Spektrogram dönüştürücü
    spectrogram_converter = AudioToSpectrogram()
    print("✅ Spektrogram dönüştürücü hazır")
    
    # Drone tipi dedektörü
    drone_detector = DroneTypeDetector()
    print("✅ Drone tipi modeli yüklendi")
    
    # Arıza tipi dedektörü
    fault_detector = FaultTypeDetector()
    print("✅ Arıza tipi modeli yüklendi")
    
    print("🚀 Tüm modeller başarıyla yüklendi!")


@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında modelleri yükle"""
    load_models()


@app.get("/")
async def root():
    """Ana sayfa"""
    return {
        "service": "DroneAI Sense ML Service",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Sağlık kontrolü"""
    return {
        "status": "healthy",
        "models_loaded": all([spectrogram_converter, drone_detector, fault_detector])
    }


@app.get("/model-info")
async def model_info():
    """Model bilgileri"""
    return {
        "drone_type": drone_detector.get_model_info() if drone_detector else None,
        "fault_type": fault_detector.get_model_info() if fault_detector else None
    }


@app.post("/analyze")
async def analyze_audio(audio: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Ses dosyasını analiz et
    
    Args:
        audio: Yüklenen ses dosyası (.wav, .mp3, vb.)
    
    Returns:
        Analiz sonuçları
    """
    start_time = time.time()
    
    # Modellerin yüklendiğini kontrol et
    if not all([spectrogram_converter, drone_detector, fault_detector]):
        raise HTTPException(status_code=503, detail="Modeller henüz yüklenmedi")
    
    # Desteklenen formatları kontrol et
    allowed_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
    file_ext = Path(audio.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Desteklenmeyen dosya formatı. İzin verilen: {allowed_extensions}"
        )
    
    try:
        # Geçici dosyaya kaydet
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_audio_path = tmp_file.name
        
        # Geçici spektrogram dosyası
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_spec:
            tmp_spec_path = tmp_spec.name
        
        try:
            # Ses -> Spektrogram dönüşümü
            spectrogram_converter.convert_audio_to_spectrogram(tmp_audio_path, tmp_spec_path)
            
            # Drone tipi tahmini
            drone_result = drone_detector.predict(tmp_spec_path)
            
            # Arıza tipi tahmini
            fault_result = fault_detector.predict(tmp_spec_path)
            
            # Çıkarım süresi
            inference_time = round((time.time() - start_time) * 1000, 2)  # ms
            
            # Sonuçları birleştir
            result = {
                "success": True,
                "filename": audio.filename,
                "drone_type": drone_result.get("drone_type"),
                "drone_confidence": drone_result.get("confidence", 0),
                "drone_probabilities": drone_result.get("all_probabilities", {}),
                "fault_type": fault_result.get("fault_type"),
                "fault_confidence": fault_result.get("confidence", 0),
                "fault_probabilities": fault_result.get("all_probabilities", {}),
                "is_faulty": fault_result.get("is_faulty", False),
                "fault_description": fault_result.get("description", ""),
                "severity": fault_result.get("severity", 0),
                "inference_time": inference_time
            }
            
            return result
            
        finally:
            # Geçici dosyaları temizle
            if os.path.exists(tmp_audio_path):
                os.unlink(tmp_audio_path)
            if os.path.exists(tmp_spec_path):
                os.unlink(tmp_spec_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")


@app.post("/analyze-spectrogram")
async def analyze_spectrogram(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Hazır spektrogram görüntüsünü analiz et
    
    Args:
        image: Spektrogram görüntüsü (.jpg, .png)
    
    Returns:
        Analiz sonuçları
    """
    start_time = time.time()
    
    # Modellerin yüklendiğini kontrol et
    if not all([drone_detector, fault_detector]):
        raise HTTPException(status_code=503, detail="Modeller henüz yüklenmedi")
    
    try:
        # Geçici dosyaya kaydet
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            content = await image.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Drone tipi tahmini
            drone_result = drone_detector.predict(tmp_path)
            
            # Arıza tipi tahmini
            fault_result = fault_detector.predict(tmp_path)
            
            # Çıkarım süresi
            inference_time = round((time.time() - start_time) * 1000, 2)
            
            return {
                "success": True,
                "drone_type": drone_result.get("drone_type"),
                "drone_confidence": drone_result.get("confidence", 0),
                "fault_type": fault_result.get("fault_type"),
                "fault_confidence": fault_result.get("confidence", 0),
                "is_faulty": fault_result.get("is_faulty", False),
                "inference_time": inference_time
            }
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "ml_service:app",
        host="0.0.0.0",
        port=5000,
        reload=True
    )
