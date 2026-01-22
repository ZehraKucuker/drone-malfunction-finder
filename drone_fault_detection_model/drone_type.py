"""
Drone Tipi Tespit Script'i
===========================
Ses spektrogram görüntüsünden drone tipini tespit eder.
Drone Tipleri: Helikopter, Duokopter, Trikopter, Quadkopter
"""

import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Union
from io import BytesIO

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image


class DroneTypeConfig:
    """Drone tipi tespit konfigürasyonu"""
    MODEL_DIR = Path(__file__).parent / "models" / "drone_type"
    MODEL_PATH = MODEL_DIR / "drone_type_vit_model.pth"
    IMAGE_SIZE = 224
    VIT_MODEL = "vit_base_patch16_224"
    CLASSES = ["Helikopter", "Duokopter", "Trikopter", "Quadkopter"]
    NUM_CLASSES = 4
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DroneTypeClassifier(nn.Module):
    """Drone tipi sınıflandırıcı model"""
    def __init__(self, num_classes: int = DroneTypeConfig.NUM_CLASSES):
        super().__init__()
        self.vit = timm.create_model(
            DroneTypeConfig.VIT_MODEL,
            pretrained=False,
            num_classes=num_classes
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


class DroneTypeDetector:
    """Drone tipi tespit sınıfı"""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else DroneTypeConfig.MODEL_PATH
        self.device = torch.device(device) if device else DroneTypeConfig.DEVICE
        self.classes = DroneTypeConfig.CLASSES
        self.num_classes = DroneTypeConfig.NUM_CLASSES
        
        self.transform = transforms.Compose([
            transforms.Resize((DroneTypeConfig.IMAGE_SIZE, DroneTypeConfig.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=DroneTypeConfig.NORMALIZE_MEAN,
                std=DroneTypeConfig.NORMALIZE_STD
            )
        ])
        
        self.model = self._load_model()
    
    def _load_model(self) -> nn.Module:
        """Eğitilmiş modeli yükle"""
        model = DroneTypeClassifier(num_classes=self.num_classes)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        return model
    
    def _preprocess_image(self, image: Union[str, Path, Image.Image, bytes]) -> torch.Tensor:
        """Görüntüyü model girişi için hazırla"""
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            img = Image.open(BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise ValueError(f"Desteklenmeyen görüntü formatı: {type(image)}")
        
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def predict(self, image: Union[str, Path, Image.Image, bytes]) -> Dict[str, Any]:
        """
        Drone tipini tespit et
        
        Args:
            image: Spektrogram görüntüsü
        
        Returns:
            Dict: Tahmin sonuçları
        """
        try:
            img_tensor = self._preprocess_image(image)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = self.classes[predicted_idx.item()]
            confidence_score = confidence.item()
            
            all_probs = {
                cls: prob.item() 
                for cls, prob in zip(self.classes, probabilities[0])
            }
            
            return {
                "success": True,
                "drone_type": predicted_class,
                "confidence": round(confidence_score, 4),
                "all_probabilities": {k: round(v, 4) for k, v in all_probs.items()}
            }
            
        except Exception as e:
            return {
                "success": False,
                "drone_type": None,
                "confidence": 0.0,
                "all_probabilities": {},
                "error": str(e)
            }
    
    def predict_from_base64(self, base64_string: str) -> Dict[str, Any]:
        """Base64 formatındaki görüntüden drone tipini tespit et"""
        try:
            image_bytes = base64.b64decode(base64_string)
            return self.predict(image_bytes)
        except Exception as e:
            return {
                "success": False,
                "drone_type": None,
                "confidence": 0.0,
                "all_probabilities": {},
                "error": f"Base64 decode hatası: {str(e)}"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Model bilgilerini döndür"""
        return {
            "task": "drone_type_detection",
            "classes": self.classes,
            "num_classes": self.num_classes,
            "model_architecture": DroneTypeConfig.VIT_MODEL,
            "image_size": DroneTypeConfig.IMAGE_SIZE,
            "device": str(self.device),
            "model_path": str(self.model_path)
        }

