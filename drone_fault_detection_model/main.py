"""
Drone Motor Arıza Tespiti - Vision Transformer ile Eğitim Scripti
================================================================
Bu script, drone motor seslerinin spektrogram görüntülerinden:
1. Drone tipi sınıflandırması (Helikopter, Duokopter, Trikopter, Quadkopter)
2. Arıza tipi sınıflandırması (Normal, Balans, Mıknatıs, Pervane, Rulman)

yapan iki ayrı Vision Transformer modeli eğitir.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Logging ayarları (sadece terminal)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# KONFIGURASYON
# ============================================================================

class Config:
    """Eğitim konfigürasyonu"""
    # Veri seti yolu
    DATASET_PATH = Path("dataset")
    
    # Model kayıt dizini
    OUTPUT_DIR = Path("models")
    
    # Senaryo -> Drone tipi eşleştirmesi
    SCENARIO_TO_DRONE = {
        "Senaryo1": "Helikopter",   # 1 motor
        "Senaryo2": "Duokopter",    # 2 motor
        "Senaryo3": "Trikopter",    # 3 motor
        "Senaryo4": "Quadkopter",   # 4 motor
    }
    
    # Arıza tipleri
    FAULT_TYPES = ["Normal", "Balans", "Miknatis", "Pervane", "Rulman"]
    
    # Drone tipleri
    DRONE_TYPES = ["Helikopter", "Duokopter", "Trikopter", "Quadkopter"]
    
    # Eğitim parametreleri
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Veri bölme oranları
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Vision Transformer model seçimi
    VIT_MODEL = "vit_base_patch16_224"  # timm model ismi
    
    # Cihaz ayarı
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Random seed
    SEED = 42


# ============================================================================
# VERİ SETİ
# ============================================================================

class DroneDataset(Dataset):
    """Drone ses spektrogram görüntüleri için Dataset sınıfı"""
    
    def __init__(
        self, 
        root_dir: Path, 
        transform: Optional[transforms.Compose] = None,
        task: str = "drone_type"  # "drone_type" veya "fault_type"
    ):
        """
        Args:
            root_dir: Veri seti kök dizini
            transform: Görüntü dönüşümleri
            task: "drone_type" (drone tipi sınıflandırma) veya "fault_type" (arıza tipi sınıflandırma)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.task = task
        
        # Etiket eşleştirmeleri
        if task == "drone_type":
            self.classes = Config.DRONE_TYPES
        else:
            self.classes = Config.FAULT_TYPES
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Veri örneklerini yükle
        self.samples = self._load_samples()
        
        logger.info(f"Task: {task}, Toplam örnek: {len(self.samples)}")
        logger.info(f"Sınıflar: {self.classes}")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Tüm görüntü yollarını ve etiketlerini yükle"""
        samples = []
        
        for folder in self.root_dir.iterdir():
            if not folder.is_dir():
                continue
            
            folder_name = folder.name
            # Senaryo ve arıza tipini parse et (örn: "Senaryo1_Normal")
            parts = folder_name.split("_")
            if len(parts) != 2:
                continue
            
            scenario, fault_type = parts
            
            # Drone tipini belirle
            drone_type = Config.SCENARIO_TO_DRONE.get(scenario)
            if drone_type is None:
                continue
            
            # Etiket belirle
            if self.task == "drone_type":
                label = self.class_to_idx.get(drone_type)
            else:
                label = self.class_to_idx.get(fault_type)
            
            if label is None:
                continue
            
            # Bu klasördeki tüm görüntüleri ekle
            for img_path in folder.glob("*.jpg"):
                samples.append((img_path, label))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Görüntüyü yükle
        image = Image.open(img_path).convert("RGB")
        
        # Dönüşümleri uygula
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(is_train: bool = True) -> transforms.Compose:
    """Görüntü dönüşümlerini döndür"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_data_loaders(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Train/Val/Test veri yükleyicilerini oluştur"""
    
    # Veri setini böl
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Random seed ile tutarlı bölme
    generator = torch.Generator().manual_seed(Config.SEED)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # DataLoader'ları oluştur (Windows'ta num_workers=0 daha stabil)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# MODEL
# ============================================================================

class DroneViTClassifier(nn.Module):
    """Vision Transformer tabanlı sınıflandırıcı"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        
        # Pretrained ViT modelini yükle
        self.vit = timm.create_model(
            Config.VIT_MODEL,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        logger.info(f"ViT model yüklendi: {Config.VIT_MODEL}")
        logger.info(f"Sınıf sayısı: {num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)
    
    def freeze_backbone(self):
        """Backbone katmanlarını dondur (sadece head eğitimi için)"""
        for param in self.vit.parameters():
            param.requires_grad = False
        # Son sınıflandırma katmanını aç
        for param in self.vit.head.parameters():
            param.requires_grad = True
        logger.info("Backbone donduruldu, sadece head eğitilecek")
    
    def unfreeze_backbone(self):
        """Tüm katmanları eğitime aç"""
        for param in self.vit.parameters():
            param.requires_grad = True
        logger.info("Tüm katmanlar eğitime açıldı")


# ============================================================================
# EĞİTİM FONKSİYONLARI
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Tek bir epoch eğitimi"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # İstatistikler
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / total
    val_acc = correct / total
    
    return val_loss, val_acc


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    classes: List[str],
    device: torch.device,
    save_path: Optional[Path] = None
) -> Dict:
    """Model değerlendirmesi ve detaylı metrikler"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=classes)
    
    logger.info("\n" + "="*50)
    logger.info("TEST SONUÇLARI")
    logger.info("="*50)
    logger.info("\n" + report_text)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    if save_path:
        # Confusion matrix görselleştirme
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Tahmin')
        plt.ylabel('Gerçek')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path / 'confusion_matrix.png', dpi=150)
        plt.close()
        
        logger.info(f"Confusion matrix kaydedildi: {save_path / 'confusion_matrix.png'}")
    
    return {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'accuracy': report['accuracy']
    }


def train_model(
    task: str,
    num_epochs: int = Config.NUM_EPOCHS,
    learning_rate: float = Config.LEARNING_RATE
) -> Tuple[nn.Module, Dict]:
    """
    Model eğitimi ana fonksiyonu
    
    Args:
        task: "drone_type" veya "fault_type"
        num_epochs: Epoch sayısı
        learning_rate: Öğrenme oranı
    
    Returns:
        Eğitilmiş model ve eğitim geçmişi
    """
    logger.info("="*60)
    logger.info(f"MODEL EĞİTİMİ BAŞLIYOR: {task.upper()}")
    logger.info("="*60)
    
    # Cihaz kontrolü
    device = Config.DEVICE
    logger.info(f"Kullanılan cihaz: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Çıktı dizini oluştur
    output_dir = Config.OUTPUT_DIR / task
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Veri seti oluştur
    train_transform = get_transforms(is_train=True)
    test_transform = get_transforms(is_train=False)
    
    # Eğitim için augmented dataset
    full_dataset = DroneDataset(Config.DATASET_PATH, transform=train_transform, task=task)
    classes = full_dataset.classes
    num_classes = len(classes)
    
    # DataLoader'ları oluştur
    train_loader, val_loader, test_loader = create_data_loaders(
        full_dataset,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO,
        batch_size=Config.BATCH_SIZE
    )
    
    # Test için transform değiştir (augmentation olmadan)
    # Not: random_split sonrası transform değiştiremediğimiz için
    # test sırasında aynı transform kullanılacak (deterministik olacak şekilde)
    
    # Model oluştur
    model = DroneViTClassifier(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # Loss ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Eğitim geçmişi
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    # Eğitim döngüsü
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 40)
        
        # Eğitim
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Geçmişi kaydet
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        logger.info(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            logger.info(f"✓ Yeni en iyi model! Val Acc: {val_acc*100:.2f}%")
    
    # En iyi modeli yükle
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Test değerlendirmesi
    test_results = evaluate_model(model, test_loader, classes, device, output_dir)
    
    # Eğitim grafiklerini kaydet
    plot_training_history(history, output_dir)
    
    # Model ve meta bilgileri kaydet
    save_model(model, task, classes, history, test_results, output_dir)
    
    logger.info(f"\n✓ {task} modeli eğitimi tamamlandı!")
    logger.info(f"  Test Accuracy: {test_results['accuracy']*100:.2f}%")
    logger.info(f"  Model kaydedildi: {output_dir}")
    
    return model, history


def plot_training_history(history: Dict, save_path: Path):
    """Eğitim geçmişini görselleştir"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss grafiği
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy grafiği
    axes[1].plot([x*100 for x in history['train_acc']], label='Train Acc', marker='o')
    axes[1].plot([x*100 for x in history['val_acc']], label='Val Acc', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_history.png', dpi=150)
    plt.close()
    
    logger.info(f"Eğitim grafikleri kaydedildi: {save_path / 'training_history.png'}")


def save_model(
    model: nn.Module,
    task: str,
    classes: List[str],
    history: Dict,
    test_results: Dict,
    output_dir: Path
):
    """Model ve meta bilgilerini kaydet"""
    
    # PyTorch model (.pth)
    model_path = output_dir / f"{task}_vit_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': classes,
        'num_classes': len(classes),
        'model_name': Config.VIT_MODEL,
        'image_size': Config.IMAGE_SIZE,
    }, model_path)
    logger.info(f"PyTorch model kaydedildi: {model_path}")
    
    # Meta bilgileri (JSON)
    meta = {
        'task': task,
        'classes': classes,
        'num_classes': len(classes),
        'model_name': Config.VIT_MODEL,
        'image_size': Config.IMAGE_SIZE,
        'training_config': {
            'batch_size': Config.BATCH_SIZE,
            'num_epochs': Config.NUM_EPOCHS,
            'learning_rate': Config.LEARNING_RATE,
            'weight_decay': Config.WEIGHT_DECAY,
        },
        'test_results': {
            'accuracy': test_results['accuracy'],
            'classification_report': test_results['classification_report']
        },
        'training_history': {
            'final_train_loss': history['train_loss'][-1],
            'final_train_acc': history['train_acc'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_acc': history['val_acc'][-1],
        },
        'created_at': datetime.now().isoformat()
    }
    
    meta_path = output_dir / f"{task}_model_meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"Meta bilgileri kaydedildi: {meta_path}")


# ============================================================================
# INFERENCE
# ============================================================================

class DroneClassifier:
    """
    Eğitilmiş modelleri kullanarak tahmin yapan sınıf.
    Mobil uygulamada kullanılacak inference pipeline'ı için temel.
    """
    
    def __init__(
        self,
        drone_type_model_path: Path,
        fault_type_model_path: Path,
        device: Optional[torch.device] = None
    ):
        self.device = device or Config.DEVICE
        
        # Drone tipi modeli yükle
        self.drone_model, self.drone_classes = self._load_model(drone_type_model_path)
        logger.info(f"Drone tipi modeli yüklendi: {self.drone_classes}")
        
        # Arıza tipi modeli yükle
        self.fault_model, self.fault_classes = self._load_model(fault_type_model_path)
        logger.info(f"Arıza tipi modeli yüklendi: {self.fault_classes}")
        
        # Transform
        self.transform = get_transforms(is_train=False)
    
    def _load_model(self, model_path: Path) -> Tuple[nn.Module, List[str]]:
        """Model yükle"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = DroneViTClassifier(
            num_classes=checkpoint['num_classes'],
            pretrained=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model, checkpoint['classes']
    
    def predict(self, image_path: str) -> Dict:
        """
        Tek bir görüntü için tahmin yap
        
        Returns:
            {
                'drone_type': str,
                'drone_type_confidence': float,
                'fault_type': str,
                'fault_type_confidence': float,
                'is_faulty': bool
            }
        """
        # Görüntüyü yükle ve dönüştür
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Drone tipi tahmini
            drone_output = self.drone_model(image_tensor)
            drone_probs = torch.softmax(drone_output, dim=1)
            drone_conf, drone_pred = torch.max(drone_probs, dim=1)
            drone_type = self.drone_classes[drone_pred.item()]
            
            # Arıza tipi tahmini
            fault_output = self.fault_model(image_tensor)
            fault_probs = torch.softmax(fault_output, dim=1)
            fault_conf, fault_pred = torch.max(fault_probs, dim=1)
            fault_type = self.fault_classes[fault_pred.item()]
        
        return {
            'drone_type': drone_type,
            'drone_type_confidence': drone_conf.item(),
            'fault_type': fault_type,
            'fault_type_confidence': fault_conf.item(),
            'is_faulty': fault_type.lower() != 'normal'
        }
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Birden fazla görüntü için tahmin yap"""
        return [self.predict(path) for path in image_paths]


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    """Ana eğitim pipeline'ı"""
    
    # Seed ayarla
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    # CUDA kontrolü
    logger.info("="*60)
    logger.info("DRONE MOTOR ARIZA TESPİTİ - VİT MODEL EĞİTİMİ")
    logger.info("="*60)
    
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA kullanılabilir")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA Version: {torch.version.cuda}")
    else:
        logger.warning("⚠ CUDA kullanılamıyor, CPU ile eğitim yapılacak")
    
    # Çıktı dizini oluştur
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. DRONE TİPİ MODELİ EĞİT
    logger.info("\n" + "="*60)
    logger.info("ADIM 1: DRONE TİPİ SINIFLANDIRMA MODELİ")
    logger.info("="*60)
    drone_model, drone_history = train_model(task="drone_type")
    
    # 2. ARIZA TİPİ MODELİ EĞİT
    logger.info("\n" + "="*60)
    logger.info("ADIM 2: ARIZA TİPİ SINIFLANDIRMA MODELİ")
    logger.info("="*60)
    fault_model, fault_history = train_model(task="fault_type")
    
    # SON ÖZET
    logger.info("\n" + "="*60)
    logger.info("EĞİTİM TAMAMLANDI!")
    logger.info("="*60)
    logger.info(f"\nKaydedilen modeller:")
    logger.info(f"  1. {Config.OUTPUT_DIR / 'drone_type' / 'drone_type_vit_model.pth'}")
    logger.info(f"  2. {Config.OUTPUT_DIR / 'fault_type' / 'fault_type_vit_model.pth'}")
    
    logger.info("\n✓ Modeller eğitildi ve kaydedildi!")
    
    # Test tahmin örneği
    logger.info("\n" + "="*60)
    logger.info("ÖRNEK TAHMİN TESTİ")
    logger.info("="*60)
    
    try:
        classifier = DroneClassifier(
            drone_type_model_path=Config.OUTPUT_DIR / "drone_type" / "drone_type_vit_model.pth",
            fault_type_model_path=Config.OUTPUT_DIR / "fault_type" / "fault_type_vit_model.pth"
        )
        
        # Test için örnek bir görüntü bul
        test_images = list(Config.DATASET_PATH.glob("*/*.jpg"))[:3]
        
        for img_path in test_images:
            result = classifier.predict(str(img_path))
            logger.info(f"\nGörüntü: {img_path.name}")
            logger.info(f"  Drone Tipi: {result['drone_type']} ({result['drone_type_confidence']*100:.1f}%)")
            logger.info(f"  Arıza Tipi: {result['fault_type']} ({result['fault_type_confidence']*100:.1f}%)")
            logger.info(f"  Arızalı mı?: {'Evet' if result['is_faulty'] else 'Hayır'}")
    
    except Exception as e:
        logger.error(f"Örnek tahmin testi sırasında hata: {e}")


if __name__ == "__main__":
    main()
