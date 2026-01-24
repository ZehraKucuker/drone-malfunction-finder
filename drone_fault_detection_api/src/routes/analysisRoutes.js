const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const router = express.Router();

const faultDetectionModel = require('../models/faultDetection');
const mlService = require('../services/mlService');

// Upload klasörünü oluştur
const uploadDir = process.env.UPLOAD_DIR || './uploads';
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

// Multer konfigürasyonu
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({
    storage: storage,
    limits: {
        fileSize: parseInt(process.env.MAX_FILE_SIZE) || 50 * 1024 * 1024 // 50MB
    },
    fileFilter: (req, file, cb) => {
        const allowedMimes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/x-wav', 'audio/wave'];
        if (allowedMimes.includes(file.mimetype) || file.originalname.match(/\.(wav|mp3|ogg|flac)$/i)) {
            cb(null, true);
        } else {
            cb(new Error('Sadece ses dosyaları yüklenebilir (wav, mp3, ogg, flac)'), false);
        }
    }
});

/**
 * @route   POST /api/analysis/upload
 * @desc    Ses dosyası yükle ve analiz et
 */
router.post('/upload', upload.single('audio'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ 
                success: false, 
                error: 'Ses dosyası yüklenmedi' 
            });
        }

        const filePath = req.file.path;
        const originalFileName = req.file.originalname;

        // Veritabanında kayıt oluştur (pending durumunda)
        const analysisRecord = await faultDetectionModel.createAnalysis({
            fileName: originalFileName,
            status: 'processing'
        });

        try {
            // ML servisine gönder
            const mlResult = await mlService.analyzeAudio(filePath);

            // Sonucu veritabanına kaydet
            const updatedRecord = await faultDetectionModel.updateAnalysis(analysisRecord.id, {
                drone_type: mlResult.drone_type,
                drone_acc: mlResult.drone_confidence,
                fault_type: mlResult.fault_type,
                fault_acc: mlResult.fault_confidence,
                inferenceTime: mlResult.inference_time,
                status: 'completed'
            });

            // Dosyayı sil (isteğe bağlı, yorumdan çıkarılabilir)
            // fs.unlinkSync(filePath);

            res.json({
                success: true,
                data: updatedRecord,
                mlResult: mlResult
            });

        } catch (mlError) {
            // ML hatası durumunda kaydı güncelle
            await faultDetectionModel.updateAnalysis(analysisRecord.id, {
                status: 'error'
            });

            throw mlError;
        }

    } catch (error) {
        console.error('Analiz hatası:', error);
        res.status(500).json({ 
            success: false, 
            error: error.message 
        });
    }
});

/**
 * @route   GET /api/analysis
 * @desc    Tüm analizleri getir
 */
router.get('/', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 100;
        const analyses = await faultDetectionModel.getAllAnalyses(limit);
        
        res.json({
            success: true,
            count: analyses.length,
            data: analyses
        });
    } catch (error) {
        console.error('Analiz listeleme hatası:', error);
        res.status(500).json({ 
            success: false, 
            error: error.message 
        });
    }
});

/**
 * @route   GET /api/analysis/stats
 * @desc    İstatistikleri getir
 */
router.get('/stats', async (req, res) => {
    try {
        const stats = await faultDetectionModel.getStatistics();
        
        res.json({
            success: true,
            data: stats
        });
    } catch (error) {
        console.error('İstatistik hatası:', error);
        res.status(500).json({ 
            success: false, 
            error: error.message 
        });
    }
});

/**
 * @route   GET /api/analysis/:id
 * @desc    ID'ye göre analiz getir
 */
router.get('/:id', async (req, res) => {
    try {
        const analysis = await faultDetectionModel.getAnalysisById(req.params.id);
        
        if (!analysis) {
            return res.status(404).json({ 
                success: false, 
                error: 'Analiz bulunamadı' 
            });
        }
        
        res.json({
            success: true,
            data: analysis
        });
    } catch (error) {
        console.error('Analiz getirme hatası:', error);
        res.status(500).json({ 
            success: false, 
            error: error.message 
        });
    }
});

/**
 * @route   DELETE /api/analysis/:id
 * @desc    Analiz kaydını sil
 */
router.delete('/:id', async (req, res) => {
    try {
        const deleted = await faultDetectionModel.deleteAnalysis(req.params.id);
        
        if (!deleted) {
            return res.status(404).json({ 
                success: false, 
                error: 'Analiz bulunamadı' 
            });
        }
        
        res.json({
            success: true,
            message: 'Analiz silindi'
        });
    } catch (error) {
        console.error('Analiz silme hatası:', error);
        res.status(500).json({ 
            success: false, 
            error: error.message 
        });
    }
});

module.exports = router;
