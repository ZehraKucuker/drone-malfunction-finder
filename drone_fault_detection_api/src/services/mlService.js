const fetch = require('node-fetch');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5000';

// AbortController for timeout
const AbortController = globalThis.AbortController || require('abort-controller');

/**
 * Python ML servisine ses dosyasını gönder ve analiz sonucunu al
 * @param {string} audioFilePath - Ses dosyasının yolu
 * @returns {Object} - Analiz sonucu
 */
async function analyzeAudio(audioFilePath) {
    try {
        const formData = new FormData();
        const fileStream = fs.createReadStream(audioFilePath);
        const fileName = path.basename(audioFilePath);
        
        formData.append('audio', fileStream, fileName);

        // 5 dakika timeout (model inference uzun sürebilir)
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 300000);

        console.log(`🔄 ML Servisine istek gönderiliyor: ${ML_SERVICE_URL}/analyze`);
        
        const response = await fetch(`${ML_SERVICE_URL}/analyze`, {
            method: 'POST',
            body: formData,
            headers: formData.getHeaders(),
            signal: controller.signal
        });

        clearTimeout(timeout);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`ML Service error: ${response.status} - ${errorText}`);
        }

        const result = await response.json();
        console.log('✅ ML Servisi yanıtı alındı:', result.fault_type);
        return result;
    } catch (error) {
        console.error('❌ ML Service hatası:', error.message);
        throw error;
    }
}

/**
 * ML servisinin sağlık durumunu kontrol et
 * @returns {boolean} - Servis aktif mi
 */
async function checkHealth() {
    try {
        const response = await fetch(`${ML_SERVICE_URL}/health`, {
            method: 'GET',
            timeout: 5000
        });
        
        return response.ok;
    } catch (error) {
        console.error('ML Service sağlık kontrolü hatası:', error.message);
        return false;
    }
}

/**
 * Model bilgilerini al
 * @returns {Object} - Model bilgileri
 */
async function getModelInfo() {
    try {
        const response = await fetch(`${ML_SERVICE_URL}/model-info`, {
            method: 'GET'
        });

        if (!response.ok) {
            throw new Error(`Model info error: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Model info hatası:', error.message);
        throw error;
    }
}

module.exports = {
    analyzeAudio,
    checkHealth,
    getModelInfo
};
