const express = require('express');
const router = express.Router();
const mlService = require('../services/mlService');
const { getConnection } = require('../config/database');

/**
 * @route   GET /api/health
 * @desc    Sistem sağlık durumu
 */
router.get('/', async (req, res) => {
    try {
        // RethinkDB bağlantı kontrolü
        const dbConnection = getConnection();
        const dbStatus = dbConnection && dbConnection.open ? 'connected' : 'disconnected';

        // ML servisi kontrolü
        const mlStatus = await mlService.checkHealth();

        const health = {
            status: 'ok',
            timestamp: new Date().toISOString(),
            services: {
                api: 'running',
                database: dbStatus,
                mlService: mlStatus ? 'running' : 'unavailable'
            }
        };

        // Eğer herhangi bir servis çalışmıyorsa status'u degraded yap
        if (dbStatus !== 'connected' || !mlStatus) {
            health.status = 'degraded';
        }

        res.json(health);
    } catch (error) {
        res.status(500).json({
            status: 'error',
            timestamp: new Date().toISOString(),
            error: error.message
        });
    }
});

/**
 * @route   GET /api/health/ml
 * @desc    ML servisi bilgileri
 */
router.get('/ml', async (req, res) => {
    try {
        const modelInfo = await mlService.getModelInfo();
        res.json({
            success: true,
            data: modelInfo
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

module.exports = router;
