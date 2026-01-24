const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const { initializeDatabase, closeConnection } = require('./config/database');
const analysisRoutes = require('./routes/analysisRoutes');
const healthRoutes = require('./routes/healthRoutes');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors({
    origin: ['http://localhost:3000', 'http://localhost:5173', 'http://127.0.0.1:3000'],
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Static files (uploads)
app.use('/uploads', express.static(path.join(__dirname, '../uploads')));

// Routes
app.use('/api/analysis', analysisRoutes);
app.use('/api/health', healthRoutes);

// Root route
app.get('/', (req, res) => {
    res.json({
        message: 'DroneAI Sense API',
        version: '1.0.0',
        endpoints: {
            analysis: '/api/analysis',
            health: '/api/health'
        }
    });
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Error:', err);
    
    if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(400).json({
            success: false,
            error: 'Dosya boyutu çok büyük. Maksimum 50MB yüklenebilir.'
        });
    }
    
    res.status(500).json({
        success: false,
        error: err.message || 'Sunucu hatası'
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        success: false,
        error: 'Endpoint bulunamadı'
    });
});

// Graceful shutdown
process.on('SIGINT', async () => {
    console.log('\n🛑 Sunucu kapatılıyor...');
    await closeConnection();
    process.exit(0);
});

process.on('SIGTERM', async () => {
    console.log('\n🛑 Sunucu kapatılıyor...');
    await closeConnection();
    process.exit(0);
});

// Start server
async function startServer() {
    try {
        // Veritabanı başlat
        await initializeDatabase();
        
        // Sunucuyu başlat
        app.listen(PORT, () => {
            console.log(`
╔══════════════════════════════════════════════════════╗
║          DroneAI Sense API Server                    ║
╠══════════════════════════════════════════════════════╣
║  🚀 Server running on http://localhost:${PORT}          ║
║  📊 Database: RethinkDB (drone)                      ║
║  🔧 Environment: ${process.env.NODE_ENV || 'development'}                        ║
╚══════════════════════════════════════════════════════╝
            `);
        });
    } catch (error) {
        console.error('❌ Sunucu başlatma hatası:', error);
        process.exit(1);
    }
}

startServer();
