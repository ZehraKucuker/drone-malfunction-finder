const r = require('rethinkdb');
require('dotenv').config();

const dbConfig = {
    host: process.env.RETHINKDB_HOST || 'localhost',
    port: parseInt(process.env.RETHINKDB_PORT) || 28015,
    db: process.env.RETHINKDB_DB || 'drone'
};

let connection = null;

/**
 * RethinkDB bağlantısını oluştur
 */
async function connect() {
    try {
        connection = await r.connect(dbConfig);
        console.log('✅ RethinkDB bağlantısı başarılı');
        return connection;
    } catch (error) {
        console.error('❌ RethinkDB bağlantı hatası:', error.message);
        throw error;
    }
}

/**
 * Veritabanı ve tabloları oluştur
 */
async function initializeDatabase() {
    try {
        const conn = await connect();
        
        // Database oluştur (yoksa)
        const dbList = await r.dbList().run(conn);
        if (!dbList.includes(dbConfig.db)) {
            await r.dbCreate(dbConfig.db).run(conn);
            console.log(`✅ '${dbConfig.db}' veritabanı oluşturuldu`);
        }

        // fault_detection tablosunu oluştur (yoksa)
        const tableList = await r.db(dbConfig.db).tableList().run(conn);
        if (!tableList.includes('fault_detection')) {
            await r.db(dbConfig.db).tableCreate('fault_detection', { primaryKey: 'id' }).run(conn);
            console.log('✅ fault_detection tablosu oluşturuldu');
            
            // İndeksler oluştur
            await r.db(dbConfig.db).table('fault_detection').indexCreate('createdAt').run(conn);
            await r.db(dbConfig.db).table('fault_detection').indexCreate('status').run(conn);
            console.log('✅ İndeksler oluşturuldu');
        }

        return conn;
    } catch (error) {
        console.error('❌ Veritabanı başlatma hatası:', error.message);
        throw error;
    }
}

/**
 * Mevcut bağlantıyı al
 */
function getConnection() {
    return connection;
}

/**
 * Bağlantıyı kapat
 */
async function closeConnection() {
    if (connection) {
        await connection.close();
        console.log('🔌 RethinkDB bağlantısı kapatıldı');
    }
}

module.exports = {
    r,
    connect,
    initializeDatabase,
    getConnection,
    closeConnection,
    dbConfig
};
