const { r, getConnection, dbConfig } = require('../config/database');
const { v4: uuidv4 } = require('uuid');

const TABLE_NAME = 'fault_detection';

/**
 * Yeni analiz kaydı oluştur
 * @param {Object} data - Analiz verileri
 * @returns {Object} - Oluşturulan kayıt
 */
async function createAnalysis(data) {
    const conn = getConnection();
    
    const record = {
        id: uuidv4(),
        fileName: data.fileName || '',
        drone_type: data.drone_type || null,
        drone_acc: data.drone_acc || 0,
        fault_type: data.fault_type || null,
        fault_acc: data.fault_acc || 0,
        inferenceTime: data.inferenceTime || 0,
        status: data.status || 'pending', // pending, processing, completed, error
        createdAt: r.now()
    };

    const result = await r.db(dbConfig.db)
        .table(TABLE_NAME)
        .insert(record, { returnChanges: true })
        .run(conn);

    return result.changes[0].new_val;
}

/**
 * Analiz kaydını güncelle
 * @param {string} id - Kayıt ID
 * @param {Object} data - Güncellenecek veriler
 * @returns {Object} - Güncellenen kayıt
 */
async function updateAnalysis(id, data) {
    const conn = getConnection();
    
    const updateData = {};
    if (data.drone_type !== undefined) updateData.drone_type = data.drone_type;
    if (data.drone_acc !== undefined) updateData.drone_acc = data.drone_acc;
    if (data.fault_type !== undefined) updateData.fault_type = data.fault_type;
    if (data.fault_acc !== undefined) updateData.fault_acc = data.fault_acc;
    if (data.inferenceTime !== undefined) updateData.inferenceTime = data.inferenceTime;
    if (data.status !== undefined) updateData.status = data.status;

    const result = await r.db(dbConfig.db)
        .table(TABLE_NAME)
        .get(id)
        .update(updateData, { returnChanges: true })
        .run(conn);

    return result.changes ? result.changes[0].new_val : null;
}

/**
 * Tüm analizleri getir (son tarihten ilk tarihe)
 * @param {number} limit - Maksimum kayıt sayısı
 * @returns {Array} - Analiz kayıtları
 */
async function getAllAnalyses(limit = 100) {
    const conn = getConnection();
    
    const cursor = await r.db(dbConfig.db)
        .table(TABLE_NAME)
        .orderBy(r.desc('createdAt'))
        .limit(limit)
        .run(conn);

    return await cursor.toArray();
}

/**
 * ID'ye göre analiz getir
 * @param {string} id - Kayıt ID
 * @returns {Object} - Analiz kaydı
 */
async function getAnalysisById(id) {
    const conn = getConnection();
    
    return await r.db(dbConfig.db)
        .table(TABLE_NAME)
        .get(id)
        .run(conn);
}

/**
 * Duruma göre analizleri getir
 * @param {string} status - Durum (pending, processing, completed, error)
 * @returns {Array} - Analiz kayıtları
 */
async function getAnalysesByStatus(status) {
    const conn = getConnection();
    
    const cursor = await r.db(dbConfig.db)
        .table(TABLE_NAME)
        .filter({ status })
        .orderBy(r.desc('createdAt'))
        .run(conn);

    return await cursor.toArray();
}

/**
 * Analiz kaydını sil
 * @param {string} id - Kayıt ID
 * @returns {boolean} - Silme başarılı mı
 */
async function deleteAnalysis(id) {
    const conn = getConnection();
    
    const result = await r.db(dbConfig.db)
        .table(TABLE_NAME)
        .get(id)
        .delete()
        .run(conn);

    return result.deleted > 0;
}

/**
 * İstatistikleri getir
 * @returns {Object} - İstatistikler
 */
async function getStatistics() {
    const conn = getConnection();
    
    const total = await r.db(dbConfig.db)
        .table(TABLE_NAME)
        .count()
        .run(conn);

    const byStatus = await r.db(dbConfig.db)
        .table(TABLE_NAME)
        .group('status')
        .count()
        .run(conn);

    const byFaultType = await r.db(dbConfig.db)
        .table(TABLE_NAME)
        .filter(r.row('status').eq('completed'))
        .group('fault_type')
        .count()
        .run(conn);

    const byDroneType = await r.db(dbConfig.db)
        .table(TABLE_NAME)
        .filter(r.row('status').eq('completed'))
        .group('drone_type')
        .count()
        .run(conn);

    return {
        total,
        byStatus: byStatus.reduce((acc, item) => {
            acc[item.group] = item.reduction;
            return acc;
        }, {}),
        byFaultType: byFaultType.reduce((acc, item) => {
            acc[item.group] = item.reduction;
            return acc;
        }, {}),
        byDroneType: byDroneType.reduce((acc, item) => {
            acc[item.group] = item.reduction;
            return acc;
        }, {})
    };
}

module.exports = {
    createAnalysis,
    updateAnalysis,
    getAllAnalyses,
    getAnalysisById,
    getAnalysesByStatus,
    deleteAnalysis,
    getStatistics
};
