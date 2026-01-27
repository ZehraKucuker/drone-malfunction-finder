import axios from 'axios';
// ya varsayılan vite api url ya da basic path olan /api kullanılması
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Ses dosyasını yükle ve analiz et
 * @param {File} audioFile - Ses dosyası
 * @param {Function} onProgress - İlerleme callback'i
 * @returns {Promise} - Analiz sonucu
 */
export const uploadAndAnalyze = async (audioFile, onProgress = null) => {
  const formData = new FormData();
  formData.append('audio', audioFile);

  const config = {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  };

  if (onProgress) {
    config.onUploadProgress = (progressEvent) => {
      const percentCompleted = Math.round(
        (progressEvent.loaded * 100) / progressEvent.total
      );
      onProgress(percentCompleted);
    };
  }

  const response = await api.post('/analysis/upload', formData, config);
  return response.data;
};

/**
 * Tüm analizleri getir
 * @param {number} limit - Maksimum kayıt sayısı
 * @returns {Promise} - Analiz listesi
 */
export const getAllAnalyses = async (limit = 100) => {
  const response = await api.get(`/analysis?limit=${limit}`);
  return response.data;
};

/**
 * ID'ye göre analiz getir
 * @param {string} id - Analiz ID
 * @returns {Promise} - Analiz detayı
 */
export const getAnalysisById = async (id) => {
  const response = await api.get(`/analysis/${id}`);
  return response.data;
};

/**
 * Analiz sil
 * @param {string} id - Analiz ID
 * @returns {Promise} - Silme sonucu
 */
export const deleteAnalysis = async (id) => {
  const response = await api.delete(`/analysis/${id}`);
  return response.data;
};

/**
 * İstatistikleri getir
 * @returns {Promise} - İstatistikler
 */
export const getStatistics = async () => {
  const response = await api.get('/analysis/stats');
  return response.data;
};

/**
 * Sistem sağlık durumu
 * @returns {Promise} - Sağlık durumu
 */
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;
