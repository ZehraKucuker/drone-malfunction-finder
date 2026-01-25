import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import AnalysisResult from './components/AnalysisResult';
import RecentAnalyses from './components/RecentAnalyses';
import { uploadAndAnalyze, getAllAnalyses } from './services/api';
import { ArrowRight, Loader2 } from 'lucide-react';

function App() {
  // State
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);
  const [recentAnalyses, setRecentAnalyses] = useState([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Geçmiş analizleri yükle
  const fetchRecentAnalyses = useCallback(async () => {
    setIsLoadingHistory(true);
    try {
      const response = await getAllAnalyses(8);
      if (response.success) {
        setRecentAnalyses(response.data);
      }
    } catch (error) {
      console.error('Geçmiş analizler yüklenemedi:', error);
    } finally {
      setIsLoadingHistory(false);
    }
  }, []);

  // Sayfa yüklendiğinde geçmiş analizleri getir
  useEffect(() => {
    fetchRecentAnalyses();
  }, [fetchRecentAnalyses]);

  // Dosya seçildiğinde
  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setAnalysisResult(null);
    setAnalysisError(null);
    
    // Otomatik analiz başlat
    handleAnalyze(file);
  };

  // Dosya temizle
  const handleClearFile = () => {
    setSelectedFile(null);
    setAnalysisResult(null);
    setAnalysisError(null);
    setUploadProgress(0);
  };

  // Analiz yap
  const handleAnalyze = async (file) => {
    if (!file) return;

    setIsUploading(true);
    setIsAnalyzing(true);
    setAnalysisError(null);
    setUploadProgress(0);

    try {
      const result = await uploadAndAnalyze(file, (progress) => {
        setUploadProgress(progress);
        if (progress === 100) {
          setIsUploading(false);
        }
      });

      if (result.success) {
        setAnalysisResult(result);
        // Listeyi güncelle
        fetchRecentAnalyses();
      } else {
        setAnalysisError(result.error || 'Analiz başarısız oldu');
      }
    } catch (error) {
      console.error('Analiz hatası:', error);
      setAnalysisError(
        error.response?.data?.error || 
        error.message || 
        'Analiz sırasında bir hata oluştu'
      );
    } finally {
      setIsUploading(false);
      setIsAnalyzing(false);
    }
  };

  const showAnalysisSection = selectedFile || isAnalyzing || analysisResult || analysisError;

  return (
    <div className="min-h-screen p-6 md:p-10">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <Header />

        {/* Ana İçerik */}
        <main>
          {/* Upload ve Analiz Bölümü */}
          <div className="relative">
            <div className={`grid gap-6 ${showAnalysisSection ? 'grid-cols-1 md:grid-cols-[1fr_auto_1fr]' : 'grid-cols-1 max-w-md'}`}>
              {/* Dosya Yükleme */}
              <FileUpload
                onFileSelect={handleFileSelect}
                isLoading={isUploading}
                selectedFile={selectedFile}
                onClear={handleClearFile}
              />

              {/* Ok */}
              {showAnalysisSection && (
                <div className="hidden md:flex items-center justify-center">
                  <ArrowRight className="w-12 h-12 text-drone-accent" />
                </div>
              )}

              {/* Analiz Sonucu */}
              {showAnalysisSection && (
                <AnalysisResult
                  result={analysisResult}
                  isLoading={isAnalyzing}
                  error={analysisError}
                />
              )}
            </div>
          </div>

          {/* Son Analizler */}
          <RecentAnalyses
            analyses={recentAnalyses}
            isLoading={isLoadingHistory}
            onRefresh={fetchRecentAnalyses}
          />
        </main>
      </div>
    </div>
  );
}

export default App;
