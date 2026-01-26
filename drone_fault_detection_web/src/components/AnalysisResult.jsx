import React from 'react';
import { Loader2, AlertTriangle, CheckCircle, ArrowRight } from 'lucide-react';

const AnalysisResult = ({ result, isLoading, error }) => {
  if (isLoading) {
    return (
      <div className="card flex flex-col items-center justify-center min-h-[200px] animate-fade-in">
        <Loader2 className="w-12 h-12 text-drone-accent animate-spin mb-4" />
        <p className="text-xl text-gray-300">Analiz ediliyor ...</p>
        <p className="text-gray-500 text-sm mt-2">Ses dosyası işleniyor</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card flex flex-col items-center justify-center min-h-[200px] border-drone-red/50 animate-fade-in">
        <AlertTriangle className="w-12 h-12 text-drone-red mb-4" />
        <p className="text-xl text-drone-red">Hata Oluştu</p>
        <p className="text-gray-400 text-sm mt-2 text-center">{error}</p>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  const { data, mlResult } = result;
  const isFaulty = mlResult?.is_faulty || data?.fault_type !== 'Normal';
  const faultType = mlResult?.fault_type || data?.fault_type;
  const droneType = mlResult?.drone_type || data?.drone_type;
  const faultConfidence = Math.round((mlResult?.fault_confidence || data?.fault_acc || 0) * 100);
  const droneConfidence = Math.round((mlResult?.drone_confidence || data?.drone_acc || 0) * 100);

  // Arıza tip çevirisi
  const faultTypeLabels = {
    'Normal': 'Sağlıklı',
    'Balans': 'Balans Arızası Tespiti',
    'Miknatis': 'Mıknatıs Arızası Tespiti',
    'Pervane': 'Pervane Arızası Tespiti',
    'Rulman': 'Rulman Arızası Tespiti'
  };

  return (
    <div className={`card min-h-[200px] animate-fade-in ${
      isFaulty 
        ? 'border-drone-red/50' 
        : 'border-drone-green/50'
    }`}>
      <div className="flex flex-col items-center justify-center h-full gap-4">
        {/* Başlık */}
        <h3 className={`text-2xl font-bold ${
          isFaulty ? 'text-drone-red' : 'text-drone-green'
        }`}>
          {faultTypeLabels[faultType] || faultType}
        </h3>

        {/* Güven Skoru */}
        <div className="text-gray-300">
          <span className="font-bold">%{faultConfidence}</span> Güven Skoru
        </div>

        {/* Drone Tipi */}
        <p className="text-gray-400">
          %{droneConfidence} doğrulukla <span className="text-white font-bold">{droneType}</span> motor tipi tespit edildi.
        </p>

        {/* Durum İkonu */}
        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
          isFaulty 
            ? 'bg-drone-red/20' 
            : 'bg-drone-green/20'
        }`}>
          {isFaulty ? (
            <AlertTriangle className="w-5 h-5 text-drone-red" />
          ) : (
            <CheckCircle className="w-5 h-5 text-drone-green" />
          )}
        </div>
      </div>
    </div>
  );
};

const AnalysisSection = ({ selectedFile, isLoading, result, error, onAnalyze }) => {
  const showArrow = selectedFile || isLoading || result;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
      {/* Sol kısım - File Upload için placeholder */}
      <div className="hidden md:block" />
      
      {/* Ok ve Sonuç */}
      {showArrow && (
        <>
          <div className="hidden md:flex absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 items-center justify-center z-10">
            <ArrowRight className="w-12 h-12 text-drone-accent" />
          </div>
          <AnalysisResult result={result} isLoading={isLoading} error={error} />
        </>
      )}
    </div>
  );
};

export { AnalysisResult, AnalysisSection };
export default AnalysisResult;
