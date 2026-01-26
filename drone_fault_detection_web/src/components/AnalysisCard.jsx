import React from 'react';
import { AlertTriangle, CheckCircle } from 'lucide-react';

const AnalysisCard = ({ analysis }) => {
  const {
    drone_type,
    drone_acc,
    fault_type,
    fault_acc,
    createdAt
  } = analysis;

  const isFaulty = fault_type && fault_type !== 'Normal';

  // Doğruluk yüzdesini formatla
  const formatAccuracy = (acc) => {
    if (acc === undefined || acc === null) return null;
    return Math.round(acc * 100);
  };

  // Tarih formatla
  const formatDate = (dateStr) => {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    const day = date.getDate().toString().padStart(2, '0');
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const year = date.getFullYear();
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `${day}.${month}.${year} - ${hours}.${minutes}`;
  };

  // Arıza tip çevirisi
  const faultTypeLabels = {
    'Normal': 'Sağlıklı',
    'Balans': 'Balans Arıza Tipi',
    'Miknatis': 'Mıknatıs Arıza Tipi',
    'Pervane': 'Pervane Arıza Tipi',
    'Rulman': 'Rulman Arıza Tipi'
  };

  return (
    <div className={`card relative animate-slide-in ${
      isFaulty 
        ? 'hover:border-drone-red/50' 
        : 'hover:border-drone-green/50'
    }`}>
      {/* Durum Göstergesi */}
      <div className="absolute top-4 right-4">
        <div className={`status-dot ${isFaulty ? 'status-faulty' : 'status-healthy'}`} />
      </div>

      {/* İçerik */}
      <div className="space-y-3">
        {/* Motor Tipi */}
        <div>
          <p className="text-gray-500 text-xs uppercase tracking-wider">Motor Tipi</p>
          <div className="flex items-center justify-between">
            <p className="text-white font-semibold">{drone_type || '-'}</p>
            {formatAccuracy(drone_acc) !== null && (
              <span className="text-drone-accent text-xs font-medium">
                %{formatAccuracy(drone_acc)}
              </span>
            )}
          </div>
        </div>

        {/* Arıza Durumu */}
        <div>
          <p className="text-gray-500 text-xs uppercase tracking-wider">Arıza Durumu</p>
          <div className="flex items-center justify-between">
            <p className={`font-semibold ${
              isFaulty ? 'text-drone-red' : 'text-drone-green'
            }`}>
              {faultTypeLabels[fault_type] || fault_type || '-'}
            </p>
            {formatAccuracy(fault_acc) !== null && (
              <span className={`text-xs font-medium ${
                isFaulty ? 'text-drone-red/80' : 'text-drone-green/80'
              }`}>
                %{formatAccuracy(fault_acc)}
              </span>
            )}
          </div>
        </div>

        {/* Analiz Tarihi */}
        <div>
          <p className="text-gray-500 text-xs uppercase tracking-wider">Analiz Tarihi</p>
          <p className="text-gray-300 text-sm">{formatDate(createdAt)}</p>
        </div>
      </div>
    </div>
  );
};

export default AnalysisCard;
