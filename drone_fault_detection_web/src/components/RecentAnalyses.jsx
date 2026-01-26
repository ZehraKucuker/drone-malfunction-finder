import React from 'react';
import AnalysisCard from './AnalysisCard';
import { History, RefreshCw } from 'lucide-react';

const RecentAnalyses = ({ analyses, isLoading, onRefresh }) => {
  return (
    <section className="mt-12">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-white flex items-center gap-2">
          <History className="w-5 h-5 text-drone-accent" />
          En Son Analizler
        </h2>
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="p-2 rounded-lg bg-drone-card border border-drone-border
                     hover:border-drone-accent/50 transition-all disabled:opacity-50"
          title="Yenile"
        >
          <RefreshCw className={`w-4 h-4 text-gray-400 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {analyses.length === 0 ? (
        <div className="card text-center py-12">
          <p className="text-gray-400">Henüz analiz yapılmamış</p>
          <p className="text-gray-500 text-sm mt-1">
            İlk analizinizi yapmak için ses dosyası yükleyin
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {analyses.map((analysis, index) => (
            <div
              key={analysis.id}
              style={{ animationDelay: `${index * 50}ms` }}
            >
              <AnalysisCard analysis={analysis} />
            </div>
          ))}
        </div>
      )}
    </section>
  );
};

export default RecentAnalyses;
