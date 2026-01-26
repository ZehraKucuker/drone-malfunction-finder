import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileAudio, X, Loader2 } from 'lucide-react';

const FileUpload = ({ onFileSelect, isLoading, selectedFile, onClear }) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
    },
    maxFiles: 1,
    disabled: isLoading
  });

  return (
    <div className="relative">
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-xl p-8
          transition-all duration-300 cursor-pointer
          flex flex-col items-center justify-center min-h-[200px]
          ${isDragActive 
            ? 'border-drone-accent bg-drone-accent/10' 
            : 'border-drone-border hover:border-drone-accent/50 bg-drone-card/50'
          }
          ${isLoading ? 'pointer-events-none opacity-70' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        {isLoading ? (
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="w-12 h-12 text-drone-accent animate-spin" />
            <span className="text-gray-400">Yükleniyor...</span>
          </div>
        ) : selectedFile ? (
          <div className="flex flex-col items-center gap-3">
            <FileAudio className="w-12 h-12 text-drone-accent" />
            <div className="text-center">
              <p className="text-white font-medium">{selectedFile.name}</p>
              <p className="text-gray-400 text-sm">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <div className="p-4 rounded-full bg-drone-card border border-drone-border">
              <Upload className="w-8 h-8 text-drone-accent" />
            </div>
            <div className="text-center">
              <p className="text-white font-medium">Ses Yükle</p>
              <p className="text-gray-400 text-sm mt-1">
                {isDragActive 
                  ? 'Dosyayı bırakın...' 
                  : 'Sürükleyin veya tıklayın'
                }
              </p>
            </div>
          </div>
        )}
      </div>
      
      {selectedFile && !isLoading && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onClear();
          }}
          className="absolute top-2 right-2 p-1 rounded-full bg-drone-card border border-drone-border
                     hover:bg-drone-red/20 hover:border-drone-red transition-all"
        >
          <X className="w-4 h-4 text-gray-400 hover:text-drone-red" />
        </button>
      )}
    </div>
  );
};

export default FileUpload;
