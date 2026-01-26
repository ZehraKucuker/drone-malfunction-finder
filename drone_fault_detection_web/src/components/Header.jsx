import React from 'react';
import AudioWaveIcon from './AudioWaveIcon';

const Header = () => {
  return (
    <header className="flex items-center gap-4 mb-8">
      <div className="bg-drone-card border border-drone-border rounded-xl p-4">
        <AudioWaveIcon className="w-10 h-10 text-drone-accent" />
      </div>
      <div>
        <h1 className="text-3xl font-bold text-white">
          Drone<span className="text-drone-accent">AI</span> Sense
        </h1>
        <p className="text-gray-400 text-sm">
          Drone Seslerinizi analiz Edin ...
        </p>
      </div>
    </header>
  );
};

export default Header;
